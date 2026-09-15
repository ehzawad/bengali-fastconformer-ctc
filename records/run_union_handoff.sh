#!/bin/bash
# SPEC §8m execution order, automated at the hand-off (owner: "training must complete on its own"):
#   all three single-arm verdicts in -> build exactly ONE union U = O ∪ passing buckets (--auto refuses while any arm is
#   undecided) -> if U equals a tested single arm the tool refuses and that arm's own 8k state is resumed to H;
#   else register U in the config and launch its ONE 8k pilot. The union's admission (+0.20 rule) and the resume to 20k
#   are decided by a human when the pilot ends; this script never goes past launching the registered next step.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=experiments/p4/union_handoff.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
say "union hand-off armed; waiting for experiments/p4/admission_vaani_m4.json"
until [ -f experiments/p4/admission_vaani_m4.json ]; do sleep 60; done
sleep 5
# Council I / H preflight: the code that launches must be the code that passed the suite.
py=$(sha256sum tools/p4/p4_train.py | cut -c1-64); pc=$(sha256sum tools/p4/p4_common.py | cut -c1-64)
ok=$(python3 -c "import json,sys;t=json.load(open('experiments/p4/tests/results.json'));c=t['code_sha256'];print(int(c['p4_train.py']=='$py' and c['p4_common.py']=='$pc' and t['failed']==0))")
[ "$ok" = 1 ] || { say "REFUSING: trainer/common sha256 differ from tests/results.json or the suite has failures -> no launch"; exit 5; }
say "preflight OK: p4_train.py ${py:0:12} p4_common.py ${pc:0:12} match the passing test record"
passing=$(python3 -c "
import json
p=[b for b in ('macro_filtered','madasr','vaani') if json.load(open(f'experiments/p4/admission_{b}_m4.json'))['ADMITTED']]
print(','.join(p))")
say "passing buckets: [$passing]"
if [ "$passing" = "macro_filtered" ]; then
  say "U == O+macro_filtered (a tested single arm): no union pilot; resuming MaCro's own 8k state to H=150,000 (SPEC §8m)"
  P4_AFTER= nohup experiments/p4/run_supervised.sh 'O+macro_filtered' 150000 'pilot_O+macro_filtered' a5000 >> experiments/p4/runs/pilot_O+macro_filtered/supervisor.out 2>&1 < /dev/null &
  say "continuation supervisor launched for pilot_O+macro_filtered (pid $!)"; exit 0
fi
say "building U = O ∪ {$passing} under the frozen rules (byte-level derivation from arm_N)"
nice -n 15 ionice -c3 .venv/bin/python tools/p4/p4_union_build.py --auto --name U --nvme >> "$L" 2>&1; rc=$?
if [ $rc -ne 0 ]; then say "union build exited $rc -> NOT launching; a human must look (log above)"; exit 6; fi
[ -s experiments/p4/union/arm_U.nvme.jsonl ] || { say "arm_U.nvme.jsonl missing/empty -> NOT launching"; exit 6; }
rows=$(wc -l < experiments/p4/union/arm_U.jsonl); nrows=$(wc -l < experiments/p4/union/arm_U.nvme.jsonl)
[ "$rows" = "$nrows" ] || { say "row count mismatch arm_U $rows vs nvme $nrows -> NOT launching"; exit 6; }
python3 - <<'PY'
import re
p='configs/p4_ctc.yaml'; s=open(p).read()
if re.search(r'^\s+U:\s', s, re.M): print('config already has U')
else:
    s=s.replace('    N: experiments/p4/corpus/arm_N.nvme.jsonl\n', '    N: experiments/p4/corpus/arm_N.nvme.jsonl\n    U: experiments/p4/union/arm_U.nvme.jsonl        # council G/§8m union, built by p4_union_build.py at the hand-off\n', 1)
    open(p,'w').write(s); print('config: U registered')
PY
say "U registered: $rows rows; launching its ONE 8k pilot (SPEC §8m) as pilot_U on the A6000"
mkdir -p experiments/p4/runs/pilot_U
P4_AFTER= nohup experiments/p4/run_supervised.sh U 8000 pilot_U a6000 >> experiments/p4/runs/pilot_U/supervisor.out 2>&1 < /dev/null &
say "pilot_U supervisor launched (pid $!); its 8k admission and the +0.20 rule are a human decision when it ends"
