#!/bin/bash

# ---------------------------------------------------------------------------
# SUPERSEDED AND PERMANENTLY DISARMED - 2026-09-09, council R (SPEC 8w.9).
# Its sibling run_post_union_N.sh was disarmed on 2026-09-08; this one was
# missed. It belongs to the H=150,000 plan that SPEC 8v replaced, and still
# launches the withdrawn O_A6000 control and a 150,000-update supervisor.
# Kept only as the record of how that chain was run.
if [ "${P4_RUN_SUPERSEDED_DISPATCHER:-0}" != 1 ]; then
  echo "run_post_union.sh is SUPERSEDED (SPEC 8v: H=120,000, NVIDIA-only) and disarmed:" >&2
  echo "it would launch the withdrawn control and a 150,000-update supervisor." >&2
  echo "The live run is N_O+macro_filtered_H120k, already supervised. Refusing." >&2
  exit 9
fi
# ---------------------------------------------------------------------------

# SPEC §8s (owner: NVIDIA first). After the union pilot ends:
#   1. admission of U (frozen tool, m=4) + the +0.20 union rule vs the best passing single -> the corpus for BOTH procedures
#   2. pin A's 8k milestone (sha256) - A resumes from it AFTER N (§8s.1); A is NOT advanced now
#   3. GPU smoke of init N on the decided corpus (60 updates + validation + save/resume round-trip) - "no untested N path launches"
#   4. complete the O_A6000 control 4,038->8,000 on the A6000 (§8s.4 hardware qualification)
#   5. agreement check vs O_ref; PASS -> launch N on the A6000 to H=150,000; FAIL -> N waits for an eligible A5000 (no revert)
set -uo pipefail
cd "$(dirname "$0")/../.."
L=experiments/p4/post_union.log; A6000=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
last_update() { [ -s experiments/p4/runs/$1/steps.jsonl ] && tail -n1 experiments/p4/runs/$1/steps.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["update"])' 2>/dev/null || echo 0; }
has_val() { python3 -c "
import json
try: print(int(any(json.loads(l).get('update')==$2 for l in open('experiments/p4/runs/$1/val.jsonl'))))
except Exception: print(0)"; }
ours_on_a6000() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$A6000" ; }

say "post-union chain armed (SPEC §8s): waiting for pilot_U 8,000"
until [ "$(last_update pilot_U)" -ge 8000 ] && [ "$(has_val pilot_U 8000)" = 1 ]; do sleep 60; done
# ---------- 1. U admission + the +0.20 rule
OUT=experiments/p4/admission_U_m4.json
nice -n 15 ionice -c3 .venv/bin/python tools/p4/p4_admission_test.py --arm experiments/p4/runs/pilot_U --control experiments/p4/runs/O_A5000 --step 8000 --out $OUT >> "$L" 2>&1
python3 - <<'PY' | tee -a "$L"
import json
u=json.load(open('experiments/p4/admission_U_m4.json'))
singles={b: json.load(open(f'experiments/p4/admission_{b}_m4.json')) for b in ('macro_filtered','madasr')}
best=min(v['S_arm_pp'] for v in singles.values() if v['ADMITTED']); ceiling=best+0.20
ok = bool(u['ADMITTED']) and (u['S_arm_pp'] <= ceiling + 1e-9)
print(f"UNION: ADMITTED={u['ADMITTED']} S={u['S_arm_pp']} dS={u['delta_S_pp']} sim={u['simultaneous_upper_pp']} | best single {best} -> ceiling {ceiling:.4f} | +0.20 rule {'PASS' if u['S_arm_pp']<=ceiling+1e-9 else 'FAIL'} | CORPUS={'U' if ok else 'O+macro_filtered'}")
open('experiments/p4/union_decision.json','w').write(json.dumps({"union_admitted":u['ADMITTED'],"S_U":u['S_arm_pp'],"best_single_S":best,"ceiling":round(ceiling,4),"corpus":"U" if ok else "O+macro_filtered"},indent=1))
PY
CORPUS=$(python3 -c "import json;print(json.load(open('experiments/p4/union_decision.json'))['corpus'])")
ARUN=$([ "$CORPUS" = U ] && echo pilot_U || echo pilot_O+macro_filtered)
# ---------- 2. pin A's 8k state
until [ -f experiments/p4/runs/$ARUN/milestones/step_8000.pt ]; do sleep 20; done
say "corpus for BOTH procedures: $CORPUS; A's 8k state pinned: runs/$ARUN/milestones/step_8000.pt sha256 $(sha256sum experiments/p4/runs/$ARUN/milestones/step_8000.pt | cut -c1-16) (A resumes from it AFTER N, §8s.1)"
# wait for the pilot's worker to exit and the A6000 to be free of our processes
until ! systemctl --user is-active --quiet worker-pilot-u-a6000 && [ "$(ours_on_a6000)" = 0 ]; do sleep 30; done
sleep 20
# ---------- 3. GPU smoke of init N
SMK=smoke_N_$CORPUS; rm -rf experiments/p4/runs/$SMK
say "init-N GPU smoke on the A6000: 60 updates + validation, corpus $CORPUS"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$A6000 P4_ALLOW_A6000=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  timeout 1500 nice -n 10 .venv/bin/python tools/p4/p4_train.py --arm $CORPUS --run-name $SMK --steps 60 --val-at 60 --val-limit 300 --config configs/p4_ctc_initN.yaml >> experiments/p4/runs/$SMK.smoke.log 2>&1; rc=$?
ok1=$(python3 -c "
import json,math
try:
    st=[json.loads(l) for l in open('experiments/p4/runs/$SMK/steps.jsonl')]; m=json.load(open('experiments/p4/runs/$SMK/run_meta.json'))
    v=[json.loads(l) for l in open('experiments/p4/runs/$SMK/val.jsonl')]
    fin=all(math.isfinite(r['loss']) for r in st)
    print(int($rc==0 and len(st)==60 and fin and m['init']=='N' and abs(m['sched']['lr_peak']-5e-4)<1e-12 and len(v)==1 and v[0]['update']==60))
    print('SMOKE1', 'rc',$rc,'steps',len(st),'finite',fin,'init',m['init'],'lr_peak',m['sched']['lr_peak'],'loss0',round(st[0]['loss'],3),'loss60',round(st[-1]['loss'],3),'val D',round(v[0]['panels']['D']['wer_frozen']*100,1),'empties',v[0]['panels']['D']['empty_rate'])
except Exception as e: print(0); print('SMOKE1 error', e)" | tee -a "$L" | head -1)
[ "$ok1" = 1 ] || { say "N SMOKE FAILED (rc=$rc) -> NOT launching anything on the N path; a human must look (SPEC §8s: no untested N path launches)"; exit 7; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$A6000 P4_ALLOW_A6000=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  timeout 900 nice -n 10 .venv/bin/python tools/p4/p4_train.py --arm $CORPUS --run-name $SMK --steps 61 --val-at 60 --val-limit 300 --resume auto --config configs/p4_ctc_initN.yaml >> experiments/p4/runs/$SMK.smoke.log 2>&1; rc2=$?
ok2=$(python3 -c "
import json
try:
    st=[json.loads(l) for l in open('experiments/p4/runs/$SMK/steps.jsonl')]; log=open('experiments/p4/runs/$SMK.smoke.log',errors='replace').read()
    print(int($rc2==0 and st[-1]['update']==61 and 'resumed update=60' in log and 'RESUME-REPLAY update=60' in log))
except Exception as e: print(0)")
[ "$ok2" = 1 ] || { say "N SMOKE RESUME ROUND-TRIP FAILED (rc=$rc2) -> NOT launching; a human must look"; exit 7; }
say "init-N smoke PASSED: 60 updates finite, validation at 60 written, save/resume round-trip replayed the owed record"
# ---------- 4. O_A6000 control completion (hardware qualification, §8s.4)
say "launching the O_A6000 control completion 4,038 -> 8,000 on the A6000 (P4_ONLY_PREF: never migrates)"
P4_AFTER= P4_ONLY_PREF=1 setsid nohup experiments/p4/run_supervised.sh O 8000 O_A6000 a6000 >> experiments/p4/runs/O_A6000/supervisor.out 2>&1 < /dev/null & disown
until [ "$(last_update O_A6000)" -ge 8000 ] && [ "$(has_val O_A6000 8000)" = 1 ]; do
  if [ -f experiments/p4/runs/O_A6000/ABORT ]; then say "O_A6000 ABORT recorded; chain stops, a human must look"; exit 9; fi
  sleep 120; done
sleep 30
# ---------- 5. agreement -> N launch or wait
.venv/bin/python tools/p4/p4_card_agreement.py --step 8000 >> "$L" 2>&1; arc=$?
if [ $arc -eq 0 ]; then
  until [ "$(ours_on_a6000)" = 0 ]; do sleep 30; done; sleep 20
  say "card qualification PASSED -> launching init N on the A6000: corpus $CORPUS, run N_$CORPUS, H=150,000, config p4_ctc_initN.yaml (SPEC §8s)"
  mkdir -p experiments/p4/runs/N_$CORPUS
  P4_AFTER= P4_ONLY_PREF=1 P4_EXTRA_ARGS="--config configs/p4_ctc_initN.yaml" setsid nohup experiments/p4/run_supervised.sh $CORPUS 150000 N_$CORPUS a6000 >> experiments/p4/runs/N_$CORPUS/supervisor.out 2>&1 < /dev/null & disown
  say "N_$CORPUS supervisor launched (pid $!)"
else
  say "CARD QUALIFICATION $([ $arc -eq 2 ] && echo FAILED || echo NOT EVALUABLE) (rc=$arc) -> N is NOT launched; per SPEC §8s.4 N waits for an eligible A5000 and the sequence does not revert to A-first. A human must look."
  exit 8
fi
