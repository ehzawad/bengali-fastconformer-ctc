#!/bin/bash

# ---------------------------------------------------------------------------
# SUPERSEDED AND PERMANENTLY DISARMED - 2026-09-09, council Q #8.
#
# This dispatcher belongs to the H=150,000 plan that SPEC 8v replaced. It still
# contains `rm -rf experiments/p4/runs/N_$CORPUS` followed by a launch of a
# 150,000-update supervisor. Run today it would DELETE production run state and
# start an unauthorized procedure at the wrong horizon. It is kept only as the
# record of how the init-N smoke and launch were performed on 2026-09-08.
#
# Refuse unless someone deliberately sets P4_RUN_SUPERSEDED_DISPATCHER=1.
if [ "${P4_RUN_SUPERSEDED_DISPATCHER:-0}" != 1 ]; then
  echo "run_post_union_N.sh is SUPERSEDED (SPEC 8v: H=120,000) and disarmed: it would" >&2
  echo "rm -rf a production run directory and launch a 150,000-update supervisor." >&2
  echo "The live run is N_O+macro_filtered_H120k, supervised already. Refusing." >&2
  exit 9
fi
# ---------------------------------------------------------------------------

# SPEC §8t (owner: NVIDIA-only; council L). After the union pilot ends:
#   1. U admission (frozen tool, m=4) + the +0.20 rule -> N's corpus (U or the MaCro fallback); pilot checkpoints are never an init
#   2. init-N qualification smoke on the A6000 (§8t.5): 40 updates -> checkpoint -> resume and replay 41-60 -> validation at 60,
#      collapse diagnostics on it; longest-clip profile; all disposable
#   3. launch N from update ZERO on the decided corpus, H=150,000, config p4_ctc_initN.yaml. No control completion. No A.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=experiments/p4/post_union_N.log; A6000=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
last_update() { [ -s experiments/p4/runs/$1/steps.jsonl ] && tail -n1 experiments/p4/runs/$1/steps.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["update"])' 2>/dev/null || echo 0; }
has_val() { python3 -c "
import json
try: print(int(any(json.loads(l).get('update')==$2 for l in open('experiments/p4/runs/$1/val.jsonl'))))
except Exception: print(0)"; }
ours_on_a6000() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep "$A6000" | while IFS=, read g p; do [ "$(ps -o user= -p ${p// /} 2>/dev/null)" = "$(id -un)" ] && echo x; done | wc -l; }
ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$A6000 P4_ALLOW_A6000=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false"

say "post-union N-only dispatcher armed (SPEC §8t): waiting for pilot_U 8,000"
until [ "$(last_update pilot_U)" -ge 8000 ] && [ "$(has_val pilot_U 8000)" = 1 ]; do
  if [ -f experiments/p4/runs/pilot_U/ABORT ]; then say "pilot_U ABORT -> a human must look"; exit 9; fi
  r=$(python3 -c 'import json,sys
try: d=json.load(open("experiments/p4/runs/pilot_U/STOP")); print(d.get("reason") or d.get("stop") or "x")
except Exception: print("")')
  if [ -n "$r" ] && [ "$r" != graceful_stop ]; then say "pilot_U terminal STOP ($r): a registered early pilot abort is terminal (§8t.1) -> a human must look"; exit 9; fi
  sleep 60; done
# ---------- 1. U admission + 0.20 rule
OUT=experiments/p4/admission_U_m4.json
nice -n 15 ionice -c3 .venv/bin/python tools/p4/p4_admission_test.py --arm experiments/p4/runs/pilot_U --control experiments/p4/runs/O_A5000 --step 8000 --out $OUT >> "$L" 2>&1
python3 - <<'PY' | tee -a "$L"
import json
u=json.load(open('experiments/p4/admission_U_m4.json'))
singles={b: json.load(open(f'experiments/p4/admission_{b}_m4.json')) for b in ('macro_filtered','madasr')}
best=min(v['S_arm_pp'] for v in singles.values() if v['ADMITTED']); ceiling=best+0.20
ok = bool(u['ADMITTED']) and (u['S_arm_pp'] <= ceiling + 1e-9)
print(f"UNION: ADMITTED={u['ADMITTED']} S={u['S_arm_pp']} dS={u['delta_S_pp']} sim={u['simultaneous_upper_pp']} ci={u['ci95']} | best single {best} -> ceiling {ceiling:.4f} | +0.20 rule {'PASS' if u['S_arm_pp']<=ceiling+1e-9 else 'FAIL'} | CORPUS FOR N = {'U' if ok else 'O+macro_filtered'}")
open('experiments/p4/union_decision.json','w').write(json.dumps({"union_admitted":u['ADMITTED'],"S_U":u['S_arm_pp'],"best_single_S":best,"ceiling":round(ceiling,4),"corpus":"U" if ok else "O+macro_filtered","policy":"SPEC 8t.1"},indent=1))
PY
CORPUS=$(python3 -c "import json;print(json.load(open('experiments/p4/union_decision.json'))['corpus'])")
say "corpus for N: $CORPUS (pilot checkpoints are never an initialisation, §8t.1)"
until ! systemctl --user is-active --quiet worker-pilot-u-a6000 && [ "$(ours_on_a6000)" = 0 ]; do sleep 30; done; sleep 20
# ---------- 2. init-N qualification smoke (§8t.5): 40 -> checkpoint -> resume 41-60 -> validation at 60
SMK=smoke_N_$CORPUS; rm -rf experiments/p4/runs/$SMK
say "init-N smoke on the A6000: 40 updates, checkpoint, resume 41-60, validation at 60 (corpus $CORPUS)"
env $ENV timeout 1500 nice -n 10 .venv/bin/python tools/p4/p4_train.py --arm $CORPUS --run-name $SMK --steps 40 --val-every 0 --config configs/p4_ctc_initN.yaml >> experiments/p4/runs/$SMK.smoke.log 2>&1; rc1=$?
env $ENV timeout 1500 nice -n 10 .venv/bin/python tools/p4/p4_train.py --arm $CORPUS --run-name $SMK --steps 60 --val-at 60 --val-limit 400 --resume auto --config configs/p4_ctc_initN.yaml >> experiments/p4/runs/$SMK.smoke.log 2>&1; rc2=$?
ok=$(python3 -c "
import json,math
try:
    st=[json.loads(l) for l in open('experiments/p4/runs/$SMK/steps.jsonl')]; m=json.load(open('experiments/p4/runs/$SMK/run_meta.json'))
    v=[json.loads(l) for l in open('experiments/p4/runs/$SMK/val.jsonl')]; log=open('experiments/p4/runs/$SMK.smoke.log',errors='replace').read()
    ups=[r['update'] for r in st]; fin=all(math.isfinite(r['loss']) for r in st)
    good = ($rc1==0 and $rc2==0 and ups==list(range(1,61)) and fin and m['init']=='N' and abs(m['sched']['lr_peak']-5e-4)<1e-12
            and 'resumed update=40' in log and 'resume handshake OK' in log and len(v)==1 and v[0]['update']==60
            and abs(st[-1]['lr']-(1e-5+(5e-4-1e-5)*59/2000))<1e-12)   # lr_at is 0-based: update 60 applied lr_at(59)
    print(int(good))
    print('SMOKE', 'rc',$rc1,$rc2,'updates',len(st),'finite',fin,'init',m['init'],'lr@60',st[-1]['lr'],'loss1',round(st[0]['loss'],3),'loss60',round(st[-1]['loss'],3),'val D',round(v[0]['panels']['D']['wer_frozen']*100,1),'empties',v[0]['panels']['D']['empty_rate'],'resumed@40','resumed update=40' in log)
except Exception as e: print(0); print('SMOKE error', e)" | tee -a "$L" | head -1)
[ "$ok" = 1 ] || { say "N SMOKE FAILED (rc $rc1/$rc2) -> NOT launching; a human must look (§8t.5)"; exit 7; }
.venv/bin/python tools/p4/p4_collapse_check.py --run $SMK --step 60 --panels D >> "$L" 2>&1
say "init-N smoke PASSED (engineering qualification only; its state is discarded). Collapse diagnostics at update 60 recorded (LR there is 2.47e-5; empties do not invalidate the recipe)."
# ---------- 3. launch N from update zero
say "launching init N on the A6000 from update ZERO: corpus $CORPUS, run N_$CORPUS, H=150,000, config p4_ctc_initN.yaml (SPEC §8t)"
mkdir -p experiments/p4/runs/N_$CORPUS   # council Q #8: the rm -rf was removed; a fresh run needs a NEW --run-name, never the deletion of an existing one
P4_AFTER= P4_ONLY_PREF=1 P4_EXTRA_ARGS="--config configs/p4_ctc_initN.yaml" setsid nohup experiments/p4/run_supervised.sh $CORPUS 150000 N_$CORPUS a6000 >> experiments/p4/runs/N_$CORPUS/supervisor.out 2>&1 < /dev/null & disown
say "N_$CORPUS supervisor launched (pid $!)"
