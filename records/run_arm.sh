#!/bin/bash
# Phase-4 arm launcher:  run_arm.sh <O|N|C> <steps> [extra p4_train.py args...]
#
#   experiments/p4/run_arm.sh O 8000          # SPEC §5 pilot, control arm
#   experiments/p4/run_arm.sh N 150000        # the winner continued to H
#
# What this does and why it is a systemd transient unit rather than nohup:
#
#  * The run outlives the terminal (Linger=yes is set for this user), survives
#    an ssh drop, and is stoppable/inspectable by name months later - which is
#    what Phase 1 needed `trainctl` for. A nohup'd process on a shared box is a
#    process nobody can safely identify later.
#  * Restart=on-failure with a LONG RestartSec: the A5000 is shared, so the
#    honest failure mode is "the card was busy", and the unit should come back
#    later rather than spin. A clean exit (the step target, or a REGISTERED
#    stop - exit 4) must stay stopped, which is why it is on-failure and not
#    always. Exit 4 is listed in SuccessExitStatus for exactly that reason: a
#    watchdog stop, a 20k-gate failure or a plateau stop is a protocol OUTCOME,
#    not a crash to be retried.
#  * StartLimitIntervalSec=0: a GPU-busy refusal must be retryable forever; a
#    burst limit would park the run permanently while the co-tenant works.
#
# GPU discipline (SPEC §0, shared-box rules):
#  * CUDA_VISIBLE_DEVICES is pinned to the A5000 UUID. The A6000 belongs to a
#    co-tenant and is off-limits; p4_common.pin_gpu() asserts this again in
#    process, so a wrong pin dies at second zero instead of spending someone
#    else's card.
#  * The unit does not start training until the A5000 is genuinely FREE - same
#    rule and same thresholds as experiments/p3/wait_gpu_then.sh: >= 19,500 MiB
#    free AND zero other compute processes on that card. The wait loop lives
#    INSIDE the unit so the launcher returns immediately.
#  * Nice/CPUWeight/IOWeight yield CPU and disk to the co-tenant, never the GPU.
#
# Resume: if the run directory already holds a checkpoint, --resume auto is
# added, so a restarted unit continues the SAME sample stream and LR curve
# (SPEC §4 resume semantics) instead of starting a second, incomparable run.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

ARM="${1:?usage: run_arm.sh <arm name from configs/p4_ctc.yaml paths.arms> <steps> [extra args...]}"
STEPS="${2:?usage: run_arm.sh <arm> <steps> [extra args...]}"
shift 2
"$PWD/.venv/bin/python" - "$ARM" <<'PY' || { echo "arm $ARM is not in configs/p4_ctc.yaml paths.arms" >&2; exit 2; }
import sys, yaml; sys.exit(0 if sys.argv[1] in yaml.safe_load(open("configs/p4_ctc.yaml"))["paths"]["arms"] else 1)
PY

RUN_NAME="${P4_RUN_NAME:-pilot_${ARM}}"
RUN_DIR="$REPO/experiments/p4/runs/$RUN_NAME"
LOG="$RUN_DIR/train.log"
# Bland unit name on purpose: nothing in `systemctl --user list-units` on this
# shared box should say what is training or from where (same reasoning as
# ops/systemd/worker.service and the "6mn lmu" process title).
UNIT="${P4_UNIT:-worker-${ARM,,}}"
UNIT="${UNIT//+/-}"
# Card selection (owner amendment 2026-09-06): P4_CARD=a6000 runs a PILOT arm on the A6000 (20 GB cap, watchdog in
# config); the main run and the canonical O control stay on the A5000.
if [ "${P4_CARD:-a5000}" = "a6000" ]; then UUID=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda; CARD=A6000; ALLOW="-p Environment=P4_ALLOW_A6000=1"; else UUID=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567; CARD=A5000; ALLOW=""; fi
PY="$REPO/.venv/bin/python"
mkdir -p "$RUN_DIR"

# The resume decision is made INSIDE the unit, at every start, not here at
# unit-WRITE time. Until 2026-09-08 it was baked in here: a unit written before
# the first checkpoint existed carried an empty $RESUME forever, so restarting
# it later would have started the run again from update 0 over its own
# artefacts. p4_train.py now refuses that outright; this makes the unit do the
# right thing rather than merely fail safely.
if compgen -G "$RUN_DIR/ckpt/step_*.pt" > /dev/null; then
  echo "run_arm: existing checkpoint found -> the unit will resume automatically"
fi

# The wait loop and the trainer are one bash -c so that systemd owns both: a
# `systemctl --user stop` during the wait kills the wait, not a detached child.
read -r -d '' INNER <<EOF || true
L="$RUN_DIR/wait_gpu.log"
echo "\$(date -u +%FT%TZ) waiting for the $CARD: arm $ARM steps $STEPS" >> "\$L"
while :; do
  free=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $UUID 2>/dev/null | head -1)
  procs=\$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
  if [ -n "\$free" ] && [ "\$free" -ge ${P4_NEED_MIB:-12000} ]; then
    echo "\$(date -u +%FT%TZ) $CARD free (\${free} MiB, 0 procs) -> starting" >> "\$L"
    RESUME=""
    if compgen -G "$RUN_DIR/ckpt/step_*.pt" > /dev/null; then RESUME="--resume auto"; fi
    # IMPORT SMOKE (workflow wf_2ba85df0, 2026-09-09). This unit execs the trainer
    # from the WORKING TREE at every start, and systemd retries on failure every
    # 10 min with StartLimitIntervalSec=0. So any edit to tools/p4/*.py is armed to
    # load involuntarily on the next crash, and a half-applied or broken edit turns
    # into a SILENT indefinite relaunch loop: the process dies at import, no ABORT
    # record is written, the GPU sits idle and it looks like a hardware problem.
    # Convert that into one recorded terminal ABORT a human will see.
    if ! $PY -c "import sys; sys.path.insert(0, '$REPO'); import tools.p4.p4_common, tools.p4.p4_train" 2>>"\$L"; then
      echo "\$(date -u +%FT%TZ) IMPORT SMOKE FAILED - refusing to exec the trainer" >> "\$L"
      $PY - <<'PYABORT' >> "\$L" 2>&1
import json, time
from pathlib import Path
Path("$RUN_DIR/ABORT").write_text(json.dumps({
    "abort": "import smoke failed",
    "detail": ("tools/p4/p4_common.py or tools/p4/p4_train.py does not import; the trainer was "
               "NOT started. This is usually a half-applied edit. Fix the tree, then delete "
               "$RUN_DIR/ABORT to allow the supervisor to restart."),
    "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}) + "\n", encoding="utf-8")
print("wrote ABORT: import smoke failed")
PYABORT
      exit 3
    fi
    echo "\$(date -u +%FT%TZ) launching with \${RESUME:-no --resume (fresh run)}" >> "\$L"
    exec $PY $REPO/tools/p4/p4_train.py --arm $ARM --run-name $RUN_NAME \
         --steps $STEPS \$RESUME $* >> "$LOG" 2>&1
  fi
  sleep 60
done
EOF

# P4_DRY=1 prints the unit's command instead of starting it. A launcher for a
# 150,000-step run must be testable without spending a GPU-hour to find a typo.
if [ "${P4_DRY:-0}" = "1" ]; then
  echo "DRY unit=$UNIT run_dir=$RUN_DIR"
  echo "$INNER"
  exit 0
fi

systemd-run --user --unit="$UNIT" --collect \
  -p Type=exec -p KillMode=mixed \
  -p Restart=${P4_RESTART:-on-failure} -p RestartSec=10min -p StartLimitIntervalSec=0 \
  -p "SuccessExitStatus=0 4" \
  -p WorkingDirectory="$REPO" \
  -p Nice=10 -p CPUWeight=50 -p IOWeight=50 -p UMask=0027 \
  -p TimeoutStopSec=15min -p KillSignal=SIGTERM -p SendSIGKILL=yes \
  -p Environment=PYTHONUNBUFFERED=1 \
  -p Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -p Environment=CUDA_VISIBLE_DEVICES=$UUID $ALLOW \
  -p Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p Environment=OMP_NUM_THREADS=1 -p Environment=MKL_NUM_THREADS=1 \
  -p Environment=OPENBLAS_NUM_THREADS=1 \
  -p Environment=TOKENIZERS_PARALLELISM=false \
  -p Environment=HF_HOME=/mnt/sdb/arafat/.cache/huggingface \
  -p Environment=TMPDIR=/mnt/sdb/arafat/.tmp-asr \
  /bin/bash -c "$INNER"

echo "run_arm: unit=$UNIT run=$RUN_NAME arm=$ARM steps=$STEPS log=$LOG"
echo "  status : systemctl --user status $UNIT"
echo "  stop   : systemctl --user stop $UNIT     (checkpoint-safe; resume continues)"
echo "  watch  : experiments/p4/watch_p4.sh $RUN_NAME"
