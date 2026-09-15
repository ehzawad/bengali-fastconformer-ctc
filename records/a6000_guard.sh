#!/bin/bash
# Owner rule 2026-09-06: "if one card gets busy, continue on one card". Guards an A6000 pilot unit: polls every 60 s; if a
# compute process that is NOT ours appears on the A6000, stops our unit (SIGTERM -> the trainer saves a resumable checkpoint at
# the next update boundary and exits 4) and re-queues the SAME run on the A5000 with --resume auto (run_arm waits for a free
# A5000). Never touches the foreign process. Usage: a6000_guard.sh <unit> <run_name> <arm> <steps>
set -uo pipefail
cd "$(dirname "$0")/../.."
UNIT=$1; RUN=$2; ARM=$3; STEPS=$4
A6000=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda
L=experiments/p4/runs/$RUN/guard.log
until systemctl --user is-active --quiet "$UNIT"; do sleep 30; done
echo "$(date -u +%FT%TZ) guarding $UNIT ($RUN) on the A6000" >> "$L"
while systemctl --user is-active --quiet "$UNIT"; do
  ours=$(cat /sys/fs/cgroup$(systemctl --user show -p ControlGroup --value "$UNIT")/cgroup.procs 2>/dev/null | tr '\n' ' ')
  foreign=0
  while IFS=, read -r uuid pid; do
    uuid=$(echo "$uuid" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
    [[ "$uuid" == "$A6000" ]] || continue
    [[ " $ours " == *" $pid "* ]] && continue
    # foreign = a process of ANOTHER USER (a co-tenant). Our own short probes (user arafat) must not trip the guard —
    # the first guard incarnation stopped O_A6000 at update 20 because of a 5-update determinism probe of ours.
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -n "$owner" && "$owner" != "$(id -un)" ]] && foreign=1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  if [ "$foreign" = 1 ]; then
    echo "$(date -u +%FT%TZ) FOREIGN process on the A6000 -> stopping $UNIT at a checkpoint and re-queuing $RUN on the A5000" >> "$L"
    systemctl --user stop "$UNIT"
    P4_RUN_NAME=$RUN P4_UNIT=${UNIT}-a5000 P4_CARD=a5000 experiments/p4/run_arm.sh "$ARM" "$STEPS" >> "$L" 2>&1
    echo "$(date -u +%FT%TZ) re-queued on the A5000 (resumes from the last checkpoint)" >> "$L"
    exit 0
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) $UNIT finished on its own; guard exits" >> "$L"
