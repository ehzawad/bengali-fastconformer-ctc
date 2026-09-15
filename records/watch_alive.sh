#!/bin/bash
# Training-alive watchdog (owner 2026-09-11: "if they want to kill our process, then let me know").
# Watches the UPDATE COUNTER, not exit codes, so it catches every way training can stop:
# SIGKILL (unit fails), SIGTERM/systemctl stop (unit exits 4 = "success", invisible to a failed-unit
# watch), an OOM, a hang, or the supervisor dying with the trainer. Read-only; signals nothing.
set -u
RUN="${1:?usage: watch_alive.sh <run_dir>}"
STALL_S="${P4_STALL_S:-720}"          # 12 min: longer than a validation (~100 s) plus a full model reload (~6 min)
POLL="${P4_ALIVE_POLL:-60}"
last=""; last_t=$(date +%s); alerted=0
upd() { tail -n1 "$RUN/steps.jsonl" 2>/dev/null | sed -n 's/.*"update": *\([0-9]*\).*/\1/p'; }
unit_state() { systemctl --user list-units 'worker-*' --no-legend 2>/dev/null | awk 'NR==1{print $3"/"$4}'; }
echo "$(date -u +%FT%TZ) ALIVE-WATCH armed | update $(upd) | stall threshold ${STALL_S}s"
while :; do
  sleep "$POLL"
  u=$(upd); now=$(date +%s)
  if [ -n "$u" ] && [ "$u" != "$last" ]; then
    if [ "$alerted" = 1 ]; then
      echo "$(date -u +%FT%TZ) TRAINING-RESUMED update $u after $(( now - last_t ))s stalled | unit $(unit_state)"
      alerted=0
    fi
    last="$u"; last_t=$now
  else
    gap=$(( now - last_t ))
    if [ "$gap" -ge "$STALL_S" ] && [ "$alerted" = 0 ]; then
      st=$(unit_state); [ -z "$st" ] && st="NO ACTIVE WORKER UNIT"
      sup=$(pgrep -c -f 'run_supervi[s]ed.sh' 2>/dev/null || echo 0)
      echo "$(date -u +%FT%TZ) TRAINING-STALLED update stuck at ${u:-unknown} for ${gap}s | unit: $st | supervisors: $sup | $(ls "$RUN"/STOP "$RUN"/ABORT "$RUN"/WATCHDOG_STOP 2>/dev/null | xargs -r -n1 basename | xargs || echo 'no stop records')"
      alerted=1
    fi
  fi
done
