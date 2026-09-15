#!/bin/bash
# Robust run supervisor (owner 2026-09-06): "training must complete on its own; prefer the A5000; if it is busy switch to the
# A6000; survive terminal/process death; always resume from the last checkpoint".
# Runs as a systemd user unit (p4-sup-<run>). Loop until the run's step target is reached:
#   1. pick a card: the preferred card if no FOREIGN (other-user) compute process is on it and >=19.5 GB free, else the other
#      card under the same test, else wait;
#   2. launch the worker unit via run_arm.sh (--resume auto is automatic when a checkpoint exists; KillMode=mixed);
#   3. while the worker is active: every 60 s, if a foreign process appears on our card -> `systemctl stop` (graceful
#      checkpoint, exit 4) and go back to 1;  if the worker exits: completed -> done; else (crash/OOM/watchdog) -> back to 1
#      after 2 min (Restart=on-failure inside the unit also retries; this loop covers the card switch).
# Usage: run_supervised.sh <arm> <steps> <run_name> [preferred a5000|a6000]
set -uo pipefail
cd "$(dirname "$0")/../.."
ARM=$1; STEPS=$2; RUN=$3; PREF=${4:-a5000}
A5000=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567; A6000=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda
L=experiments/p4/runs/$RUN/supervisor.log; mkdir -p experiments/p4/runs/$RUN
me=$(id -un)
FSEEN=/tmp/p4_foreign_seen; mkdir -p $FSEEN
foreign_on() {  # card uuid -> 0 if a compute process of another user is on it, or was within the last 10 min (co-tenant restarts)
  local uuid=$1 rc=1
  while IFS=, read -r u pid; do u=$(echo "$u"|tr -d ' '); pid=$(echo "$pid"|tr -d ' ')
    [[ "$u" == "$uuid" ]] || continue
    own=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -n "$own" && "$own" != "$me" ]] && rc=0
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  # A foreign process is only a blocker while we are RUNNING on that card (we yield), or if it leaves too little memory.
  # For card SELECTION the free-memory test below decides: sharing a card with an idle co-tenant kernel is allowed
  # (owner 2026-09-07: "work smart"), because our footprint is measured at 9.23 GiB peak reserved.
  [ $rc = 0 ] && date +%s > $FSEEN/$uuid
  return $rc
}
busy_by_us() {  # card uuid -> 0 if one of OUR processes is on it (another arm)
  local uuid=$1 rc=1
  while IFS=, read -r u pid; do u=$(echo "$u"|tr -d ' '); pid=$(echo "$pid"|tr -d ' ')
    [[ "$u" == "$uuid" ]] || continue
    own=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' '); [[ "$own" == "$me" ]] && rc=0
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  return $rc
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1; }
last_update() { [ -s experiments/p4/runs/$RUN/steps.jsonl ] && tail -n1 experiments/p4/runs/$RUN/steps.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["update"])' 2>/dev/null || echo 0; }
LOCKDIR=/tmp/p4_card_locks; mkdir -p $LOCKDIR
pick_card() {  # echoes the card; on success this shell HOLDS the card lock on fd 9 until release_card
  local order; if [ "$PREF" = a6000 ]; then order="a6000 a5000"; else order="a5000 a6000"; fi
  [ "${P4_ONLY_PREF:-0}" = 1 ] && order="$PREF"      # never migrate (used for the A6000 control: it only makes sense on the A6000)
  local P4_SELECTING=1
  for c in $order; do
    uuid=$A5000; [ "$c" = a6000 ] && uuid=$A6000
    exec 9>$LOCKDIR/$c.lock
    if flock -n 9; then
      # Need = our measured peak reserved (9.23 GiB) + headroom. P4_NEED_MIB defaults to 12000: enough for the run and a
      # 2.7 GiB cushion, small enough to share a card with a co-tenant's idle notebook instead of waiting for a whole card.
      if ! busy_by_us $uuid && [ "$(free_mib $uuid)" -ge "${P4_NEED_MIB:-12000}" ] 2>/dev/null; then echo $c; return 0; fi
      flock -u 9
    fi
    exec 9>&-
  done
  return 1
}
release_card() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
echo "$(date -u +%FT%TZ) supervisor start arm=$ARM steps=$STEPS run=$RUN pref=$PREF after=${P4_AFTER:-none}" >> "$L"
# D11 (2026-09-07, workflow WS2): a FRESH invocation of this script used to reach the in-loop `rm -f STOP` before any
# worker had exited, so it deleted the record of a terminal outcome and resumed a run the protocol had ended -- exactly the
# resurrection council F #5 legislated against, one shell command away. The in-loop classifier (below) only runs AFTER a
# worker exits, so it could never see a STOP written by a previous invocation. Refuse at startup instead.
stop_reason() {   # -> the STOP record's reason, "" when there is no STOP file, "unparsable" when it cannot be read
  [ -f experiments/p4/runs/$RUN/STOP ] || { echo ""; return 0; }
  python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1])); print(d.get("reason") or d.get("stop") or "unparsable")
except Exception:
    print("unparsable")' experiments/p4/runs/$RUN/STOP 2>/dev/null || echo unparsable
}
abort_detail() {  # -> the ABORT record's text, "" when there is no ABORT file
  [ -f experiments/p4/runs/$RUN/ABORT ] || { echo ""; return 0; }
  python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1])); print((d.get("abort","") + " " + str(d.get("detail",""))).strip() or "unparsable")
except Exception:
    print("unparsable")' experiments/p4/runs/$RUN/ABORT 2>/dev/null || echo unparsable
}
_refuse=""
# Council R (SPEC 8w.9): an external watchdog stop is terminal and needs
# owner-authorised recovery; it must not be resumed as a graceful handover.
[ -f "experiments/p4/runs/$RUN/WATCHDOG_STOP" ] && \
  _refuse="external watchdog stop recorded ($(head -c 200 "experiments/p4/runs/$RUN/WATCHDOG_STOP" 2>/dev/null))"
_r=$(stop_reason)
[ -n "$_r" ] && [ "$_r" != graceful_stop ] && _refuse="terminal STOP recorded ($_r)"
_a=$(abort_detail)
if [ -z "$_refuse" ] && [ -n "$_a" ]; then
  case "$_a" in
    *"CUDA OOM"*|*"out of memory"*|*"DataLoader worker"*|*"killed by signal"*) : ;;   # infrastructure: resumable
    *) _refuse="terminal ABORT recorded ($_a)" ;;
  esac
fi
if [ -n "$_refuse" ]; then
  if [ "${P4_OVERRIDE_TERMINAL_STOP:-0}" = 1 ]; then
    echo "$(date -u +%FT%TZ) $_refuse OVERRIDDEN by P4_OVERRIDE_TERMINAL_STOP=1" >> "$L"
  else
    echo "$(date -u +%FT%TZ) REFUSING TO START: $_refuse. The protocol ended this run; restarting it would discard a registered outcome. Set P4_OVERRIDE_TERMINAL_STOP=1 only with an owner decision recorded in SPEC." >> "$L"
    exit 5
  fi
fi
# Serial wave order (owner 2026-09-06, A5000 only): P4_AFTER=<run> holds this run until that run has logged >= its target
if [ -n "${P4_AFTER:-}" ]; then
  until [ "$( [ -s experiments/p4/runs/$P4_AFTER/steps.jsonl ] && tail -n1 experiments/p4/runs/$P4_AFTER/steps.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["update"])' || echo 0)" -ge "${P4_AFTER_STEPS:-8000}" ]; do sleep 120; done
  echo "$(date -u +%FT%TZ) predecessor $P4_AFTER complete; proceeding" >> "$L"
fi
adopt_active() {  # an already-running worker unit for this run (launched before the supervisor existed) -> echo "unit card"
  local key=$(echo "$RUN" | tr 'A-Z+_' 'a-z--')
  for unit in $(systemctl --user list-units --no-legend 'worker-*' --state=active 2>/dev/null | awk '{print $1}'); do
    case "$unit" in *"$key"*) ;; *) continue ;; esac
    envc=$(systemctl --user show -p Environment --value "$unit" 2>/dev/null)
    case "$envc" in *"$A6000"*) echo "$unit a6000" ;; *) echo "$unit a5000" ;; esac; return 0
  done; return 1
}
while :; do
  # Council R (SPEC 8w.9): completion used to be declared from the last LOGGED
  # update. steps.jsonl is written BEFORE the checkpoint and the validation, so
  # "log 120,000 -> crash -> COMPLETE" declared a finished run with no final
  # checkpoint, no final validation and no owed finalisation. Require all three.
  u=$(last_update)
  if [ "$u" -ge "$STEPS" ]; then
    _ck="experiments/p4/runs/$RUN/ckpt/step_$STEPS.pt"
    _mi="experiments/p4/runs/$RUN/milestones/step_$STEPS.pt"
    _val=$(python3 -c "
import json
try:
    print(int(any(json.loads(l).get('update')==$STEPS for l in open('experiments/p4/runs/$RUN/val.jsonl') if l.strip())))
except Exception: print(0)" 2>/dev/null || echo 0)
    if { [ -f "$_ck" ] || [ -f "$_mi" ]; } && [ "$_val" = 1 ]; then
      echo "$(date -u +%FT%TZ) COMPLETE at update $u (final checkpoint and validation both present)" >> "$L"; exit 0
    fi
    echo "$(date -u +%FT%TZ) update $u >= $STEPS but finalisation INCOMPLETE (validation=$_val); relaunching to finish it, not declaring COMPLETE" >> "$L"
  fi
  if adopted=$(adopt_active); then
    UNIT=${adopted% *}; card=${adopted#* }; uuid=$A5000; [ "$card" = a6000 ] && uuid=$A6000
    echo "$(date -u +%FT%TZ) adopted active $UNIT on $card (last update $u)" >> "$L"
  else
  card=""; until card=$(pick_card); do sleep 60; done   # NOTE: $(...) runs pick_card in a subshell -> lock not inherited; re-take it below
  exec 9>$LOCKDIR/$card.lock; flock -n 9 || { sleep 30; continue; }
  uuid=$A5000; [ "$card" = a6000 ] && uuid=$A6000
  UNIT="worker-$(echo "$RUN" | tr 'A-Z+_' 'a-z--')-$card"
  echo "$(date -u +%FT%TZ) launching on $card as $UNIT (last update $u)" >> "$L"
  # Only a graceful hand-over record may be consumed by a launch. A terminal STOP is caught by the startup guard
  # above and by the in-loop classifier below, and must never be deleted here (D11).
  [ "$(stop_reason)" = graceful_stop ] && rm -f experiments/p4/runs/$RUN/STOP
  # P4_EXTRA_ARGS (SPEC §8s): e.g. "--config configs/p4_ctc_initN.yaml" for the init-N run; reaches p4_train.py via run_arm.sh's $*
  P4_RESTART=no P4_CARD=$card P4_RUN_NAME=$RUN P4_UNIT=$UNIT experiments/p4/run_arm.sh "$ARM" "$STEPS" ${P4_EXTRA_ARGS:-} >> "$L" 2>&1
  sleep 90
  fi
  # Snapshot the co-tenant's footprint on OUR card once we are running. Yield only when THEY need more (a new foreign
  # process, or their total grows materially) — never because free memory is low, which it always is once our own
  # 9.2 GiB is allocated. The first version of this rule measured absolute free memory and stopped a healthy run six
  # times in eleven minutes (2026-09-07 16:34-16:45, pilot_O+vaani on the A5000).
  foreign_stat() {   # -> "<pid,pid,...> <total MiB>" for processes of other users on $uuid
    local pids="" tot=0
    while IFS=, read -r u p m; do u=$(echo "$u"|tr -d ' '); p=$(echo "$p"|tr -d ' '); m=$(echo "$m"|tr -d ' MiB')
      [[ "$u" == "$uuid" ]] || continue
      own=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      [[ -n "$own" && "$own" != "$me" ]] && { pids="$pids,$p"; tot=$((tot+m)); }
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)
    echo "$pids $tot"
  }
  base=$(foreign_stat); base_pids=${base% *}; base_tot=${base#* }
  echo "$(date -u +%FT%TZ) co-tenant baseline on $card: pids[$base_pids] ${base_tot} MiB" >> "$L"
  [ "${P4_NO_YIELD:-0}" = 1 ] && echo "$(date -u +%FT%TZ) P4_NO_YIELD=1: this run does NOT yield its card to co-tenants (owner 2026-09-08); crash/restart handling unchanged" >> "$L"
  stopped_by_us=0
  # A NEW foreign process must be seen on two consecutive polls (>= 60 s) before we yield to it: on 2026-09-08 06:48 a
  # 358 MiB job that lived under a minute cost pilot_U a checkpoint-and-restart. Memory GROWTH beyond the cushion still
  # yields immediately, because that is a co-tenant actually taking the card. Yielding is our courtesy; it is not free.
  prev_new=""
  while systemctl --user is-active --quiet "$UNIT"; do
    now=$(foreign_stat); now_pids=${now% *}; now_tot=${now#* }
    newcomer=0; cur_new=""
    for p in ${now_pids//,/ }; do case ",$base_pids," in *",$p,"*) ;; *) cur_new="$cur_new,$p" ;; esac; done
    for p in ${cur_new//,/ }; do case ",$prev_new," in *",$p,"*) newcomer=1 ;; esac; done
    prev_new="$cur_new"
    # P4_NO_YIELD=1 (owner 2026-09-08, verbatim: "no don't yield to another it's just MB man; as full training has started you
    # won't yield it to anyone"): the production run never yields its card. Co-tenant processes are still never touched;
    # the crash/restart handling below is unchanged. Only the courtesy stop is off.
    if [ "${P4_NO_YIELD:-0}" != 1 ] && { [ "$newcomer" = 1 ] || [ "$now_tot" -gt "$((base_tot + ${P4_YIELD_GROWTH_MIB:-2000}))" ]; }; then
      echo "$(date -u +%FT%TZ) co-tenant needs more on $card (was ${base_tot} MiB, now ${now_tot} MiB, new process=$newcomer) -> graceful stop of $UNIT (last update $(last_update))" >> "$L"
      systemctl --user stop "$UNIT"; stopped_by_us=1; sleep 30; break
    fi
    sleep 60
  done
  release_card
  # Terminal scientific outcomes (council F #5): a registered STOP (20k progress gate, plateau, pilot abort, watchdog) or an
  # ABORT is NOT an infrastructure interruption. The supervisor must not resurrect a run the protocol ended. Only a
  # graceful hand-over stop (reason graceful_stop) and crashes are resumable.
  if [ -f experiments/p4/runs/$RUN/STOP ]; then
    reason=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason") or json.load(open(sys.argv[1])).get("stop","?"))' experiments/p4/runs/$RUN/STOP 2>/dev/null)
    case "$reason" in
      graceful_stop) : ;;                              # card hand-over: resumable
      *) echo "$(date -u +%FT%TZ) TERMINAL STOP recorded ($reason) at update $(last_update): supervisor exits without restarting" >> "$L"; release_card; exit 0 ;;
    esac
  fi
  if [ -f experiments/p4/runs/$RUN/ABORT ]; then
    # INFRASTRUCTURE aborts (a co-tenant squeezing us into CUDA OOM, a killed dataloader worker) are resumable: they say
    # nothing about the science. NUMERICAL/divergence aborts are terminal and need a human. Council F #5 draws the line at
    # "resumable infrastructure interruption" vs "scientific STOP"; OOM sits on the infrastructure side.
    detail=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print((d.get("abort","")+" "+str(d.get("detail",""))))' experiments/p4/runs/$RUN/ABORT 2>/dev/null)
    case "$detail" in
      *"CUDA OOM"*|*"out of memory"*|*"DataLoader worker"*|*"killed by signal"*)
        echo "$(date -u +%FT%TZ) infrastructure ABORT ($detail) at update $(last_update): resumable, re-queuing" >> "$L"
        rm -f experiments/p4/runs/$RUN/ABORT ;;
      *)
        echo "$(date -u +%FT%TZ) TERMINAL ABORT ($detail) at update $(last_update): supervisor exits without restarting (needs a human)" >> "$L"; release_card; exit 0 ;;
    esac
  fi
  u=$(last_update); if [ "$u" -ge "$STEPS" ]; then echo "$(date -u +%FT%TZ) COMPLETE at update $u" >> "$L"; exit 0; fi
  if [ "$stopped_by_us" = 1 ]; then
    echo "$(date -u +%FT%TZ) hand-over: re-queue now (last update $u)" >> "$L"; sleep 30
  else
    echo "$(date -u +%FT%TZ) worker $UNIT exited without completing (last update $u); waiting 11 min for any unit auto-restart before re-queuing" >> "$L"; sleep 660
  fi
done
