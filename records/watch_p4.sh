#!/bin/bash
# Phase-4 hourly reporter / external watchdog:
#
#   experiments/p4/watch_p4.sh <run_name> [interval_seconds]
#
# Two jobs, deliberately separated from the trainer:
#
# REPORT (SPEC §0 "hourly loss+WER reporting"). Every interval it appends one
# line to <run_dir>/watch.jsonl and prints it: the newest hourly.jsonl summary
# (loss and its denominator, LR, updates, utterances, audio-hours, steps/h,
# grad-clip frequency, peak reserved GiB, cumulative GPU-h, ETA), the newest
# validation (S, panel D on both surfaces, guard verdict), the unit state and
# the card's own view of memory. The trainer writes the numbers; this reads
# them, so a hung trainer cannot fake a healthy report - a stale hourly file is
# itself the signal.
#
# WATCHDOG. The primary watchdog is INSIDE p4_train.py (SPEC §4, §8i): it
# sees per-process reserved memory and stops itself with a checkpoint written.
# This script is the backstop for the case that process is wedged and cannot
# act - it never inspects or signals anything but the ONE systemd unit for this
# run, and it stops that unit with SIGTERM (checkpoint-safe, TimeoutStopSec
# 15 min) rather than killing a PID. On a shared box a watchdog that reasons
# about "the process using the GPU" is one nvidia-smi ordering surprise away
# from killing a co-tenant's job, so it is not written that way here.
#
# Trip conditions, all reported before they act:
#   * the unit's own cgroup shows GPU reserved > WATCHDOG_GIB for 3 consecutive
#     checks (only ever measured on OUR pid, from nvidia-smi's compute-apps list)
#   * no new line in steps.jsonl for STALL_MIN minutes while the unit is active
# An ABORT or STOP file is reported and ends the watch: those are the trainer's
# own decisions and need no help.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

RUN_NAME="${1:?usage: watch_p4.sh <run_name> [interval_seconds]}"
INTERVAL="${2:-3600}"
RUN_DIR="$REPO/experiments/p4/runs/$RUN_NAME"
UNIT="${P4_UNIT:-worker-$(echo "${RUN_NAME##*_}" | tr 'A-Z' 'a-z')}"
# Both cards, because SPEC §8i's share-a-card ruling lets an arm run on either
# and the A6000 already carried O_ref's tail and the MaCro arm. Matching only
# the A5000 made `ours_mib` 0 for every A6000 run, which is not "no breach" -
# it is no measurement, and the backstop below would never have tripped.
UUIDS="GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 GPU-4754ca84-fd1d-907b-5c18-7c578d001bda"
# The registered ceiling, read from tools/p4/p4_common.py (SPEC §8i lowered it
# from 17.0 to 13.0 when the run started sharing a card). Hard-coding it here
# left the external backstop 4 GiB ABOVE the trainer's own limit, i.e. unable
# to fire for any breach the trainer would not already have caught itself.
WATCHDOG_GIB="${WATCHDOG_GIB:-$(python3 -c 'import re,sys;print(re.search(r"^WATCHDOG_GIB = ([0-9.]+)", open("tools/p4/p4_common.py").read(), re.M).group(1))' 2>/dev/null || echo 13.0)}"
STALL_MIN="${STALL_MIN:-45}"
OUT="$RUN_DIR/watch.jsonl"
mkdir -p "$RUN_DIR"
breaches=0

json_tail() {  # last line of a jsonl file, or null
  [ -s "$1" ] && tail -n 1 "$1" || echo null
}

while :; do
  now=$(date -u +%FT%TZ)
  # is-active EXITS NONZERO for a stopped unit while still printing its state,
  # so `|| echo unknown` would append a second line to a perfectly good answer.
  state=$(systemctl --user is-active "$UNIT" 2>/dev/null | head -1); state=${state:-unknown}
  pid=$(systemctl --user show "$UNIT" -p MainPID --value 2>/dev/null || echo 0)

  # GPU memory for OUR pid only. nvidia-smi reports the whole card too, but the
  # card's total includes the co-tenant and must never be the trip condition.
  ours_mib=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
             --format=csv,noheader,nounits 2>/dev/null \
             | awk -F', *' -v uu="$UUIDS" -v p="$pid" \
                   'BEGIN{n=split(uu,U," ")} {for(i=1;i<=n;i++) if($1==U[i] && $2==p) print $3}' | head -1)
  ours_mib=${ours_mib:-0}
  card=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
         | awk -F', *' -v p="$pid" '$2==p {print $1}' | head -1)
  card_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${card:-0}" 2>/dev/null | head -1)

  hourly=$(json_tail "$RUN_DIR/hourly.jsonl")
  val=$(json_tail "$RUN_DIR/val.jsonl")
  last_step_age=""
  if [ -s "$RUN_DIR/steps.jsonl" ]; then
    last_step_age=$(( ( $(date +%s) - $(stat -c %Y "$RUN_DIR/steps.jsonl") ) / 60 ))
  fi

  line=$(HOURLY="$hourly" VAL="$val" python3 - "$now" "$state" "$pid" "$ours_mib" \
            "${card_free:-0}" "${last_step_age:-null}" <<'PY'
import json, os, sys
now, state, pid, ours, free, age = sys.argv[1:7]
def j(x):
    try: return json.loads(x)
    except Exception: return None
h, v = j(os.environ.get("HOURLY", "")), j(os.environ.get("VAL", ""))
d = (v or {}).get("panels", {}).get("D", {})
print(json.dumps({
    "wall_utc": now, "unit_state": state, "pid": int(pid or 0),
    "our_gpu_mib": int(ours or 0), "our_gpu_gib": round(int(ours or 0)/1024, 3),
    "card_free_mib": int(free or 0),
    "steps_file_age_min": None if age == "null" else int(age),
    "hourly": h,
    "validation": None if v is None else {
        "update": v.get("update"), "S_pp": v.get("S_pp"),
        "D_wer_frozen": d.get("wer_frozen"), "D_wer_nemo": d.get("wer_nemo"),
        "D_empty_rate": d.get("empty_rate"),
        "guards_pass": (v.get("guards") or {}).get("pass")},
}, ensure_ascii=False))
PY
)
  echo "$line" | tee -a "$OUT"

  for f in ABORT STOP; do
    if [ -s "$RUN_DIR/$f" ]; then
      echo "watch_p4: $f recorded by the trainer -- watch ends"
      tail -n 2 "$RUN_DIR/$f"
      exit 0
    fi
  done

  if [ "$state" != "active" ]; then
    echo "watch_p4: unit $UNIT is $state -- watch ends"
    exit 0
  fi

  over=$(awk -v m="$ours_mib" -v g="$WATCHDOG_GIB" 'BEGIN{print (m/1024 > g) ? 1 : 0}')
  if [ "$over" = "1" ]; then
    breaches=$((breaches + 1))
    echo "watch_p4: WARNING our pid $pid holds $((ours_mib/1024)) GiB > ${WATCHDOG_GIB} GiB (${breaches}/3)"
    if [ "$breaches" -ge 3 ]; then
      echo "watch_p4: stopping $UNIT (SIGTERM, checkpoint-safe) -- in-process watchdog did not act"
      # Council R (SPEC 8w.9): the trainer answers SIGTERM with a generic
      # reason="graceful_stop", which the supervisor treats as resumable and
      # clears - so a watchdog stop degraded into a routine handover and lost
      # its registered requirement for owner-authorised recovery. Persist the
      # external cause BEFORE signalling; it takes precedence over the trainer's
      # generic acknowledgement.
      python3 -c "
import json, time, sys
json.dump({'watchdog': 'memory', 'gib_limit': float(sys.argv[2]), 'observed_mib': int(sys.argv[3]),
           'unit': sys.argv[4], 'terminal': True,
           'detail': ('external memory watchdog stopped this run after 3 breaches; NOT a graceful '
                      'handover - owner-authorised recovery is required before any resume'),
           'wall_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, open(sys.argv[1], 'w'))
" "$RUN_DIR/WATCHDOG_STOP" "$WATCHDOG_GIB" "$ours_mib" "$UNIT" 2>/dev/null \
        || echo "watch_p4: WARNING could not write WATCHDOG_STOP"
      systemctl --user stop "$UNIT"
      exit 4
    fi
  else
    breaches=0
  fi

  if [ -n "$last_step_age" ] && [ "$last_step_age" -ge "$STALL_MIN" ]; then
    echo "watch_p4: stalled -- no new optimizer step for ${last_step_age} min; stopping $UNIT"
    systemctl --user stop "$UNIT"
    exit 5
  fi

  sleep "$INTERVAL"
done
