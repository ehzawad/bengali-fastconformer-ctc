#!/bin/bash
# Liveness of every Phase-4 monitor, MEASURED - never inferred from a log line.
#
# Three times in one session a monitor was reported "running" on the strength of
# a banner in supervisor.log that merely PRINTS the suggested command, or on a
# pgrep whose only hit was the querying shell itself. Both failure modes are
# closed here: every row below is a live process table lookup, and the querying
# process tree is excluded by PID so it can never match itself.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:-N_O+macro_filtered_H120k}"
RUN_DIR="experiments/p4/runs/$RUN"
# Absolute run dir for argv matching. A failed `cd` would yield the EMPTY string
# and turn "preserve_ckpts <dir>" into a prefix that matches any run - fail-open
# again - so an absent directory becomes a path that can never match.
ABS_RUN_DIR=$( [ -d "$RUN_DIR" ] && (cd "$RUN_DIR" && pwd) || echo "/nonexistent/$RUN" )

# Every ancestor of this shell, so a self-match is impossible.
mine=""; p=$$
while [ "$p" -gt 1 ]; do mine="$mine $p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break; done

alive() {  # alive <pattern> -> prints "pid ELAPSED" or nothing
  # The pattern travels in the ENVIRONMENT, never in argv. Passed as `awk -v
  # pat=...` it appears in awk's OWN command line, which `ps` then snapshots -
  # so the search matched the searcher and every monitor looked UP, including
  # for a run name that does not exist. Ancestors are skipped by pid; this
  # pipeline's own members are skipped by name.
  # In `VAR=x cmd1 | cmd2` the assignment binds to cmd1 ONLY, so it must sit on
  # awk, not on ps. Bound to ps the pattern reached awk empty, index($0,"")
  # returned true, and every monitor reported UP as pid 1 - a fail-OPEN in a
  # checker whose whole job is to fail closed. Hence the empty-pattern abort.
  ps -eo pid,etime,args --no-headers 2>/dev/null | PAT="$1" MINE="$mine" awk '
    BEGIN{n=split(ENVIRON["MINE"],M," "); for(i=1;i<=n;i++) skip[M[i]]=1
          pat=ENVIRON["PAT"]; if (pat == "") exit 1}
    ($1 in skip) { next }
    /[ \/]awk( |$)/ || index($0,"monitors.sh") { next }
    index($0,pat) { print $1" "$2; exit }'
}

row() {  # row <label> <pattern> <required>
  out=$(alive "$2")
  if [ -n "$out" ]; then printf "  %-22s UP    pid %s  up %s\n" "$1" ${out}
  elif [ "$3" = "required" ]; then printf "  %-22s DOWN  (required)\n" "$1"; RC=1
  else printf "  %-22s down  (optional)\n" "$1"; fi
}

RC=0
echo "monitors for $RUN @ $(date -u +%FT%TZ)"
# The trainer's process title is deliberately anonymised on this shared box
# (see sitecustomize), so NO cmdline grep can find it - an earlier version of
# this row matched a transient pid instead. Resolve it through systemd, then
# prove that pid is both alive and the one holding our GPU allocation.
UNIT="worker-$(echo "$RUN" | tr 'A-Z+_' 'a-z--')-a5000.service"
mpid=$(systemctl --user show "$UNIT" -p MainPID --value 2>/dev/null | tr -d ' ')
if [ -n "${mpid:-}" ] && [ "${mpid:-0}" -gt 1 ] && kill -0 "$mpid" 2>/dev/null; then
  gmib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', *' -v p="$mpid" '$1==p{print $2; exit}')
  et=$(ps -o etime= -p "$mpid" 2>/dev/null | tr -d ' ')
  if [ -n "${gmib:-}" ]; then
    printf "  %-22s UP    pid %s  up %s  %s MiB on GPU\n" "trainer (MainPID)" "$mpid" "$et" "$gmib"
  else
    printf "  %-22s ALIVE pid %s  up %s  but holds NO GPU memory\n" "trainer (MainPID)" "$mpid" "$et"; RC=1
  fi
else
  printf "  %-22s DOWN  (no live MainPID for %s)\n" "trainer (MainPID)" "$UNIT"; RC=1
fi
row "supervisor"          "run_supervised.sh"                 required
row "watch_p4 (watchdog)" "watch_p4.sh $RUN"                  required
row "watch_alive (stall)" "watch_alive.sh experiments/p4/runs/$RUN" required
row "watch_cotenants"     "watch_cotenants.sh"                required
row "preserve_ckpts"      "preserve_validation_ckpts.sh $ABS_RUN_DIR" required

u=$(systemctl --user is-active "$UNIT" 2>/dev/null | head -1)
printf "  %-22s %s\n" "systemd worker" "${u:-unknown}"
s=$(ls "$RUN_DIR"/STOP "$RUN_DIR"/ABORT "$RUN_DIR"/WATCHDOG_STOP 2>/dev/null | xargs -r -n1 basename | xargs)
printf "  %-22s %s\n" "stop records" "${s:-none}"
[ "$RC" = 0 ] && echo "  ALL REQUIRED MONITORS UP" || echo "  *** A REQUIRED MONITOR IS DOWN ***"
exit $RC
