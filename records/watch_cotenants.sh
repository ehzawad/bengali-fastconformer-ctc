#!/bin/bash
# Co-tenant GPU notifier (owner 2026-09-09: "every time someone occupy GPU you should let me know").
# Emits ONE line per change: a foreign compute process appearing on, or leaving, either card.
# Read-only: it never signals anything and never touches a co-tenant process.
set -u
ME="$(id -un)"
A5000=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567
A6000=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda
snapshot() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r g p m; do
      g=$(echo "$g"|tr -d ' '); p=$(echo "$p"|tr -d ' '); m=$(echo "$m"|tr -d ' ')
      u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      [ -z "$u" ] && continue
      [ "$u" = "$ME" ] && continue
      case "$g" in "$A5000") c=A5000;; "$A6000") c=A6000;; *) c=other;; esac
      echo "$c|$p|$u"          # identity is card+pid+user ONLY: keying on memory made a few MiB of drift look like a departure plus an arrival
    done | sort
}
free_of() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1; }
ours_card() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r g p; do
      p=$(echo "$p"|tr -d ' '); u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      [ "$u" = "$ME" ] || continue
      case "$(echo "$g"|tr -d ' ')" in "$A5000") echo A5000;; "$A6000") echo A6000;; esac
    done | head -1
}
# A co-tenant must survive TWO consecutive polls before it is reported (owner wants
# to know about occupation, not about a job that lives for 60 s). Departures are
# reported immediately: losing a neighbour is never something to wait on.
prev="$(snapshot)"
pending=""
echo "$(date -u +%FT%TZ) COTENANT-WATCH armed | ours=$(ours_card) | A5000 free $(free_of 0) MiB | A6000 free $(free_of 1) MiB | baseline: $(echo "$prev" | tr '\n' ' ')"
while :; do
  sleep "${P4_COTENANT_POLL:-60}"
  cur="$(snapshot)"
  if [ "$cur" != "$prev" ]; then
    # xargs collapses whitespace to empty, so a no-op diff does not fire an event
    gone=$(comm -23 <(echo "$prev") <(echo "$cur") 2>/dev/null | xargs)
    new=$(comm -13 <(echo "$prev") <(echo "$cur") 2>/dev/null | xargs)
    oc=$(ours_card)
    # only announce arrivals that were already pending from the previous poll
    confirmed=""
    for e in $new; do case " $pending " in *" $e "*) confirmed="$confirmed $e";; esac; done
    confirmed=$(echo "$confirmed" | xargs)
    [ -n "$confirmed" ] && echo "$(date -u +%FT%TZ) COTENANT-ARRIVED $confirmed | ours=$oc | A5000 free $(free_of 0) MiB | A6000 free $(free_of 1) MiB"
    pending="$new"
    [ -n "$gone" ] && echo "$(date -u +%FT%TZ) COTENANT-LEFT $gone | ours=$oc | A5000 free $(free_of 0) MiB | A6000 free $(free_of 1) MiB"
    # a process merely resizing is not an event
    prev=$(comm -23 <(echo "$cur") <(printf '%s\n' $pending | sort) 2>/dev/null; printf '%s\n' $confirmed)
    prev=$(echo "$prev" | grep -v '^$' | sort)
  fi
done
