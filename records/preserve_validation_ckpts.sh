#!/bin/bash
# Council R bridge (2026-09-09): retention ranks checkpoints by the ROUNDED S, so a
# 5-dp tie can delete the checkpoint the registered rule would have selected -- and
# preserved hypotheses cannot reconstruct deleted model weights. Until exact decision
# handling lands, no checkpoint produced by a validation may be deleted.
#
# HARD LINKS into preserved/: zero extra bytes while the original lives, and the data
# survives the trainer's unlink. Runs entirely outside the trainer: it only reads the
# run directory and creates links, so it cannot disturb the live process.
set -u
RUN="${1:?usage: preserve_validation_ckpts.sh <run_dir>}"
EVERY="${P4_VAL_EVERY:-2000}"
mkdir -p "$RUN/preserved"
LOG="$RUN/preserved/preserve.log"
while :; do
  for d in ckpt best milestones; do
    [ -d "$RUN/$d" ] || continue
    for f in "$RUN/$d"/step_*.pt; do
      [ -e "$f" ] || continue
      b=$(basename "$f"); n=${b#step_}; n=${n%.pt}
      case "$n" in ''|*[!0-9]*) continue;; esac
      [ $(( n % EVERY )) -eq 0 ] || continue
      [ -e "$RUN/preserved/$b" ] && continue
      if ln "$f" "$RUN/preserved/$b" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) preserved $b (hard link from $d/)" >> "$LOG"
      elif cp -n "$f" "$RUN/preserved/$b.tmp" 2>/dev/null && mv -n "$RUN/preserved/$b.tmp" "$RUN/preserved/$b"; then
        echo "$(date -u +%FT%TZ) preserved $b (COPY - cross-device)" >> "$LOG"
      fi
    done
  done
  sleep 60
done
