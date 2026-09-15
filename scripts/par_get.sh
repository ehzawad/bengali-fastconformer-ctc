#!/bin/bash
# Parallel ranged download for a single large S3-style presigned URL (Range supported), N segments, resumable per segment.
# Usage: par_get.sh <url_file> <out> <nseg>
set -euo pipefail
URL=$(cat "$1"); OUT=$2; N=${3:-8}
LEN=$(curl -sS -r 0-0 -D - -o /dev/null "$URL" | tr -d '\r' | awk 'tolower($1)=="content-range:"{split($3,a,"/");print a[2]}')
[[ -n "$LEN" ]] || { echo "no content-length"; exit 1; }
echo "$(date -u +%FT%TZ) total $LEN bytes, $N segments"
SEG=$(( (LEN + N - 1) / N ))
pids=()
for i in $(seq 0 $((N-1))); do
  s=$((i*SEG)); e=$(( (i+1)*SEG - 1 )); [[ $e -ge $LEN ]] && e=$((LEN-1))
  part="$OUT.part$i"
  ( for t in $(seq 1 100); do
      have=0; [[ -f "$part" ]] && have=$(stat -c%s "$part")
      need=$((e - s + 1))
      [[ $have -ge $need ]] && exit 0
      curl -sS --retry 5 --retry-delay 10 -r $((s+have))-$e "$URL" >> "$part" && [[ $(stat -c%s "$part") -ge $need ]] && exit 0
      sleep 20
    done; exit 1 ) &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail -eq 0 ]] || { echo "segment failure"; exit 1; }
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.part$i"; done) > "$OUT" && rm -f "$OUT".part* && echo "DONE $(date -u +%FT%TZ) $(stat -c%s "$OUT")"
