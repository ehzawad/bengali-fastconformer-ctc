#!/bin/bash
# Download a whole Kaggle dataset as its archive.zip bundle (single GCS stream, resumable).
# Usage: kg_bundle.sh <owner/dataset> <out.zip>
set -euo pipefail
[[ -n "${KAGGLE_API_TOKEN:-}" ]] || KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
for attempt in $(seq 1 200); do
  loc=$(curl -sS -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
        "https://www.kaggle.com/api/v1/datasets/download/$1")
  if [[ "$loc" == https://storage.googleapis.com* ]]; then
    if curl -sS --retry 3 --retry-delay 10 -C - -o "$2" "$loc"; then echo "DONE $(date -u +%FT%TZ) $(stat -c%s "$2")"; exit 0; fi
  else
    echo "$(date -u +%FT%TZ) no redirect (rate limit?) attempt $attempt" >&2
  fi
  sleep 60
done
exit 1
