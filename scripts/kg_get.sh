#!/bin/bash
# Download ONE file from a Kaggle dataset with a KGAT access token.
# Kaggle's v1 API answers 302 to a signed GCS URL; forwarding the Bearer header to GCS breaks
# the request, so resolve the redirect first and fetch the signed URL without auth.
# Usage: kg_get.sh <owner/dataset> <path/inside/dataset> <out_file>
set -euo pipefail
[[ -n "${KAGGLE_API_TOKEN:-}" ]] || KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
enc=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$2")
loc=$(curl -sS -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
      "https://www.kaggle.com/api/v1/datasets/download/$1/$enc")
[[ "$loc" == https://storage.googleapis.com* ]] || { echo "no signed redirect for $2 (got: ${loc:0:80})" >&2; exit 2; }
curl -sS --retry 5 --retry-delay 5 -C - -o "$3" "$loc"
