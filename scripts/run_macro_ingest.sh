#!/bin/bash
# Wait for the competition bundle extraction to finish, then ingest both MaCro buckets (SPEC §1).
# nice'd and capped at 24 workers so the co-tenant keeps CPU headroom on the 40-core box.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q UNZIP_DONE data4/macro/unzip.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) extraction done: $(ls data4/macro/raw/train_mp3s | wc -l) mp3"
for b in macro_validated macro_filtered; do
  echo "$(date -u +%FT%TZ) ingest $b"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c2 -n6 nice -n 10 .venv/bin/python bnasr/ingest_macro.py --bucket $b --workers 8 || echo "INGEST_FAILED $b"
done
echo "$(date -u +%FT%TZ) MACRO_INGEST_DONE"
