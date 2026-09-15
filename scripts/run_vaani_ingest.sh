#!/bin/bash
# Ingest Vaani Bengali parquet after the MaCro/RESPIN chain is done (one bulk-I/O job at a time).
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q CHAIN_DONE data4/ingest_chain.log; do sleep 60; done
until awk '{exit !($1<50)}' /proc/loadavg; do sleep 30; done
echo "$(date -u +%FT%TZ) vaani ingest start"
ionice -c2 -n6 nice -n 10 .venv/bin/python bnasr/ingest_parquet.py --parquet-dir data4/vaani/audio/Bengali --out-dir data4/vaani --source vaani --id-col none --text-col transcript --tag-policy vaani --workers 6 || echo VAANI_INGEST_FAILED
echo "$(date -u +%FT%TZ) VAANI_INGEST_DONE"
