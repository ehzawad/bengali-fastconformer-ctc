#!/bin/bash
# Sequential bulk-ingestion chain (one bulk-I/O job at a time; see memory shared-box-io-and-cpu-storms):
# wait for load < 50, MaCro ingest (both buckets), then RESPIN streamed extraction + ingest.
set -uo pipefail
cd "$(dirname "$0")/../.."
until awk '{exit !($1<50)}' /proc/loadavg; do sleep 30; done
echo "$(date -u +%FT%TZ) CHAIN start load=$(cut -d' ' -f1 /proc/loadavg)"
scripts/run_macro_ingest.sh
echo "$(date -u +%FT%TZ) macro done; waiting for load < 50 before RESPIN"
until awk '{exit !($1<50)}' /proc/loadavg; do sleep 30; done
scripts/run_respin_ingest.sh
echo "$(date -u +%FT%TZ) CHAIN_DONE"
