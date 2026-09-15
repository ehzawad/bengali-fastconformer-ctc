#!/bin/bash
# Third chain (after all buckets exist): decoded-PCM hashing of every training bucket + panels (resumable), text audit over
# everything, then the corpus build. Sequential, nice'd, ionice best-effort; ≤8 workers (shared box).
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q CHAIN2_DONE data4/ingest_chain2.log; do sleep 60; done
until awk '{exit !($1<50)}' /proc/loadavg; do sleep 30; done
echo "$(date -u +%FT%TZ) CHAIN3 start"
T=(--train data2/train_corpus_ready.json --train data4/macro/manifest_macro_validated.jsonl --train data4/macro/manifest_macro_filtered.jsonl
   --train data4/respin/manifest_madasr.jsonl --train data4/vaani/manifest_train.jsonl)
# SPRING-INX excluded (G0.9(a), council D) -> not hashed
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c2 -n6 nice -n 10 .venv/bin/python bnasr/leakage_audit.py --procs 8 "${T[@]}" pcm-hash > data4/audit/pcm_hash_run2.log 2>&1 || echo PCM_HASH_FAILED
echo "$(date -u +%FT%TZ) pcm-hash done: $(wc -l < data4/audit/pcm_sha256.tsv) rows"
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python bnasr/leakage_audit.py --procs 8 "${T[@]}" text > data4/audit/text_run3.log 2>&1 || echo TEXT_AUDIT_FAILED
echo "$(date -u +%FT%TZ) text audit done"
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python bnasr/build_corpus.py --skip-missing-buckets > experiments/corpus_build.log 2>&1 || echo BUILD_FAILED
echo "$(date -u +%FT%TZ) CHAIN3_DONE"
