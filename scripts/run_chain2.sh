#!/bin/bash
# Second sequential chain: RESPIN streamed extraction + ingest -> Vaani ingest -> SPRING-INX download resume + ingest.
set -uo pipefail
cd "$(dirname "$0")/../.."
scripts/run_respin_ingest.sh
echo "$(date -u +%FT%TZ) vaani ingest start"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c2 -n6 nice -n 10 .venv/bin/python bnasr/ingest_parquet.py --parquet-dir data4/vaani/audio/Bengali --out-dir data4/vaani --source vaani --id-col none --text-col transcript --tag-policy vaani --workers 6 > data4/vaani/ingest2.log 2>&1 || echo VAANI_INGEST_FAILED
echo "$(date -u +%FT%TZ) spring download resume"
ionice -c2 -n7 nice -n 15 .venv/bin/python bnasr/hf_fetch.py SPRINGLab/SPRING_INX_Bengali_R1 data4/spring_inx/R1 >> data4/spring_inx/download.log 2>&1
ionice -c2 -n7 nice -n 15 .venv/bin/python bnasr/hf_fetch.py SPRINGLab/SPRING_INX_Bengali_R2 data4/spring_inx/R2 >> data4/spring_inx/download.log 2>&1
echo "$(date -u +%FT%TZ) spring ingest start"
for R in R1 R2; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c2 -n6 nice -n 10 .venv/bin/python bnasr/ingest_parquet.py --parquet-dir data4/spring_inx/$R/data --out-dir data4/spring_inx/$R --source spring_inx_${R,,} --id-col utterance_id --text-col text --workers 6 > data4/spring_inx/ingest_$R.log 2>&1 || echo SPRING_INGEST_FAILED $R
done
echo "$(date -u +%FT%TZ) CHAIN2_DONE"
