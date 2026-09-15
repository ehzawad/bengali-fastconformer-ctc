#!/bin/bash
# Extract the RESPIN train_clean tarball STREAMED from its 12 downloaded range-parts (no 87 GB concatenation:
# the extra write+read saturated /mnt/sdb and pushed the shared box to load >1,000), then ingest (SPEC §1 madasr).
# I/O and CPU priority are lowered (ionice idle, nice 15) so co-tenants keep the disk.
set -uo pipefail
cd "$(dirname "$0")/../.."
cd data4/respin
PARTS=$(for i in $(seq 0 11); do printf "IISc_RESPIN_train_bn_clean.tar.gz.part%s " "$i"; done)   # space-separated: a newline list broke the streamed cat
for p in $PARTS; do [[ -f $p ]] || { echo "MISSING $p"; exit 1; }; done
mkdir -p train
echo "$(date -u +%FT%TZ) streaming extraction start ($(du -ch $PARTS | tail -1 | cut -f1))"
ionice -c3 nice -n 15 bash -c "cat $PARTS | tar -xzf - -C train" || { echo TAR_FAILED; exit 1; }
echo "$(date -u +%FT%TZ) extracted: $(ls train)"
cd ../..
SET=$(ls -d data4/respin/train/*/ | head -1)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c3 nice -n 15 .venv/bin/python bnasr/ingest_respin.py --set-dir "$SET" --source madasr --out data4/respin/manifest_madasr.jsonl || echo INGEST_FAILED
echo "$(date -u +%FT%TZ) RESPIN_INGEST_DONE"
