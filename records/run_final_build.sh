#!/bin/bash
# After the fingerprint match: leakage report, final corpus build with fingerprint exclusions and the fixed (indexed) M,
# NVMe remap of the arms, checksum freeze record. Sequential; CPU only.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q FP_MATCH_EXIT data4/audit/fp_match_run3.log; do sleep 60; done
echo "$(date -u +%FT%TZ) fingerprint match finished: $(tail -n1 data4/audit/fp_match_run3.log)"
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python tools/p4/p4_leakage_audit.py report > data4/audit/report_run.log 2>&1 || echo REPORT_FAILED
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python tools/p4/p4_build_corpus.py --skip-missing-buckets --fixed-M experiments/p4/corpus/panel_M.indexed_by_fingerprint.jsonl > experiments/p4/corpus_build_final.log 2>&1 || echo BUILD_FAILED
C=experiments/p4/corpus
.venv/bin/python tools/p4/p4_nvme_cache.py --rewrite-manifests $C/arm_O.jsonl "$C/arm_O+macro_filtered.jsonl" "$C/arm_O+madasr.jsonl" "$C/arm_O+vaani.jsonl" $C/arm_N.jsonl > $C/nvme_remap_final.log 2>&1 || echo REMAP_FAILED
( cd $C && sha256sum arm_*.jsonl panel_M.jsonl protected_texts.json build_report.json > FREEZE.sha256 ) && sha256sum tools/p4/p4_build_corpus.py tools/p4/p4_train.py tools/p4/p4_common.py configs/p4_ctc.yaml data/tokenizer_v1024_corpus/tokenizer.model experiments/fastconformer_ctc_bn_1kh/final.nemo data2/val_corpus.json data2/holdout_corpus.json data4/kathbath/manifest_valid.jsonl data4/respin/manifest_madasr_dev.jsonl >> $C/FREEZE.sha256
echo "$(date -u +%FT%TZ) FINAL_BUILD_DONE"
