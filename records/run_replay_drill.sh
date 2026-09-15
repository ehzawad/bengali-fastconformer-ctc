#!/bin/bash
# Gate-0 GPU item (council C #10 / D Q8): 1,000-update interrupted-vs-uninterrupted replay across an epoch boundary on the A5000.
# Subset of frozen arm_O with 150k rows -> 595 updates/epoch; cut at 400, resume to 1,000 (boundary at 595 lies inside the resumed leg).
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
M=experiments/p4/corpus/replay_subset_O_150k.nvme.jsonl
nice -n 10 .venv/bin/python tools/p4/p4_train.py --manifest $M --run-name replay_full --steps 1000 --val-every 0 > experiments/p4/replay_full.log 2>&1 || echo REPLAY_FULL_FAILED
nice -n 10 .venv/bin/python tools/p4/p4_train.py --manifest $M --run-name replay_cut --steps 400 --val-every 0 > experiments/p4/replay_cut_a.log 2>&1 || echo REPLAY_CUT_A_FAILED
nice -n 10 .venv/bin/python tools/p4/p4_train.py --manifest $M --run-name replay_cut --resume auto --steps 1000 --val-every 0 > experiments/p4/replay_cut_b.log 2>&1 || echo REPLAY_CUT_B_FAILED
echo "$(date -u +%FT%TZ) REPLAY_DRILL_DONE"
