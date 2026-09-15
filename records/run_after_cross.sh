#!/bin/bash
# Waits for the A6000 200-update leg and the A5000 replay's first 200 updates, runs the cross-card check, then the
# 120-step worker-side-tokenisation timing test on the A6000. Runs as a systemd unit (the harness kills long shell waiters).
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q CROSS_A6000_EXIT experiments/p4/cross_a6000.log; do sleep 30; done
until [ "$(wc -l < experiments/p4/runs/replay_full/steps.jsonl)" -ge 200 ]; do sleep 30; done
.venv/bin/python tools/p4/p4_cross_card_check.py experiments/p4/runs/replay_full/steps.jsonl experiments/p4/runs/cross_a6000/steps.jsonl --updates 200 --out experiments/p4/cross_card_check.json > /dev/null
echo "$(date -u +%FT%TZ) CROSS_CHECK_DONE"
export CUDA_VISIBLE_DEVICES=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda P4_ALLOW_A6000=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
nice -n 10 .venv/bin/python tools/p4/p4_train.py --manifest experiments/p4/corpus/replay_subset_O_150k.nvme.jsonl --run-name tput_pretok0 --steps 120 --val-every 0 --pretokenize 0 > experiments/p4/tput_pretok0.log 2>&1
echo "$(date -u +%FT%TZ) TPUT_EXIT $?"
