#!/bin/bash
# A6000 leg (owner-permitted, 20 GB cap): listening boards (short decodes) then the 200-update cross-card check (council D Q1).
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=GPU-4754ca84-fd1d-907b-5c18-7c578d001bda P4_ALLOW_A6000=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1)" ]] || { echo "A6000 busy with a foreign process; abort"; exit 3; }
C=experiments/p4/corpus
nice -n 10 .venv/bin/python tools/p4/p4_listening_board.py build --random-manifest $C/retained_macro_filtered.jsonl --extra-manifest $C/retained_madasr.jsonl --n-random 100 --n-worst 20 --n-extra 50 --worst-pool 600 --label "MaCro-filtered + RESPIN" --out experiments/p4/audit/board_macro_madasr.html > experiments/p4/audit/board_A.log 2>&1 || echo BOARD_A_FAILED
nice -n 10 .venv/bin/python tools/p4/p4_listening_board.py build --random-manifest $C/retained_vaani.jsonl --n-random 50 --n-worst 10 --n-extra 0 --worst-pool 300 --label "Vaani" --out experiments/p4/audit/board_vaani.html > experiments/p4/audit/board_B.log 2>&1 || echo BOARD_B_FAILED
echo "$(date -u +%FT%TZ) BOARDS_DONE"
nice -n 10 .venv/bin/python tools/p4/p4_train.py --manifest $C/replay_subset_O_150k.jsonl --run-name cross_a6000 --steps 200 --val-every 0 > experiments/p4/cross_a6000.log 2>&1 || echo CROSS_A6000_FAILED
echo "$(date -u +%FT%TZ) A6000_QUAL_DONE"
