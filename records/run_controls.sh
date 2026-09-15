#!/bin/bash
# Owner ruling 2026-09-06: start the two O controls as soon as the FINAL frozen build exists (they contain no new-bucket data).
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q FINAL_BUILD_DONE experiments/p4/final_build.log; do sleep 60; done
[ -s experiments/p4/corpus/FREEZE.sha256 ] || { echo "no FREEZE record; refusing to launch"; exit 2; }
grep -q 'arm_O.nvme.jsonl' experiments/p4/corpus/nvme_remap_final.log || { echo "nvme remap missing"; exit 2; }
P4_RUN_NAME=O_A5000 P4_UNIT=worker-o-a5000 experiments/p4/run_arm.sh O 8000
P4_CARD=a6000 P4_RUN_NAME=O_A6000 P4_UNIT=worker-o-a6000 experiments/p4/run_arm.sh O 8000
echo "$(date -u +%FT%TZ) CONTROLS_LAUNCHED"
