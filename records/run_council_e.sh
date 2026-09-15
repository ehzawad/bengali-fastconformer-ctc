#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q FINAL_BUILD_DONE experiments/p4/final_build.log; do sleep 60; done
.venv/bin/python experiments/p4/make_council_e.py
export PATH=/mnt/sdb/arafat/.nvm/versions/node/v24.14.1/bin:$PATH
cd /tmp/council_p4e && codex exec --model gpt-6-astra -c model_reasoning_effort=xhigh --sandbox read-only --skip-git-repo-check - < prompt.md > out.log 2> err.log
echo "$(date -u +%FT%TZ) COUNCIL_E_DONE"
