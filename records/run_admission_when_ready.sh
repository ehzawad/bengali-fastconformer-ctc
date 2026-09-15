#!/bin/bash
# Wait for each pilot arm to reach its 8,000-update endpoint and run the registered admission test (SPEC §5, §8m, m=4)
# the moment it does, so the union/main-run decision is never waiting on a human to notice a run finished.
# It decides nothing: it runs the frozen tool and writes the artifact. Terminal outcomes abort the wait.
set -uo pipefail
cd "$(dirname "$0")/../.."
CTL=experiments/p4/runs/O_A5000
L=experiments/p4/admission_chain.log
last_update() { [ -s experiments/p4/runs/$1/steps.jsonl ] && tail -n1 experiments/p4/runs/$1/steps.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.readline())["update"])' 2>/dev/null || echo 0; }
has_val8k() { python3 -c "
import json,sys
try: print(int(any(json.loads(l).get('update')==8000 for l in open('experiments/p4/runs/$1/val.jsonl'))))
except Exception: print(0)"; }
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

say "admission chain armed for: $*"
for RUN in "$@"; do
  ARM=${RUN#pilot_}
  while :; do
    u=$(last_update "$RUN")
    if [ -f experiments/p4/runs/$RUN/STOP ]; then
      r=$(python3 -c 'import json,sys
try: d=json.load(open(sys.argv[1])); print(d.get("reason") or d.get("stop") or "unparsable")
except Exception: print("unparsable")' experiments/p4/runs/$RUN/STOP 2>/dev/null)
      [ "$r" != graceful_stop ] && { say "$RUN: TERMINAL STOP ($r) at update $u -> no admission test for this arm"; break; }
    fi
    if [ -f experiments/p4/runs/$RUN/ABORT ]; then
      d=$(python3 -c 'import json,sys
try: x=json.load(open(sys.argv[1])); print((x.get("abort","")+" "+str(x.get("detail",""))).strip())
except Exception: print("unparsable")' experiments/p4/runs/$RUN/ABORT 2>/dev/null)
      case "$d" in *"CUDA OOM"*|*"out of memory"*|*"DataLoader worker"*|*"killed by signal"*) : ;;
        *) say "$RUN: TERMINAL ABORT ($d) at update $u -> no admission test for this arm"; break ;; esac
    fi
    if [ "$u" -ge 8000 ] && [ "$(has_val8k "$RUN")" = 1 ]; then
      say "$RUN reached 8,000 -> running admission test (m=4) vs O_ref"
      OUT=experiments/p4/admission_${ARM#O+}_m4.json
      nice -n 15 ionice -c3 .venv/bin/python tools/p4/p4_admission_test.py \
        --arm experiments/p4/runs/$RUN --control $CTL --step 8000 --out "$OUT" >> "$L" 2>&1
      rc=$?
      v=$(python3 -c "import json;d=json.load(open('$OUT'));print('ADMITTED='+str(d['ADMITTED']),'dS='+str(d['delta_S_pp']),'S='+str(d['S_arm_pp']),'sim='+str(d['simultaneous_upper_pp']),'ci='+str(d['ci95']))" 2>/dev/null || echo "artifact unreadable")
      say "$RUN admission rc=$rc -> $v  ($OUT)"
      break
    fi
    sleep 120
  done
done
say "admission chain done for: $*"
