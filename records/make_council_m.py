#!/usr/bin/env python3
"""Council M: the co-tenant yield rule is costing the production run a restart per small transient job."""
import json, datetime, subprocess
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
sup = (R / 'experiments/p4/run_supervised.sh').read_text(encoding='utf-8')
logs = subprocess.run("grep -ah 'co-tenant needs more\\|co-tenant baseline\\|graceful stop\\|launching on' experiments/p4/runs/*/supervisor.log | grep 2026-09-08 | sort", shell=True, capture_output=True, text=True, cwd=R).stdout
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "production_run": "N_O+macro_filtered (init N, cold start, H=150k) on the A6000, launched 13:11 UTC; yielded at update 1,177 at 14:34 to a third user's 356 MiB process; each yield = checkpoint + stop + relaunch + dataloader init ≈ 3-5 min of training lost, and the run re-baselines its co-tenants on relaunch.",
 "yield_rule_as_registered": "SPEC 8m 'yield-floor reconciliation' + 8s.5/8t.5: yield (graceful stop, resumable) when a NEW foreign process appears on our card, or when foreign memory grows > 2,000 MiB (P4_YIELD_GROWTH_MIB). Refined 06:50 UTC: a newcomer must persist across two 60-s polls. Card-selection floor: 12,000 MiB free.",
 "todays_events": logs,
 "the_newcomer": "co-tenant B, python, ~356-360 MiB on the A6000, appears periodically (06:48 for <1 min; 07:45 persisting ~1 h; 14:34 persisting). The A6000 has 49,140 MiB; co-tenant A holds 33,064; we hold ~10,200; free with us present ≈ 5,500 MiB, i.e. the 356 MiB job coexists with us without any need for our card.",
 "owner_constraints": ["never close/kill/shut a co-tenant process (verbatim, standing)", "don't throttle the server", "training must complete on its own; resume from checkpoint", "bugs: council then fix; ambiguity/decision: ask the owner immediately with the council verdict attached"],
 "orchestrator_proposal": "Unify the two triggers under the ONE registered cushion: yield when foreign memory on our card grows by > 2,000 MiB relative to the baseline, whether by a newcomer or by growth of an existing process; a newcomer below that adds to the baseline after two polls without a yield. Rationale: yielding exists to free memory for a co-tenant who needs the card; a sub-2 GB job on a card with 5.5 GB free does not. Keep: immediate yield on growth > 2,000 MiB; the 12,000 MiB launch floor; the 13 GiB watchdog; 60-s polling. Alternative considered: threshold on the newcomer's own size (>= 2,000 MiB) - equivalent in effect.",
 "supervisor_script": sup[:6000],
}
q = """
# QUESTIONS (owner's watcher; a co-tenant-policy change; answer with picks and exact wording)
Q1 Is the proposal safe for the co-tenants under the owner's 'never touch a co-tenant' rule? A co-tenant that needs more than the free memory would OOM on allocation if we do not yield first: quantify when that can happen on the A6000 (49,140 total, co-tenant A 33,064, us ~10,200, free ~5,500) and on the A5000, and whether a 2,000 MiB growth trigger polled every 60 s protects them adequately. If not, give the number that does.
Q2 Rule on the exact trigger: (a) unified growth > 2,000 MiB (newcomer or growth); (b) newcomer size >= X MiB OR growth > 2,000; (c) keep 'any persistent newcomer'. Give X if (b).
Q3 The relaunch cost: each yield loses 3-5 min and re-baselines co-tenants, so a periodic small job can cause a yield per appearance. Is there a cheaper courtesy than a full stop (e.g. none)? Confirm that a graceful stop remains the ONLY mechanism (we never signal a co-tenant).
Q4 Exact SPEC wording (a short 8u) and the exact bash condition for run_supervised.sh.
Q5 Adversarial: the way this change harms a co-tenant or the run, the number that reveals it, the cheapest check.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR programme on a SHARED two-GPU box. "
          "The production run is yielding its card to a 356 MiB co-tenant job. The owner forbids touching co-tenant processes and requires a council round "
          "before any decision reaches him. Be concrete, numeric, brief. Below: the CONTRACT, then FACTS, then QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4m').mkdir(exist_ok=True); Path('/tmp/council_p4m/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
