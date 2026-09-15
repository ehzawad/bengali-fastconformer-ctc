#!/usr/bin/env python3
"""Council N: is H=150k the right horizon for a COLD START ALONE on 1.08M clips? Sunk cost is ~2 h. Decide now or never."""
import json, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cl = (R / 'experiments/p4/council/council_l_nvidia_only.md').read_text(encoding='utf-8')
st = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered/steps.jsonl')]
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_verbatim": "\"why there is no early stopping there and also after for t thousand step what will happen and sixty thousand step diagnostics Tell me nuances. What's the plan? Did you take actually right decision? because we are already very early stage of the training so sunk cost will be pretty low if we have to restart ... see just if you are truly sure that you need to change then change because i think all this thing came from a series of verification and tinkering\"",
 "production_run_now": {"run": "N_O+macro_filtered", "update": st[-1]['update'], "loss": round(st[-1]['loss'], 3), "lr": st[-1]['lr'], "sunk_cost": f"~{st[-1]['update']/820:.1f} GPU-h; a horizon change means RESTART FROM ZERO (the cosine is a function of H; resume under a different H is refused by design)"},
 "the_horizon_as_set": "H=150,000 was set by councils A/B for a WARM START on an anticipated ~1.4M-row corpus ('≈27 epochs of 1.4M rows; owner asked for steps to scale with data') and kept by council I/L for N so that a cold start's anneal tail is preserved and (originally) so A and N had matched budgets. The matched-budget reason is GONE (N is the only run). The corpus is 1,084,696 rows, so H=150k is 34.8 epochs, not 27.",
 "phase1_proven_cold_start_recipe": "nvidia init, 984 h / 552,932 rows, batch 252, peak 5e-4, warmup 2000, cosine to 1e-5 at H=40,000 = 18.2 epochs; reached dev 20.35 frozen (S 21.03). Its curve (internal surface): 25.9% @8k, 22.9 @12k, 20.8 @19k, final @40k - most of the gain before 20k, ~0.5 pp from the anneal tail 19k->40k.",
 "scaled_alternatives_for_1084696_rows": {"Phase-1 epochs (18.2)": "H ≈ 78,000", "council A/B logic (27 epochs)": "H ≈ 116,000", "current": "H = 150,000 (34.8 epochs)", "H=100k": "23.2 epochs", "H=120k": "27.9 epochs"},
 "LR_at_60k_by_H": {"H=150k": "3.37e-4 (33.7x terminal)", "H=120k": "2.17e-4", "H=100k": "1.36e-4", "H=78k": "3.9e-5"},
 "why_no_early_stop_as_ruled": "council L Q5: a plateau anchored at 50k/earliest 70k would stop a cold start at LR 2.86e-4 (noise, not convergence); the anneal tail is where a cold start finishes; keep the full schedule. NOT ruled: a LATE plateau (e.g. anchor at 0.8H) or a shorter H.",
 "gates_as_ruled": {"40k terminal screen": "S<=22.0296 (P1+1.0), D<=21.348, empties<=0.1% - 'is it learning Bengali at all'; fail = Phase 4 ends, no fallback", "60k diagnostic": "reports S<=19.5296 & D<=19.35 & guards; a miss is recorded, never acted on (council L's top risk was killing a cold start mid-anneal)"},
 "selection_protection": "min eligible S over all validations with the 0.05 tie band -> a too-long H costs GPU-h but cannot select an over-trained checkpoint; a too-short H changes the LR curve itself (earlier anneal), which CAN change the best achievable checkpoint - unknown direction.",
 "throughput": "~820-870 updates/h on the A6000; H=150k ≈ 180-185 GPU-h; H=120k ≈ 145; H=100k ≈ 120; H=78k ≈ 95.",
 "owner_principles": ["generalisation over benchmark", "quality over quantity", "not six months; unbounded GPU-h otherwise", "ambiguity/decision -> ask the owner immediately with the council verdict attached"],
}
q = """
# QUESTIONS (owner's watcher; the sunk cost is ~2 GPU-h and rising ~0.8/h; answer with a PICK and exact numbers; 'keep as is' is valid only with a reason that survives the facts above)
Q1 HORIZON. For a COLD START ALONE (no comparison arm) on 1,084,696 rows with Phase 1's recipe otherwise unchanged, what H? Weigh: Phase 1's proven 18 epochs; council A/B's 27-epoch logic; the anneal-tail argument; over-training risk (mitigated by selection but not by the LR-curve effect); GPU-h. Pick ONE of 78k/100k/120k/150k (or another number) and state what evidence would have changed your pick.
Q2 EARLY STOPPING. With one run: (a) none (current); (b) a LATE plateau: anchor at some update where LR is already low (give the update and LR), reset on >=0.20 pp cumulative eligible improvement, N non-reset validations, earliest stop; (c) something else. Pick, with exact numbers under the H from Q1, and state the GPU-h it can save and the risk.
Q3 THE 40k SCREEN under the H from Q1: keep S<=22.0296/D<=21.348 or re-derive? At what LR will the run be at 40k under that H, and is 40k still the right update for a 'learning at all' screen (or 30k/50k)? Give exact numbers.
Q4 THE 60k DIAGNOSTIC under the H from Q1: keep as a report, or does a shorter H make a mid-run TERMINAL qualification defensible again (LR at 60k would be lower)? Rule with the number.
Q5 RESTART. If Q1 changes H: confirm restart from update 0 (fresh head, same seed -> byte-identical init), the ~2-3 GPU-h written off, and the exact launch config changes. If Q1 keeps 150k: say so plainly and why the owner's instinct to reconsider is answered.
Q6 One paragraph the orchestrator can show the owner: what the 40k and 60k points mean in plain terms, why there is (or is not) early stopping, and whether the design was right.
Q7 ADVERSARIAL: the way the chosen H produces a worse final model than the alternative, the number that reveals it, the cheapest check.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher. The owner has asked whether the horizon / early-stopping design "
          "for the sole cold-start production run is actually right, and has said to change it now if truly warranted since the sunk cost is ~2 h. "
          "Councils A-M shaped the contract; council L's verdict is included verbatim. Be concrete, numeric, decisive. Below: CONTRACT, COUNCIL L, FACTS, QUESTIONS.\n\n"
          "# CONTRACT\n" + spec + "\n\n# COUNCIL L (verbatim)\n" + cl + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4n').mkdir(exist_ok=True); Path('/tmp/council_p4n/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
