#!/usr/bin/env python3
"""Council K: the owner flipped the order - cold start (init N) FIRST, warm start (init A) second."""
import json, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cj = (R / 'experiments/p4/council/council_j_init_weights.md').read_text(encoding='utf-8')
def last(run):
    return json.loads(open(R / f'experiments/p4/runs/{run}/steps.jsonl').readlines()[-1])['update']
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_directive_verbatim": "\"okay slight change of decision... like you're removing some data we did from the phase 1... what you're doing is okay, but flip it like use NVIDIA weights first then mine's one, got it?\"",
 "orchestrator_reading": "Run the cold-start arm (init N = nvidia/stt_en_fastconformer_ctc_large + fresh Bengali head) as the FIRST full run on the admitted corpus, and the warm-start arm (init A = Phase-1 weights) SECOND, serially on the one usable card. Both to completion; both finalists decoded on the held panels (SPEC 8r). If the reading is wrong the owner will correct it.",
 "owner_standing_rule": "\"before asking questions you must reconcile with codex council as it is my bot my watcher, my bystander\" - this council round is that reconciliation; the orchestrator is building the init-N trainer path meanwhile (needed under either order).",
 "adopted_sequence_being_changed": "SPEC 8q/8r (councils I, J): A resumes from the admitted 8k state ~12:50 UTC today -> A's 20k gate ~05:00 UTC 9 Sep -> N launches only AFTER A's 20k PASS, serially after A finishes (~16 Sep) because the A5000 cannot host us (10,538 MiB free vs 10,096-10,230 MiB whole-process need).",
 "pilot_U_status": {"update": last('pilot_U'), "of": 8000, "eta_8k_utc": "~12:40-12:50", "purpose": "admit the union corpus U = O u {macro, madasr} (a DATA decision, made with init A); its 8k checkpoint was to be A's resume point. Under the flip, U's admission still decides the corpus; A's resume point simply waits on disk."},
 "gates_by_init": {"A": "20k gate S<=19.5296 & D<=19.35 & guards (terminal); plateau from 50k, earliest stop 70k", "N (council I 8q)": "40k screen S<=min(22.0296, S_A40+2.00) & D<=min(21.348, D_A40+2.00) & empties<=0.1%; 60k qualification S<=19.5296 & D<=19.35 & all P1 guards; no plateau stop after 60k; runs to 150k. NOTE: the 40k screen references A's LITERAL 40k checkpoint, which under N-first does not exist yet."},
 "cost_to_first_go_no_go": {"A first": "20k gate at ~15 GPU-h / ~15 h", "N first": "40k screen at ~47-55 GPU-h / ~2.3 days; 60k qualification at ~70-83 GPU-h / ~3.5 days"},
 "calendar_serial_either_order": "first run ~7.5 days, second ~7.5 days (A may plateau-stop earlier), both finish ~23-25 Sep either way; the held-panel comparison and the FLEURS decode happen after BOTH finish, so the headline date does not move with the order.",
 "budget_adopted": "350 GPU-h through N's 40k decision, 525 if it passes (SPEC 8r).",
 "init_N_implementation_status": "NOT BUILT. Needed: NVIDIA .nemo loader with FULL CTC-head replacement (1,025 outputs incl. blank, recorded seed+hash), encoder+BN buffers preserved, no Phase-1 tensors; recipe 5e-4/2,000 warmup/cosine->1e-5 at 150k as a registered config; N's own durable 40k/60k decisions with owed-validation replay and resume invariants; the warm-only 20k gate/invariant must NOT fire for N; no pilot aborts for N. Estimated build+test: several hours; the A6000 idles from ~12:50 until it is ready.",
 "phase1_recipe_for_N": "lr 5e-4, warmup 2000, cosine to 1e-5, wd 1e-3, batch 12x21=252 (Phase 1 used H=40k; council I set H=150k for N)",
}
q = """
# QUESTIONS (the owner has DIRECTED a change; this round is his required reconciliation, not a request for permission; answer with picks, numbers, exact SPEC wording; if the flip is harmful say so plainly with the number)

Q1 READING. Is the orchestrator's reading of "flip it like use NVIDIA weights first then mine's one" correct (cold-start full run first, warm-start second, serial, both to completion)? If another reading is plausible, state it and what it would change.

Q2 CONSEQUENCES THE OWNER SHOULD SEE. In one table: what the flip changes and what it does not - first go/no-go timing and GPU-h (20k@~15 GPU-h for A vs 40k@~50 for N), calendar to the headline (unchanged?), risk that the known-good path is delayed behind the unknown one, and anything the owner may not have considered. Be concrete.

Q3 N'S 40k SCREEN WITHOUT A's 40k CHECKPOINT. Council I's screen is `S_N40 <= min(22.0296, S_A40 + 2.00)`. Under N-first, A's 40k does not exist. Rule: drop the A-relative clause and use the absolute 22.0296 / 21.348 only? Keep it and run A to 40k first (defeats the flip)? Something else? Give the exact amended text.

Q4 A's RESUME POINT. Under the flip, A resumes from its admitted 8k checkpoint AFTER N finishes (~16 Sep). Any risk in the checkpoint sitting on disk for a week (none technical)? Must A's 20k gate still apply when A runs second? (Yes?) Confirm and state the SPEC consequence.

Q5 WHAT IF N FAILS ITS SCREEN OR QUALIFICATION FIRST? Then A runs as before. Confirm nothing about A's registered path changes. What if N PASSES and reaches 150k first - does A still run? (Yes - the comparison needs both.) Confirm.

Q6 THE IDLE CARD. The A6000 is free from ~12:50 until the init-N trainer path is built and tested. Options: (a) idle it; (b) resume A for its 8k->20k stretch meanwhile (~15 GPU-h, gets the cheap gate answered, then pause A at its 20k checkpoint and hand the card to N) - is (b) a legitimate use of the gap that does NOT violate the owner's "NVIDIA first" intent, given A's 20k gate is a go/no-go on the WHOLE phase? Rule.

Q7 REGISTRATION TEXT. Give the exact SPEC amendment (a short 8s) that records the owner's directive verbatim, the new order, N's amended gates, A's unchanged gates, and the sequence with dates.

Q8 ADVERSARIAL. The single most likely way the flip produces a wrong belief or a wasted week, the number that reveals it, the cheapest check.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme, acting as the OWNER's designated watcher: he has "
          "directed a change to the adopted sequence and requires a council reconciliation before it executes. Councils A-J shaped the contract; "
          "council J's verdict is included verbatim. Be concrete, numeric, adversarial, and brief where the answer is simply 'yes'. Below: the "
          "CONTRACT, then COUNCIL J, then FACTS, then QUESTIONS.\n\n# CONTRACT (experiments/p4/SPEC.md)\n" + spec +
          "\n\n# COUNCIL J VERDICT (verbatim)\n" + cj + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4k').mkdir(exist_ok=True); Path('/tmp/council_p4k/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
