#!/usr/bin/env python3
"""Council J: the owner's direct challenge on initial weights, at maximum reasoning."""
import json, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
ci = (R / 'experiments/p4/council/council_i_init_ablation.md').read_text(encoding='utf-8')
def recs(d): return {r['update']: r for r in (json.loads(l) for l in open(R / f'experiments/p4/runs/{d}/val.jsonl', encoding='utf-8'))}
ctl = recs('O_A5000'); U = recs('pilot_U'); mac = recs('pilot_O+macro_filtered'); mad = recs('pilot_O+madasr'); va = recs('pilot_O+vaani')
def row(r): p = r['panels']; return {"S": r['S_pp'], "D": round(p['D']['wer_frozen']*100, 2), "M": round(p['M']['wer_frozen']*100, 2), "R": round(p['R']['wer_frozen']*100, 2), "K": round(p['K']['wer_frozen']*100, 2)}
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_question_verbatim": "\"ehzawad/stt_bn_fastconformer I feel like you use this in the code, it will make things better than nvidia/stt_en_fastconformer_ctc_large ? how you know which one will work better, did you test out anything empirical? reconcile with max reasoning Astra model using codex council\"",
 "owner_principle_verbatim": "\"it is not about beating the benchmark it is about being a good generalizable model so be careful about that\"",
 "identity_check": "ehzawad/stt_bn_fastconformer (hub) = 463,155,200 bytes; local experiments/fastconformer_ctc_bn_1kh/final.nemo = 463,155,200 bytes, sha256 61362a36; every Phase-4 arm incl. pilot_U restored from that local file (train.log). They are the same model.",
 "HONEST_EVIDENTIARY_STATUS": {
   "direct_test_of_init_A_vs_init_N_on_the_phase4_corpus": "NONE. Never run. The warm start was a registered design decision (councils A/B), not a measured result.",
   "what_exists_indirectly": {
     "phase1_cold_start_from_nvidia_on_984h_nemo_internal_dev_wer": {"8000": 25.9, "12000": 22.9, "19000": 20.8, "40000": "20.35 frozen / S 21.03 (final)"},
     "phase4_warm_start_control_on_the_same_984h_frozen_S": {"2000": 21.540, "4000": 21.318, "6000": 21.175, "8000": 21.073, "start_point_P1": 21.030},
     "caveats": "different surface (internal vs frozen; offset +0.63 at zero updates, +1.46 at 8k - NOT constant), different LR (5e-4/2000 vs 2e-4/1000), different horizon (40k vs 150k), different corpus. NOT a controlled comparison."},
   "council_I_prediction": "init A weakly favoured at 150k, 60% subjective confidence, 'a judgment, not an estimated probability'."},
 "arms_at_8000_frozen_S": {"control": row(ctl[8000]), "O+macro_filtered": row(mac[8000]), "O+madasr": row(mad[8000]), "O+vaani": row(va[8000])},
 "union_pilot_U_so_far": {"2000": dict(row(U[2000]), gap_to_control=round(U[2000]['S_pp']-ctl[2000]['S_pp'], 3), guards_failing=[(c['check'], c['delta_pp']) for c in U[2000]['guards']['checks'] if not c['pass']]),
   "status": "running on the A6000, update ~2,600 of 8,000, ETA ~12:50 UTC; ceiling at 8k = 20.280 (madasr 20.080 + 0.20) plus the O_ref rule"},
 "timeline_now": {"union_verdict": "08 Sep ~12:50 UTC", "full_run_resumes_if_U_passes": "08 Sep ~13:00 UTC", "20k_gate": "09 Sep ~05:00 UTC", "council_I_says_init_N_launches": "only AFTER A's 20k PASS, on the other card, IF the owner adopts 350->525 GPU-h AND rules on the launch floor (A5000 has 11,014 MiB free < 12,000 floor while the co-tenant holds 13,550)"},
 "cards": "A5000: co-tenant 13,550 MiB, 10,538 free, our floor 12,000 -> cannot launch there. A6000: co-tenant 33,064, pilot_U 9,7xx, ~5,700 free.",
 "budget": "300 soft ceiling; ≈250-280 committed by the registered plan; init N full ≈175-190 more; council I: 350 through N's 40k, 525 if it passes.",
}
q = """
# QUESTIONS (the owner has asked the init question THREE times; answer him directly, in his terms, at maximum reasoning; 'we do not know and here is exactly what it costs to know' is a valid and preferred answer over false confidence)

Q1 ANSWER THE OWNER DIRECTLY. "did you test out anything empirical?" - state in one paragraph what empirical evidence exists about init A (Phase-1 warm start = ehzawad/stt_bn_fastconformer) vs init N (nvidia/stt_en_fastconformer_ctc_large cold start) on THIS corpus, what it does and does not establish, and your best-calibrated belief with its basis. Do not soften; do not inflate.

Q2 SHOULD THE FULL RUN WAIT? The union pilot ends ~12:50 UTC and the full run (init A) resumes minutes later. Council I said run init N AFTER A's 20k PASS, in parallel. The owner's instinct is that the init question should be settled first. Rule: (a) proceed with A now and run N alongside after the gate (council I); (b) HOLD the full run and run a matched N screen first (what length, what cost, what it would and would not settle); (c) run A and N from 13:00 in parallel from the same corpus, both from update 0 of their own recipe, with the one-shots held (what it costs, what blocks it). Give the GPU-h and calendar for each and pick one with reasons that survive the owner's generalisation principle.

Q3 THE MINIMUM DECISIVE EXPERIMENT. What is the smallest experiment that would actually settle "which init generalises better on this corpus" to a standard you would sign - not 'looks ahead at 8k' but settled? Specify: horizon for N, the comparison point, which panels decide (S? the held one-shots? both?), the statistic, the stopping rule, and the cost. If NO experiment under 200 GPU-h settles it, say so.

Q4 GENERALISATION, NOT BENCHMARK. The owner's principle is verbatim above. S is a weighted development score; FLEURS is one sealed read-speech register; the holdout and the official Vaani split are the never-selected one-shots. Under that principle, which of these should DECIDE between A and N, which should merely be REPORTED, and does council I's 'decode the one-shots on the winner only' survive the principle? The orchestrator proposed decoding the holdout + Vaani one-shots on BOTH finalists as disclosed diagnostics after the S-selection. Rule on that.

Q5 WHAT WOULD MAKE A LOOK BETTER THAN IT IS. Name the ways a warm start can win on S while being the worse generalising model (inherited Phase-1 biases; D was Phase 1's own selection panel; the corpus was admitted using A; the buckets were chosen by A's errors). For each, the number that would expose it and whether it is available before or only after the full run.

Q6 WHAT WOULD MAKE N LOOK WORSE THAN IT IS. Same, for the cold start (LR still high at any early checkpoint; English BN statistics; fresh head; plateau anchor timing; the 4-point early deficit).

Q7 THE 20k GATE INTERACTION. If A FAILS its 20k gate (S > 19.5296), Phase 4 terminates by rule and council I says N cannot rescue it. Is that right under the owner's principle - or is a gate failure of A exactly the moment N should be tried? Rule, with the exact SPEC consequence.

Q8 IF YOU HAD TO BET. Give a single number: P(N beats A on the held one-shots at full budget). Then say what single observation between now and 8k of N would move it most.

Q9 THE OWNER'S TWO OPEN DECISIONS (budget 350->525; launch floor 12,000 -> 10,500 MiB). Recommend each, with the one-line risk of each choice, so he can decide in one message.

Q10 ADVERSARIAL. The single most likely way this whole line of inquiry produces a WRONG belief about initial weights, and the cheapest check.
"""
prompt = ("You are the reviewer (gpt-6-astra, MAXIMUM reasoning) for Phase 4 of a Bengali ASR programme. The OWNER has asked the same question three "
          "times and wants it answered without evasion: which initial weights are better, and what empirical evidence exists. Councils A-I shaped the "
          "contract; council I designed an init ablation (its verdict is included verbatim); the union pilot is running and the full warm-start run "
          "resumes in ~7 hours. Be concrete, numeric, adversarial, and above all HONEST about what is and is not known. Below: the CONTRACT, then "
          "COUNCIL I's verdict, then FACTS, then QUESTIONS.\n\n# CONTRACT (experiments/p4/SPEC.md)\n" + spec +
          "\n\n# COUNCIL I VERDICT (verbatim)\n" + ci + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4j').mkdir(exist_ok=True); Path('/tmp/council_p4j/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
