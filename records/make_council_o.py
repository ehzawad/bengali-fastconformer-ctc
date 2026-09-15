#!/usr/bin/env python3
# CORRECTED 2026-09-08 (SPEC 8w): this builder used to emit Phase 1's EarlyStopping
# improvement events labelled (i+1)*1000, i.e. the running-minimum envelope on a wrong
# x-axis. It now reads the true per-validation curve from the TensorBoard event files
# via P.phase1_val_curve(). The ARCHIVED prompt under experiments/p4/council/ is left as
# the historical record of what that council was actually told.
"""Council O: is the 40k terminal screen mis-calibrated for a cold start? The owner has raised it BEFORE any 40k data exists."""
import json, datetime, re, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
from tools.p4 import p4_common as P
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cn = (R / 'experiments/p4/council/council_n_horizon.md').read_text(encoding='utf-8')
p1 = P.phase1_val_curve()   # TRUE curve; see SPEC 8w
ours = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_verbatim": "\"But look, initially it is getting, uh, it is doing worse actually than the phase one. So maybe it's possible after 40,000 it will not beat the previous model. Pretty much possible. So, do you really think it's a good idea? Did you reconcile with Codex Council?\"",
 "status": "N_O+macro_filtered_H120k, update ~2,300 of 120,000, ~48 GPU-h to the 40k screen, ~140-150 total. NO 40k data exists. This round is PROSPECTIVE.",
 "first_datapoint": {"our_update_2000": {"D_frozen_pct": round(ours[0]['panels']['D']['wer_frozen']*100, 2), "D_internal_pct": round(ours[0]['panels']['D']['wer_nemo']*100, 2), "S_pp": ours[0]['S_pp'], "empties_pct": round(ours[0]['panels']['D']['empty_rate']*100, 3), "collapse_check": "healthy-shape: hyp/ref word ratio 0.985, deletions 10.5% of errors"},
   "phase1_at_ITS_2000": p1.get(2000), "gap_at_2000_pp": round(ours[0]['panels']['D']['wer_nemo']*100 - p1[2000], 1)},
 "phase1_cold_start_curve_internal_same_panel": p1,
 "phase1_final": "40,000 updates on 552,932 rows = 18.23 epochs, LR fully annealed to 1e-5 -> D 20.35 frozen / 20.98 internal, S 21.0296",
 "THE_STRUCTURAL_QUESTION": {
   "what_the_screen_compares": "Phase 1's FULLY ANNEALED final model (LR 1e-5, 18.23 epochs, converged) against OUR model at update 40,000 where LR = 3.84953e-4 = 38.50x terminal and only 9.29 epochs are done, with 80,000 updates and the entire anneal still ahead.",
   "threshold": "D <= 21.348 frozen (= P1 final + 1.0), S <= 22.0296 (= S(P1) + 1.0), empties <= 0.1%",
   "council_N_already_said": "'This is a near-baseline spending screen, not an is-it-learning-Bengali-at-all test. A recogniser with 25% WER plainly learned Bengali and would nevertheless fail.' and 'These limits remain policy choices, not validated predictions of eventual failure.'",
   "council_L_precedent": "council L converted the 60k TERMINAL qualification to a DIAGNOSTIC for exactly this reason - its top-ranked risk was 'inherited control logic falsely declaring the cold start unsuccessful' at 33.7x terminal LR - but left the 40k screen terminal at 38.5x terminal LR. That asymmetry was not argued.",
   "no_council_had_cold_start_data": "councils I, L and N set/kept 22.0296 with ZERO cold-start observations on this corpus. The first one now exists."},
 "two_projections_to_our_40000": {
   "per_UPDATE_alignment": "we are +6.4 pp behind Phase 1 at update 2,000; Phase 1 at 40,000 was ~20.98 internal -> ~27.4 internal ~26.3 frozen => FAILS the 21.348 screen by ~5 pp",
   "per_EPOCH_alignment": "our 40,000 = 9.29 epochs = Phase 1's update 20,390 where it was ~20.8 internal -> ~19.7 frozen => PASSES",
   "note": "identical presentations at identical updates (both 252/update), so per-update = per-presentation; the pools differ (552,932 vs 1,084,696 rows) so coverage differs 2x. The two framings disagree by ~7 pp and NEITHER is validated. Phase 1's curve is also a DIFFERENT corpus (984 h, no MaCro) and its 'best score' series is a running minimum from a lightning early-stopping monitor, not a per-step curve."},
 "upcoming_evidence_before_40k": "validations every 2,000: 4k/6k/8k tonight, 10k-16k tomorrow morning, 40k on 10 Sep ~14:00 UTC. Phase 1's own curve has 19 points over 1k-19k for shape comparison.",
 "LR_at_key_updates_H120k": {u: f"{P.lr_at(u, sched=P.SCHED_N):.4e} ({P.lr_at(u, sched=P.SCHED_N)/1e-5:.1f}x terminal)" for u in (8000, 16000, 24000, 40000, 60000, 80000, 100000, 120000)},
 "adaptive_analysis_constraint": "SPEC 8m/council G: adapting a registered threshold AFTER seeing the result it governs is exactly what the protocol forbids. Amending NOW, before any 40k observation, is prospective. The owner asked the question; the orchestrator did not raise it to escape a failing number.",
 "owner_principles": ["generalisation over benchmark", "quality over quantity", "not six months", "ambiguity/decision -> ask the owner immediately with the council verdict attached", "no fallback exists: an N40 failure ends Phase 4 with nothing"],
}
q = """
# QUESTIONS (PROSPECTIVE: no 40k data exists. Answer with picks, exact numbers, exact wording. 'Keep as registered' is valid and must be argued, not assumed.)

Q1 STRUCTURAL. Is the 40,000 screen mis-specified as a comparison? It requires a model at 38.50x its terminal LR, 9.29 epochs in, with 80,000 updates remaining, to be within 1.0 pp of a FULLY ANNEALED converged model. Quantify, from any evidence you can cite, the typical gap between a cosine-schedule model at ~0.33 of its horizon and its own annealed endpoint. If that gap is materially larger than 1.0 pp, the screen is not a spending screen but a near-certain kill. Rule.

Q2 PROJECTION. Given the single datapoint (+6.4 pp behind at update 2,000, same panel, same internal surface, same init, same batch, same LR at that point) and Phase 1's 19-point curve on a DIFFERENT corpus half the size: what is the defensible projection for our update 40,000, with an interval? Which alignment (per-update/per-presentation, per-epoch, neither) is the right one for a cold start on 2x data, and what evidence decides it? Say plainly if no defensible projection exists.

Q3 EARLIEST RELIABLE READ. What update gives the first trustworthy signal of whether the run is on a passing trajectory - 8k, 12k, 16k, 24k? Give the exact quantity to watch and the threshold that would distinguish 'tracking per-epoch (pass)' from 'tracking per-update (fail)'. This must be computable from the registered validations with zero extra GPU.

Q4 AMENDMENT OPTIONS, ranked, each with exact update/threshold/ending-class and the GPU-h at risk:
   (a) keep 40k terminal at S<=22.0296 / D<=21.348;
   (b) keep 40k terminal, re-derive the threshold for a mid-anneal model (give the number and its basis);
   (c) move the terminal screen later (which update? at 60k LR is 26.2x terminal, at 80k 12.6x, at 100k 4.4x) and keep 40k as a diagnostic;
   (d) make 40k diagnostic and put the ONLY terminal screen at a low-LR point;
   (e) a two-stage screen: a cheap absolute floor at 40k (e.g. 'is it learning at all': D <= X) plus a real qualification later.
   Pick ONE. State what it costs if the run is genuinely hopeless (GPU-h spent before we stop) and what it costs if the run is fine but the current screen would have killed it.

Q5 LEGITIMACY. Is amending this threshold NOW - after one datapoint at 2,000 that is 6.4 pp behind, before any 40k observation - a prospective amendment or an adaptive escape? Give the exact disclosure wording either way. If it is an escape, say so and rule (a).

Q6 IF WE KEEP IT AND IT FAILS. Exact wording for what a 40k failure would and would not establish, and what the owner's options are (all requiring a new registration).

Q7 THE OWNER'S ACTUAL QUESTION - 'is it doing worse than Phase 1, is this a good idea?' Answer it directly in one paragraph he can read: is being 6.4 pp behind at update 2,000 on 2x data expected, alarming, or uninformative, and what would make you change your mind.

Q8 ADVERSARIAL. The single most likely way this decision goes wrong in EITHER direction (killing a good run / funding a hopeless one for 140 GPU-h), the number that reveals it, the cheapest check.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR programme. The sole production run is a "
          "COLD START from NVIDIA English weights; there is NO fallback (the owner removed the Phase-1 warm start). The owner has looked at the first "
          "validation, seen it 6.4 pp behind Phase 1 at the same update, and asked whether the registered 40,000-update TERMINAL screen is a good idea. "
          "No 40k data exists; this is prospective. Councils A-N shaped the contract; council N (which set the horizon and kept this screen) is included "
          "verbatim. Be concrete, numeric, decisive, and honest about what cannot be projected. Below: CONTRACT, COUNCIL N, FACTS, QUESTIONS.\n\n"
          "# CONTRACT\n" + spec + "\n\n# COUNCIL N (verbatim)\n" + cn + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4o').mkdir(exist_ok=True); Path('/tmp/council_p4o/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
