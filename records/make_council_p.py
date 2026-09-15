#!/usr/bin/env python3
"""Council P: council O ruled on a CORRUPTED Phase-1 curve. Re-ask with the corrected series.

The curve councils H..O used was built by labelling EarlyStopping "New best score"
events as (i+1)*1000. Truth read from the TensorBoard event files reverses the
premise: the cold start is AHEAD of Phase 1 at every matched update, not behind.
"""
import json, datetime, math, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
from tools.p4 import p4_common as P

spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
co = Path('/tmp/council_p4o/out.log').read_text(encoding='utf-8')
curve = json.loads((R / 'experiments/p4/phase1_val_curve.json').read_text())
p1 = {r['update']: r['val_wer_pct'] for r in curve['curve']}
ours = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]

def rm(u):
    return min(v for k, v in p1.items() if k <= u)

cmp_rows = []
for v in ours:
    u = v['update']; D = v['panels']['D']
    cmp_rows.append({
        'update': u, 'ours_internal': round(D['wer_nemo'] * 100, 3), 'ours_frozen': round(D['wer_frozen'] * 100, 3),
        'S_pp': v['S_pp'], 'empties': D['empties'], 'conversational_frozen': round(D['conversational']['wer_frozen'] * 100, 3),
        'P1_raw_same_update': p1.get(u), 'P1_runmin_same_update': round(rm(u), 3),
        'we_are_ahead_of_P1_raw_by_pp': round(p1[u] - D['wer_nemo'] * 100, 3),
        'we_are_ahead_of_P1_runmin_by_pp': round(rm(u) - D['wer_nemo'] * 100, 3)})

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "why_this_round": ("Council O's verdict rests on a FALSE premise I supplied: 'the cold start is 6.4 pp behind Phase 1 at "
                    "update 2,000'. That number came from a Phase-1 curve built by enumerating EarlyStopping 'New best score' "
                    "lines in lightning_logs.txt and labelling event i as update (i+1)*1000. Those are improvement events, not "
                    "a 1,000-step grid. Read from the TensorBoard event protobufs, Phase 1's TRUE value at update 2,000 is "
                    "56.550, not 40.3. 40.3 is its update-3,000 reading. Councils I, J, K, L, M, N and O all carried the corrupted curve."),
 "bug_proof": {"packet_curve_19_values": [56.6,40.3,39.8,32.8,29.5,27.6,26.2,25.9,25.5,24.5,23.7,22.9,22.6,22.3,21.9,21.5,21.3,21.2,20.8],
   "true_improvement_events_rounded_1dp": [56.6,40.3,39.8,32.8,29.5,27.6,26.2,25.9,25.5,24.5,23.7,22.9,22.6,22.3,21.9,21.5,21.3,21.2,20.8],
   "identical": True,
   "true_updates_of_those_events": [1998,2998,3999,4999,6999,9000,11000,13001,14001,16001,18001,20002,22002,24003,27003,29003,31004,33004,35005],
   "max_understatement": "packet label 19,000 is truly update 35,005 - understated by 16,005 updates (84%)",
   "artefact": "experiments/p4/phase1_val_curve.json, 39 validations, extractor tools/p4/p4_phase1_curve.py",
   "thresholds_unaffected": ("The registered limits 22.0296/21.348/19.5296/19.35 derive from experiments/p4/baseline_p1.json - a frozen "
                             "re-evaluation of final.nemo measured 2026-09-06 - NOT from this curve. Only the mid-run comparison narrative was wrong.")},
 "phase1_TRUE_curve_internal_pct_same_panel_D_5385_rows": p1,
 "phase1_final": ("best checkpoint update 35,005 at LR 3.064e-5, val_wer 20.826 internal; restored as final.nemo; frozen re-eval "
                  "D 20.348 / internal 20.979, S_pp 21.0296. Run ended at max_steps 40,000; update 40,000 was NEVER validated. "
                  "Corpus 552,932 rows / 984.13 h, 18.23 epochs."),
 "phase1_curve_is_noisy": ("The raw series alternates by ~2-3 pp with epoch phase (e.g. 25.477 at 14,001 then 26.730 at 15,001 then "
                           "24.514 at 16,001). Phase 1's epoch = 2,194 updates; ours = 4,304. Any single-validation threshold must "
                           "carry that dispersion."),
 "our_run_vs_phase1_at_MATCHED_UPDATES": cmp_rows,
 "our_run": ("N_O+macro_filtered_H120k, NVIDIA cold start, 1,084,696 rows / 1,692.4 h, batch 252, warmup 2,000 to 5e-4, cosine to "
             "1e-5 at 120,000 (27.88 epochs). Update ~6,450 now. Recipe is IDENTICAL to Phase 1's except corpus (1.72x) and horizon (3x). "
             "Loss 0.98-1.16, empties 0 at 4k and 6k, no collapse."),
 "the_40k_screen": {
   "requirement": "at literal update 40,000: S <= 22.0296 AND D_frozen <= 21.348 AND D empties <= 0.1%. Failure = Phase 4 ends, no fallback.",
   "where_40k_sits": {"cosine_fraction": 0.32203, "LR": f"{P.lr_at(40000, sched=P.SCHED_N):.6e} = 38.50x terminal", "epochs": 9.293,
                      "updates_remaining": 80000, "GPU_h_to_reach": "46.0-48.8"},
   "SCHEDULE_MATCHED_PHASE1_POINT": ("our 40,000 is at cosine fraction 0.32203; the same fraction of Phase 1's cosine is its update 14,237, "
      "where its LR was 3.890e-4 = 38.90x terminal - within 1% of ours. Phase 1's measured value there: 25.477 internal (u=14,001), "
      "26.730 (u=15,001). Internal-minus-frozen offsets measured in this repo: +0.631 (P1 final), +1.104/+1.299/+1.400 (our 2k/4k/6k). "
      "So Phase 1 at the schedule-matched point was 24.08 to 24.85 frozen-equivalent, against the 21.348 limit: PHASE 1'S OWN RECIPE "
      "WOULD HAVE MISSED THIS SCREEN BY 2.73 to 3.50 pp."),
   "forward_reading": ("Phase 1 gained 4.651 pp internal from its schedule-matched point to its best. A run that passes 21.348 at 40k and "
                       "gains the same would finish near 16.70 frozen - better than Phase 1's 20.348 by 3.65 pp, i.e. MORE than the entire "
                       "registered Phase-4 target of 3.03 pp, demanded as evidence at one third of the horizon."),
   "but": ("our run is measurably AHEAD of Phase 1 at every matched update so far (+9.85, +5.36, +6.08 pp on the raw series; "
           "+9.85, +5.36, +2.40 on the running-minimum). The lead on the running-minimum comparator is COMPRESSING. Neither trend is 34,000 updates long.")},
 "council_O_verdict_on_the_WRONG_data": co,
 "owner_position": ("Owner, verbatim, on the false premise: 'So I think we gotta restart the full training with new hyperparameter. So after "
                    "getting report back from Codex Council, you will think maximally and then get settled.' He also holds: generalisation over "
                    "benchmark; no sunk-cost fallacy; ambiguity/decision -> ask him immediately with the council verdict attached."),
 "orchestrator_position_to_be_attacked": ("(1) NO restart: the restart premise was the data error; the recipe is Phase 1's own, proven at this "
                                          "scale, and the run is ahead of it at equal updates. (2) AMEND the 40k screen to diagnostic, add a "
                                          "gross-failure floor derived from Phase 1's own trajectory at the schedule-matched point plus measured "
                                          "dispersion, move terminal qualification to the selected checkpoint at 120k with the existing numbers."),
 "LR_table_H120k": {u: f"{P.lr_at(u, sched=P.SCHED_N):.4e} ({P.lr_at(u, sched=P.SCHED_N)/1e-5:.2f}x terminal)"
                    for u in (6000, 8000, 12000, 16000, 40000, 60000, 80000, 100000, 120000)},
}

q = """
# QUESTIONS. No 40k data exists; every amendment is prospective. Give picks, exact numbers, exact wording. "Keep as registered" is a valid answer and must be argued.

Q1 DOES THE CORRECTION CHANGE YOUR VERDICT? You (council O) chose option (d) - 40k diagnostic, sole qualification at the 120k selected
checkpoint - on the belief that this run is 6.4 pp BEHIND Phase 1 at update 2,000. It is 9.85 pp AHEAD. State plainly whether (d) survives,
and whether the direction of the error changes the cost calculus (you accepted 92.0-97.6 extra GPU-h against an unknown failure probability;
that probability now looks different). If your answer is unchanged, say why the premise was not load-bearing.

Q2 THE RESTART. The owner asked to restart the full training with new hyperparameters BECAUSE he believed the run was behind Phase 1.
It is ahead. The recipe (warmup 2,000 to 5e-4, cosine to 1e-5, batch 252, SpecAugment 2x15/2x0.05, 1,024-token Bengali SP) is exactly the
recipe that produced Phase 1's 20.348, on 1.72x the corpus with 3x the horizon. Is there ANY hyperparameter defect visible in the corrected
evidence that would justify discarding ~8 GPU-h and restarting? Name it with a number or rule the restart out explicitly. Do not be polite
about this: if you would change a hyperparameter, say which, to what, and on what evidence.

Q3 THE COMPARATOR. Three candidate comparisons of our update u against Phase 1: (i) P1's RAW val_wer at the same update; (ii) P1's
running-minimum at the same update; (iii) P1's value at the same COSINE FRACTION (our 40,000 <-> its 14,237). They give +6.08, +2.40 and a
very different picture. Which is the correct comparator for "is this run on track", and which for "will it pass a threshold at 40,000"?
Phase 1's series is noisy with epoch phase (+-2-3 pp) and our epoch is 4,304 updates vs its 2,194. Rule, and say what the noise allowance
must be for ANY single-validation terminal threshold.

Q4 THE SCREEN, RE-RULED ON CORRECT DATA. At the schedule-matched point Phase 1 itself sat 2.73-3.50 pp WORSE than the limit our run must
meet at 40,000. Is the screen therefore mis-specified as a continuation test, independent of whether our run happens to pass it? Rule. Then
pick ONE:
   (a) keep 40k terminal at S<=22.0296 / D<=21.348;
   (b) keep 40k terminal with a re-derived mid-anneal threshold (give the number AND its derivation);
   (c) 40k diagnostic + terminal screen moved to a named later update (give it);
   (d) 40k diagnostic + sole terminal qualification on the selected checkpoint at 120k (your council-O choice);
   (e) two-stage: a cheap GROSS-FAILURE floor at 40k plus real qualification at 120k.
If (e), give the floor's exact number and its derivation. My proposed derivation: Phase 1's own frozen-equivalent value at the
schedule-matched point (24.08-24.85) plus its measured single-validation dispersion (~2 pp), i.e. D_frozen <= 27.0. Attack or improve that.
State the GPU-h at risk in both error directions.

Q5 EARLY WARNING. Give the exact quantity to watch at 8k/12k/16k/24k that would tell us the run is NOT on a passing trajectory, computed
from validations we already run (zero extra GPU decode). Council O proposed m_u = max(S_u - 22.0296, D_u - 21.348) plus the weighted
contributions 0.60dD/0.20dM/0.10dR/0.10dK. With the corrected Phase-1 curve now available at 1,000-update cadence, is there a better
shape-comparison test? Give it as a formula with numeric boundaries at 8,000, 12,000, 16,000 and 24,000.

Q6 EXACT AMENDMENT TEXT for whatever you pick in Q4, in the style of SPEC sections 8s-8v, covering: what is recorded at 40k, ending class
on miss, durable milestone + replay + resume invariant, what the 60k milestone does, what qualification applies at 120k and to which
checkpoint, and the disclosure wording that this was a data-informed prospective amendment made after a data-error correction.

Q7 DISCLOSURE OF THE ERROR ITSELF. Councils I-O were given a corrupted curve. Which of their rulings, if any, must now be revisited beyond
the 40k screen? Be specific: council N chose H=120,000 partly on "Phase 1's completed cold start: 18.23 epochs" (that number is correct and
came from elsewhere) - does anything it decided depend on the corrupted curve? Same for councils I, J, K, L, M.

Q8 ADVERSARIAL. The single most likely way this decision goes wrong now that the premise has flipped from "we are behind" to "we are ahead",
the number that reveals it, and the cheapest check. Include the risk that the orchestrator over-corrects and treats an early lead as proof
of eventual quality.
"""

prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR programme. You already ruled "
          "on this exact question as council O; your verdict is included verbatim. It rested on a data error that has since been found and proven: "
          "the Phase-1 comparison curve in every packet since council H was an EarlyStopping improvement-event envelope re-labelled onto a "
          "1,000-update grid. The true curve, read from the TensorBoard event protobufs, reverses the premise. This round asks you to re-rule on "
          "correct data. The sole production run is a COLD START from NVIDIA English weights with NO fallback. Be concrete, numeric, decisive, "
          "and explicit where the corrected data does NOT change anything. Below: CONTRACT, FACTS (including your council-O verdict), QUESTIONS.\n\n"
          "# CONTRACT\n" + spec + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
d = Path('/tmp/council_p4p'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words ->', d / 'prompt.md')
