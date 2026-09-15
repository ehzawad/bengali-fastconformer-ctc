#!/usr/bin/env python3
"""Council S: the owner wants the clean SPRING-INX rows back. Restart at 7.6%?"""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
ours = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_verbatim": ("\"what do you mean by code switches one oh, the why whole but bucket gate fails tell me I don't want "
   "it back but I would want it back if some rows are really clean I could at least get more data right that's my thing "
   "and we are still early in the training maybe we can get back to it and start training\". Earlier tonight, also: "
   "\"I hate sunk cost fallacy\" and \"my goal is not to restart this my goal is what should be done should be done\"."),
 "what_I_measured_just_now": {
   "method": ("Applied the frozen registered surface (metric_surface(NFC(text), score_punctuation=False)) to every "
              "SPRING-INX R1+R2 train row, then the existing script predicate, the duration band, and an exact-sentence "
              "duplicate test against the FULL protected set (D 5,385 + holdout 6,767 + panel M 1,829 + RESPIN dev 1,500 "
              "+ Kathbath valid 2,236 = 16,217 unique sentences)."),
   "all_rows": 240941, "all_hours": 625.6,
   "dropped_latin_codeswitch": {"rows": 113943, "hours": 317.7, "pct_rows": 47.3},
   "dropped_digit_or_other_script": {"rows": 2926, "hours": 8.8},
   "dropped_leak_into_protected_panels": {"rows": 910, "hours": 0.7, "pct_incoming": 0.3777,
      "ceiling_G09b": 0.5, "verdict": "PASS but NOT zero"},
   "ADMISSIBLE": {"rows": 123160, "hours": 298.5, "pct_rows": 51.1},
   "corpus_effect": "1,692.4 h -> 1,990.9 h (+17.6%); 1,084,696 -> 1,207,856 rows; steps/epoch 4,304 -> 4,793; "
                    "epochs at H=120,000 fall from 27.88 to 25.04",
   "correction_I_made": ("my first pass reported 0.000% leakage; the loader had silently failed on val_corpus.json and "
                         "holdout_corpus.json (JSONL, not JSON), so only 1,829 sentences were actually compared. "
                         "Re-run with all 16,217. The real figure is 0.3777%.")},
 "why_SPRING_was_excluded": ("G0.9(a): script violations counted on incoming rows, >1% = the BUCKET fails. Measured 48.3% "
   "of R1 rows carry Latin-script code-switch words. SPEC: 'Rows are never partially repaired (no deletion of Latin words, "
   "no transliteration).' Arm O+spring_inx was NOT RUN. A drafted amendment already exists in SPEC (Bengali-script-only "
   "subset, whole-row reject) and REQUIRES OWNER AUTHORISATION, with the disclosure that it is informed by observed script rates."),
 "THE_PROCESS_PROBLEM": ("Every other bucket earned admission on a measured 8k pilot against the O control: macro_filtered "
   "ADMITTED, madasr ADMITTED, vaani NOT admitted (inconclusive, CI crossed 0), union U NOT admitted (best S of any arm, "
   "19.871, but conversational +0.829 vs the +0.5 guard). SPRING has NO pilot and NO ablation. Admitting ~300 h now, "
   "immediately after I filtered it row-by-row and looked at the result, is the post-hoc selection the bucket gate exists "
   "to prevent - and it would be done in our own favour."),
 "the_case_FOR_adding_it": ("SPRING is genuinely conversational (mock conversations and monologues, human transcripts, "
   "vendor QC, IIT Madras/NLTM, CC BY 4.0, ungated, 16 kHz). The conversational slice is this run's WEAKEST panel: 29.39% "
   "at update 8,000 against an eligibility limit of 22.075%. The registered survey ranked SPRING FIRST among all "
   "conversational candidates. +17.6% corpus for ~300 h of the exact speech type we are short of."),
 "the_cost_of_restarting_now": ("update ~9,100 of 120,000 = 7.6%, about 10.5 GPU-h spent, 871 upd/h. A restart from zero "
   "discards that. Corpus rebuild + re-audit + re-freeze + identity re-bind is additional wall-clock. An 8k admission "
   "pilot is roughly 9-10 GPU-h on top. Aggregate ceiling is 350 GPU-h; roughly 128 h of the current run remain."),
 "run_state": {"update": 9100, "validations": [{"u": v["update"], "S_pp": v["S_pp"],
    "D_frozen_pp": round(v["panels"]["D"]["wer_frozen"]*100, 3),
    "conversational_pp": round(v["panels"]["D"]["conversational"]["wer_frozen"]*100, 2)} for v in ours]},
 "options": {
   "A": "continue the current run untouched; register the SPRING subset for the NEXT phase",
   "B": "stop now, run the registered 8k admission pilot for O+macro+spring against the existing control, and restart the "
        "full run from zero ONLY if it earns admission on the registered guards",
   "C": "stop now and restart from zero with the SPRING subset included, no pilot",
   "D": "something else you specify"},
}
q = """
# QUESTIONS. The owner wants the ~298.5 admissible hours. Answer concretely; he hates sunk-cost reasoning in BOTH directions.

Q1 PICK ONE of A/B/C/D and defend it with numbers. State the GPU-h each costs and what each buys. If you pick B, give the
exact admission criterion the pilot must meet, in the same form as the criteria that admitted macro_filtered and rejected
vaani and U - including what the conversational guard must do, given the conversational slice is currently the weakest panel.

Q2 THE POST-HOC PROBLEM. Is a row-level Bengali-script filter applied AFTER measuring the script rates a legitimate
amendment, or is it selection in our own favour? SPEC already drafts it with a disclosure. Is that disclosure sufficient,
and what exact wording would you require? Would your answer change if the filter had been written before any measurement?

Q3 THE 910 LEAKING ROWS. 0.3777% of incoming SPRING rows are byte-identical (after the frozen surface) to sentences in the
protected panels. That passes the 0.5% ceiling. But the union corpus was rejected on a DIFFERENT guard while passing this
one. Does a non-zero leak rate change the admission calculus, and must those rows simply be dropped or does their existence
implicate the rest of the bucket?

Q4 THE HORIZON. Adding 123,160 rows moves steps/epoch 4,304 -> 4,793, so H=120,000 falls from 27.88 to 25.04 epochs. Should
H change if SPRING is admitted, and to what? Recall you set H=120,000 on an exposure argument.

Q5 IF THE ANSWER IS "CONTINUE" - what exactly should be registered NOW so the SPRING subset can be admitted to the next
phase without re-litigating any of this, and what must be measured before then?

Q6 ADVERSARIAL. The single most likely way this decision goes wrong in either direction, the number that reveals it, and
the cheapest check. Include the possibility that adding 300 h of conversational speech makes the conversational slice WORSE.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. A production run is live at 7.6% of its horizon. The owner has just asked whether the clean half of "
          "an EXCLUDED dataset can be recovered and the run restarted with it. Be concrete, numeric and decisive, and say "
          "plainly where the orchestrator's framing is wrong. Below: CONTRACT, FACTS, QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
d = Path('/tmp/council_p4s'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
