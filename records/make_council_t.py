#!/usr/bin/env python3
"""Council T: the owner wants Vaani back too. Rule on it, and on what the pilot should actually test."""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
cs = (R / 'experiments/p4/council/council_s_spring_recovery.md').read_text(encoding='utf-8')
adm = json.loads((R / 'experiments/p4/admission_vaani_m4.json').read_text())

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%HZ'),
 "your_own_council_S_verdict": cs,
 "what_changed": ("You ruled on SPRING only. The owner has since asked for VAANI back as well, and his reasoning is "
   "explicit: \"to me, you should add both data because they are pretty clean. Because the data we already have, those "
   "are not hundred percent perfect, and [Vaani] has no latent issues. Because better data means better pattern, and "
   "the datasets are not that bad!\""),
 "VAANI_measured_record": {
   "note": "Vaani was NOT excluded for script or licence. It PASSED every gate and RAN its own 8k admission pilot.",
   "S_arm_pp": adm.get("S_arm_pp"), "S_control_pp": adm.get("S_control_pp"), "delta_S_pp": adm.get("delta_S_pp"),
   "simultaneous_quantile": adm.get("simultaneous_quantile"), "simultaneous_upper_pp": adm.get("simultaneous_upper_pp"),
   "guards_pass": adm.get("guards_pass"),
   "guards": {k: v for k, v in (adm.get("guards") or {}).items()},
   "verdict_recorded": "NOT ADMITTED, INCONCLUSIVE (SPEC 8: 'S 20.999 vs 21.073, delta -0.074, 95% CI includes 0')",
   "retained": "85,947 clips / 117.4 h (train 86,861 / 118.1 h ingested, 117.4 h retained after carving/caps)",
   "domain": "manually transcribed spontaneous image-description speech, West Bengal districts, transcribers from the "
             "same district; CC-BY-4.0; its official validation (10,105) and test (12,695) are PROTECTED one-shot panels"},
 "SPRING_candidate_after_your_corrections": {
   "status": "UPPER BOUND on candidate hours, not admissible - your correction accepted",
   "rows": 123160, "hours": 298.5,
   "corrections_I_accepted": ["the 910 matches are TRANSCRIPT overlap, not the G0.9(b) audio gate; all 910 excluded and "
     "retained overlap must be ZERO", "protected set omitted FLEURS test and Vaani val/test", "comparator must be the "
     "O+MaCro arm B, not the O control", "the pilot must be init A to compare with the A-initialised 8k controls, which "
     "needs an explicit 8t exception", "H scales by ROWS (+11.354%), not audio hours"],
   "corrections_I_dispute": [
     "You said the four disposition counts total 240,939 leaving 2 rows unexplained. They balance exactly: latin 113,943 "
     "+ empty 2 + digit_or_other 2,926 + out_of_band 0 + LEAK 910 + ADMISSIBLE 123,160 = 240,941. The 'empty 2' bucket "
     "was in the table you were given.",
     "The 647.71 h vs 625.6 h gap is reconciled, not unexplained: 647.71 h is the UPSTREAM published total; the ingested "
     "manifests are 625.60 h because ingest already rejected 5,935 rows for cps_out_of_range, 408 for "
     "duration_out_of_range and 10 empty targets (data4/spring_inx/R{1,2}/ingest_report.json)."]},
 "combined_mixture_arithmetic": {
   "current": {"hours": 1692.4, "rows": 1084696, "steps_per_epoch": 4304, "epochs_at_120k": 27.88},
   "with_SPRING_only": {"hours": 1990.9, "rows": 1207856, "H_row_scaled": 133625, "H_rounded": 134000},
   "with_SPRING_and_VAANI": {"hours": 2108.3, "rows": 1293803, "steps_per_epoch": 5134,
                             "epochs_at_120k": 23.37, "H_row_scaled": 143140, "H_rounded": 144000},
   "mixture_shift_with_both": ("macro_filtered 42.9%->34.5%, indicvoices 33.9%->27.2%, spring 14.2%, vaani 5.6%; "
                               "the conversational-leaning share (indicvoices+spring+vaani) goes 33.9% -> 47.0%")},
 "live_run": "N_O+macro_filtered_H120k at ~update 9,300 of 120,000 (7.8%), ~10.7 GPU-h spent, untouched.",
}
q = """
# QUESTIONS. Rule on Vaani and on what the pilot must actually test. Be decisive and numeric.

Q1 VAANI RE-ADMISSION. It was tested and found INCONCLUSIVE (delta S -0.074, simultaneous upper +0.154, guards clean,
conversational delta exactly 0.000), not harmful. The owner wants it back because the corpus is being rebuilt anyway.
Is re-admitting a previously-REJECTED-as-inconclusive bucket legitimate, or is it the same post-hoc selection the rules
exist to stop? Distinguish clearly between (a) new information, (b) a rule that was always arguable, and (c) wanting
more data. If it is legitimate, say under exactly what wording; if not, say so plainly and give the owner the reason in
one sentence he can act on.

Q2 WHAT THE PILOT TESTS. You specified one mixture T = O+MaCro+SPRING and a comparator B = O+MaCro at 8k. The owner
wants to train on O+MaCro+SPRING+VAANI. Pick ONE:
   (i)  pilot T1 = O+MaCro+SPRING only, admit Vaani separately or not at all;
   (ii) pilot T2 = O+MaCro+SPRING+VAANI - the mixture actually proposed for production - accepting that SPRING's and
        Vaani's individual contributions become inseparable;
   (iii) pilot both T1 and T2 (state the extra GPU-h and the multiplicity consequence);
   (iv) something else.
Give the exact admission criterion for your choice, in the same form as your council-S table, including the
conversational requirement and the comparator identity. State the family size and the simultaneous quantile.

Q3 MULTIPLICITY. SPRING was to be the fifth candidate slot (q = 1 - 0.025/5 = 0.995). Does adding Vaani back make it a
sixth, or does Vaani's earlier test already occupy a slot? Give the exact q you require and say what happens to the
historical m=4 verdicts.

Q4 HORIZON with both: rows 1,084,696 -> 1,293,803 (+19.28%), so row-scaled H = 143,140, rounded to 144,000, versus
27.88 epochs today. Confirm or correct. State the GPU-h delta against H=120,000 at 871 updates/hour and whether the
350 GPU-h ceiling still holds for the whole phase including the pilot and the abandoned prefix.

Q5 THE OWNER'S ARGUMENT, answered directly. He says: the existing corpus is not perfect either, Vaani has no latent
issues, better data means better patterns. Where is he right and where is he wrong? One paragraph he can read.

Q6 ADVERSARIAL. If the owner adds both and the pilot passes and the full run still ends worse than the current
trajectory would have, what will have caused it? Name the mechanism, the number that would have revealed it at 8k, and
whether any 8k test could have caught it.
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. You ruled as council S on recovering a Bengali-script subset of SPRING-INX; that verdict is "
          "included verbatim and its corrections have been accepted, with two exceptions the orchestrator disputes and "
          "documents below. The owner now also wants the Vaani bucket back. Rule on it, and on what the admission pilot "
          "must actually test. Be concrete, numeric, and say plainly where the owner or the orchestrator is wrong.\n\n"
          "# CONTRACT\n" + spec + "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False, default=str) + "\n" + q)
d = Path('/tmp/council_p4t'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
