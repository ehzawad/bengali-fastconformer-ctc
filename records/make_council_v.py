#!/usr/bin/env python3
"""Council V: are the empty outputs a bug, a data defect, or normal? Owner asked directly."""
import json, datetime, sys
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline'); sys.path.insert(0, str(R))
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
vals = [json.loads(l) for l in open(R / 'experiments/p4/runs/N_O+macro_filtered_H120k/val.jsonl')]

facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_question": "\"damn why this happened? Fixable thing or training or decoding bugs or model limitations\"",
 "what_an_empty_is": ("a clip for which the greedy CTC decode produced a zero-length transcript. Guard: <=0.1% of the "
   "5,385-clip dev panel D, i.e. at most 5. Empties are guarded separately from WER because blank collapse is CTC's "
   "characteristic failure and a handful of empties barely moves a corpus-level WER."),
 "history": [{"update": v["update"], "empties": v["panels"]["D"]["empties"],
              "rate_pct": round(v["panels"]["D"]["empty_rate"]*100, 3),
              "D_wer_pct": round(v["panels"]["D"]["wer_frozen"]*100, 3)} for v in vals],
 "which_clips": {
   "note": "never the same clip twice; four distinct clips across nine validations",
   "idx_25":   {"empty_at": [8000], "src": "indicvoices", "dur_s": 0.45, "ref_words": 1,
                "voiced_pct": 100.0, "peak_dBFS": -28.5, "tokens_per_s": 2.23},
   "idx_4982": {"empty_at": [2000], "src": "indicvoices", "dur_s": 0.54, "ref_words": 2,
                "voiced_pct": 95.2, "peak_dBFS": -20.7, "tokens_per_s": 3.73},
   "idx_5103": {"empty_at": [8000], "src": "subakko", "dur_s": 12.28, "ref_words": 15,
                "voiced_pct": 98.6, "peak_dBFS": -9.0, "tokens_per_s": 2.12},
   "idx_5101": {"empty_at": [18000], "src": "subakko", "dur_s": 12.62, "ref_words": 6,
                "voiced_pct": 11.5, "peak_dBFS": -3.0, "tokens_per_s": 0.79,
                "ref": "কথা দিলাম বাবা হাই হাই বাই"}},
 "voiced_pct_method": ("25 ms frames, RMS per frame, a frame counted voiced if its RMS exceeds 5% of the clip's peak "
                       "RMS (about -26 dB relative to peak). Comparison group: 8 random subakko clips of 10-14 s, "
                       "median voiced 72.5% (range 60.6-100.0). idx_5101 at 11.5% is far outside that."),
 "CTC_feasibility_checked": ("10 ms feature hop, 8x depthwise striding -> one output frame per 80 ms. idx_25: 5 frames "
   "for 1 token. idx_4982: 6 frames for 2. idx_5101: 157 frames for 10. idx_5103: 153 frames for 26. ALL feasible. "
   "Across the whole panel: 0 CTC-infeasible clips, 207 'tight' (<2x frames per token). tokens/s median 3.70, "
   "p99 7.23 against the registered G0.9(d) gate of p99 <= 11. So output-length starvation is ruled out."),
 "orchestrator_diagnosis_to_be_attacked": (
   "(1) idx_5101 is a DATA defect, not a model failure: 88.5% of its 12.62 s is below the voiced threshold, its peak is "
   "-3.0 dBFS suggesting a brief loud non-speech transient, and its 0.79 tokens/s is the lowest of any clip involved. "
   "Emitting nothing for near-silence is arguably correct behaviour; the clip and/or its reference is the anomaly. "
   "(2) The other three are ordinary early-training misses that have RESOLVED - the two 0.4-0.5 s clips failed at 2,000 "
   "and 8,000 and pass now; idx_5103 is dense speech missed at 8,000 and correct today. "
   "(3) This is not blank collapse: collapse produces empties that ACCUMULATE and PERSIST on the same clips, whereas "
   "these scatter across distinct clips, never exceed 2 of 5,385, and disappear. "
   "(4) Therefore: no training bug, no decoding bug, no architectural limit, and nothing to fix in the run."),
 "run_state": ("N_O+macro_filtered_H120k, update ~18,900 of 120,000, WER 22.765% at the 18,000 validation, loss 0.733, "
               "LR 4.77e-4 = 96% of peak, epoch 4, no STOP/ABORT. Corpus 1,084,696 clips / 1,692.4 h."),
}
q = """
# QUESTIONS. The owner asked a direct diagnostic question. Attack the diagnosis; do not ratify it.

Q1 IS THE DIAGNOSIS RIGHT? Take each of the four clips and say whether the orchestrator's attribution holds. In
particular: is "88.5% silence therefore emitting nothing is correct" sound reasoning, or is it an excuse for a model
that should have transcribed the 6 words that ARE present? What would distinguish those two readings using evidence
already available (no new GPU decode)?

Q2 IS IT REALLY NOT BLANK COLLAPSE? State the quantitative signature of early blank collapse in a CTC model and check
it against the supplied history (empties per validation, WER trajectory, which clips). If the supplied evidence is
insufficient to rule collapse out, say exactly what would settle it.

Q3 THE SHORT CLIPS. 0.45 s and 0.54 s clips give 5 and 6 output frames after 8x striding. They are feasible by the
T >= L test, but is that the right test? Does a 5-frame output leave enough room for the blank-separated alignment CTC
actually needs, and is there a systematic risk for the 207 'tight' clips in panel D and their counterparts in the
1,084,696-clip training set? Quantify if you can.

Q4 IS THERE ANYTHING TO FIX? Distinguish clearly: (a) something wrong with the RUN that should change; (b) something
wrong with the DATA that should have been caught by a gate and matters for future phases; (c) nothing, this is normal.
If (b), name the gate that should have caught idx_5101 and why it did not - note G0.9(c) covers duration-vs-metadata
mismatch and G0.9(d) covers tokens/s, and the clip passed both.

Q5 THE OWNER-READABLE ANSWER. One short paragraph answering "why did this happen, is it fixable, is it a bug" in plain
language, correcting the orchestrator wherever he overstated.

Q6 ADVERSARIAL. If empties are in fact an early warning of something real, what is it, which number would show it
first, and at what threshold should the orchestrator escalate rather than explain it away?
"""
prompt = ("You are the reviewer (gpt-6-astra, xhigh), the OWNER's designated watcher for Phase 4 of a Bengali ASR "
          "programme. The owner noticed a single empty transcript in a validation and asked whether it indicates a "
          "training bug, a decoding bug, or a model limitation. The orchestrator has investigated and reached a "
          "diagnosis. Attack it. The orchestrator has already been wrong once today on a data claim, so verify rather "
          "than accept. Below: CONTRACT, FACTS, QUESTIONS.\n\n# CONTRACT\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
d = Path('/tmp/council_p4v'); d.mkdir(exist_ok=True)
(d / 'prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
