#!/usr/bin/env python3
"""Compose the council-H (progress review, pre-continuation) brief and write /tmp/council_p4h/prompt.md."""
import json, subprocess, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
def J(p):
    try: return json.load(open(R / p, encoding='utf-8'))
    except Exception as e: return {"_missing": str(p), "_err": str(e)}
def vals(run):
    try: return [json.loads(l) for l in open(R / f'experiments/p4/runs/{run}/val.jsonl', encoding='utf-8')]
    except Exception: return []
def slim_val(r):
    p = r['panels']
    return {"update": r['update'], "S_pp": r['S_pp'],
            "D": round(p['D']['wer_frozen']*100, 3), "M": round(p['M']['wer_frozen']*100, 3),
            "R": round(p['R']['wer_frozen']*100, 3), "K": round(p['K']['wer_frozen']*100, 3),
            "D_per_source_pp": {k: round(v*100, 2) for k, v in p['D']['per_source'].items()},
            "conversational_pp": round(p['D']['conversational']['wer_frozen']*100, 2),
            "empty_rate": p['D']['empty_rate'],
            "guards_vs_P1_failing": [(c['check'], c['delta_pp']) for c in r['guards']['checks'] if not c.get('pass', True)]}
def last_update(run):
    try: return json.loads(open(R / f'experiments/p4/runs/{run}/steps.jsonl').readlines()[-1])['update']
    except Exception: return None
runs = {"O_ref_control": "O_A5000", "O+macro_filtered": "pilot_O+macro_filtered", "O+madasr": "pilot_O+madasr", "O+vaani": "pilot_O+vaani"}
adm = J('experiments/p4/admission_macro_filtered_m4.json')
adm_slim = {k: adm.get(k) for k in ("m", "ADMITTED", "S_arm_pp", "S_control_pp", "delta_S_pp", "ci95", "simultaneous_upper_pp", "simultaneous_quantile", "n_sentence_clusters", "draws", "seed", "tests", "guards_pass", "per_panel_wer_arm_vs_control", "continuation_eligibility_vs_P1_INFO_ONLY")}
tests = J('experiments/p4/tests/results.json')
tests_slim = {"passed": tests.get("passed"), "failed": tests.get("failed"), "skipped": tests.get("skipped"),
              "check_names": [c["name"] for c in tests.get("checks", [])],
              "failed_names": [c["name"] for c in tests.get("checks", []) if c.get("passed") is False],
              "code_sha256": tests.get("code_sha256"), "supervisor_sha256": tests.get("supervisor_sha256")}
def corpus(f):
    rows = [json.loads(l) for l in open(R / f'experiments/p4/corpus/{f}', encoding='utf-8')]
    h = sum(r['duration'] for r in rows) / 3600
    return {"rows": len(rows), "hours": round(h, 1), "updates_per_epoch": round(len(rows)/252), "epochs_at_H150k": round(150000/(len(rows)/252), 1)}
corp = {k: corpus(f) for k, f in [("O", "arm_O.jsonl"), ("O+macro_filtered", "arm_O+macro_filtered.jsonl"), ("O+madasr", "arm_O+madasr.jsonl"), ("O+vaani", "arm_O+vaani.jsonl"), ("U=macro+madasr", "arm_Nfinal_macro+madasr.jsonl"), ("U=macro+vaani", "arm_Nfinal_macro+vaani.jsonl"), ("N_final_all3", "arm_N.jsonl")]}
now = datetime.datetime.utcnow()
facts = {
 "as_of_utc": now.strftime('%Y-%m-%dT%H:%MZ'),
 "arm_progress": {k: {"last_update": last_update(v), "validations": [slim_val(r) for r in vals(v)]} for k, v in runs.items()},
 "macro_admission_m4": adm_slim,
 "gate_selection_tests": tests_slim,
 "corpora": corp,
 "throughput_updates_per_h_measured": {"macro_A6000_9h_mean": 807, "madasr_A6000": 723, "vaani_A5000": 846},
 "timeline_at_810_per_h": {"madasr_8k": "08 Sep ~01:20 UTC", "vaani_8k": "08 Sep ~02:10 UTC",
   "pathA_no_union": {"resume": "08 Sep ~02:20", "gate20k": "08 Sep ~17:30", "earliest_plateau_70k": "11 Sep ~07:00", "horizon_150k": "15 Sep ~10:00"},
   "pathB_union_pilot_first": {"union_8k_pilot": "~10 h", "resume": "08 Sep ~12:30", "gate20k": "09 Sep ~03:50", "horizon_150k": "15 Sep ~20:00"}},
 "budget_gpu_h": {"ceiling": 300, "spent_by_end_of_pilots_est": 65, "union_pilot_if_needed": 10, "continuation_8k_to_150k_at_810": 175, "total_est": "≈250 (path A) / ≈260 (path B)"},
 "commonvoice_mozilla_registry_bn": {"17.0 2024-03-15": {"validated": 44121, "validHrs": 53.51, "other": 997561, "invalidated": 7811, "clips": 1049493},
   "20.0 2024-12-06": {"validated": 44351, "validHrs": 53.81}, "23.0 2025-09-05": {"validated": 44634, "validHrs": 54.18},
   "26.0 2026-06-12 (newest; no 27.0)": {"validated": 44739, "validHrs": 54.32, "other": 999500, "invalidated": 7939, "clips": 1052178, "dev": 9402, "test": 9402},
   "phase1_ingest": "cv-corpus-26.0 validated.tsv minus its own dev/test = 44739-9402-9402 = 25935 rows = exactly data2/commonvoice/manifest.jsonl (41.51 h); 24,712 rows / 39.5 h survive Phase-4 dedup+caps"},
 "owner_rulings_today": [
   "Common Voice unconfirmed `other` tier PERMANENTLY excluded, its WER/CER filter struck, 26.7 GB deleted.",
   "EVERY release-17 artefact deleted incl. the 888 KB analysis record; every SPEC claim resting on it STRUCK (see SPEC 8n(iii), 8n-bis).",
   "Owner asked for full hyperparameters/architecture/hours before the continuation: delivered as experiments/p4/PRELAUNCH_CARD.md."],
}
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
card = (R / 'experiments/p4/PRELAUNCH_CARD.md').read_text(encoding='utf-8')
survey_head = (R / 'experiments/p4/surveys/conversational_survey_2026-09-07.md').read_text(encoding='utf-8')[:7000]
q = """
# QUESTIONS (progress review before the continuation is committed; answer with picks, numbers, exact SPEC wording where needed; 'keep as registered' is a valid and preferred answer)

Q1 Gate/selection code. Ten defects (SPEC 8n(i), D1-D11) were fixed today under 61 executable tests. Review the list. Does any FIX change the registered science rather than implement it? Scrutinise D4 (eligibility-aware retention that also keeps every eligible checkpoint inside the 0.05 pp tie band), D5 (tie band swept from the best, never chained), D6 (validation forced at 20,000 regardless of --val-every), D7 (resume refusal as a recorded ABORT exit 3 instead of a bare assert), D11 (fresh supervisor invocation refuses on a non-graceful STOP; override env var). Name any that needs an OWNER ruling rather than an implementation fix.

Q2 The union decision at 8,000, on the live trajectory. MaCro is admitted (m=4). madasr at 4k is -0.760 below the control with only the conversational guard failing (+0.664 vs P1). vaani at 2k is +0.165 ABOVE the control. Most likely outcome: madasr passes, vaani fails, so U = O ∪ macro ∪ madasr = arm_Nfinal_macro+madasr, which is NOT a tested single arm. Per 8m/8f: U trains 8,000 from init A with its own schedule prefix, then admission vs O_ref at m=4 AND S(U) ≤ S(best passing single) + 0.20. (a) Confirm this sequence and the ≈10 GPU-h. (b) Which number is 'best passing single' if madasr's 8k S ends below MaCro's 20.531 — the lower of the two? (c) If U FAILS its admission, the registered fallback is the best single arm's corpus, resumed from its own 8k — confirm, and confirm G0.8 (≥500 new h) holds for both MaCro (726.5 h) and madasr-alone (149 h does NOT — so madasr-alone can never be the long-run corpus; state the consequence).

Q3 madasr's gain is concentrated on its own panel: R moved -9.55 pp at 4k (R weight 0.10 → ≈0.96 of the S gap), D moved -0.09, three D sources moved slightly WORSE than control (indicvoices +0.29, kathbath +0.21). Under the registered rule this is a mechanical pass. Is an in-domain-dominated admission legitimate, and what EXACT card disclosure wording should accompany it? Should the continuation-eligibility guards (vs P1, at every validation) be relied on to catch a 'helps R, hurts D' drift during the long run, or is a registered per-source check vs the CONTROL at 8k also needed (it exists in the admission test: each D source ≤ control +1.0 pp)?

Q4 vaani: +0.165 above control at 2k; helps indicvoices (-0.29) and the conversational slice (21.95 vs control 22.59), hurts subakko (+1.15), openslr53 (+0.96), kathbath (+0.45). No registered early-stop applies (non-finite / >1 % empties / D > P1+5 pp). Confirm run-to-8,000 (≈5 more GPU-h) rather than an unregistered early stop. If it fails, is there any registered path to salvage its conversational benefit (e.g. a capped mixture) — or is that a new-registration item only?

Q5 The 20k gate constants. Code evaluates S ≤ S(P1) − 1.5 = 19.5296 (P1 measured 21.0296) and D ≤ D(P1) − 1.0 = 19.35, and refuses to run if either drifts >0.001 pp from council F's printed 19.530 / 19.35. Confirm the relative form governs and the tolerance is right.

Q6 CV17 disposition. On owner ruling every release-17 artefact is deleted and every measurement STRUCK (8n(iii)). Two things survive: (a) the release-choice error, evidenced by tools/convert_commonvoice.py + Mozilla's public registry; (b) 'PCM-sha256 is blind to cross-decoder recycling' kept as a FORWARD-LOOKING instrument rule with NO measurement behind it in this repository. Is (b) acceptable as a rule, or must it be struck too until re-derived on a non-CV17 pair? Propose exact wording either way.

Q7 Common Voice is CUMULATIVE (Mozilla registry: Bengali validated 44,121 → 44,739 clips across 17.0 → 26.0, every bucket only grows; Phase 1 holds all 25,935 usable clips of 26.0). Any implication for (a) the G0.9(b) 0.5 % incoming-defect rule, (b) the MaCro ∩ CommonVoice 1,047-copy finding in 8b, (c) how future buckets that are RE-RELEASES of already-held corpora must be screened?

Q8 Pre-launch card (below). 115.64 M params, 18L/512d/8h, k=9 BN, effective batch 252 = 12×21, AdamW (0.9,0.98) wd 1e-3, clip 1.0, bf16, warmup 1k 1e-5→2e-4, cosine→1e-5 at H=150k, SpecAug 2×15/2×5 %, val every 2k, ckpt every 500. One epoch = rows/252: H = 34.8 epochs (MaCro alone) or 30.6 (U=macro+madasr) or 29.9 (N). Phase 1 ran 18. Is 30-35 epochs at this LR a registered concern (overfitting) given plateau + held-out selection, or acceptable? Name any hyperparameter you would change BEFORE the continuation — and whether changing it now is even permitted (it would break 'pilot = literal prefix').

Q9 Timeline and budget (facts JSON). Confirm ≈250-260 GPU-h is within the 300 ceiling on either path. Name any step you would INSERT before the continuation (e.g. a CPU-only re-verification, a dry resume on the A6000, a specific test to re-run) and its cost.

Q10 NEXT registration only (nothing admitted now). Survey: SPRING-INX Bengali-script-only ≈280 h needs a G0.9(a) amendment with the threshold fixed BEFORE the retained rate is known and a disclosure that it is a CONTENT filter; IndicVoices upstream delta 0-116 h unresolved (CPU-only diff); Ben-10 ≈35 h. Which, if any, should be PRE-REGISTERED now so it is ready when Phase 4 closes, and with what exact threshold?

Q11 Adversarial. From here, what is the single most likely way Phase 4 reaches a WRONG conclusion (a wrongly admitted union, a wrongly passed gate, a wrongly selected checkpoint, a wrongly claimed headline)? Name the mechanism, the number that would reveal it, and the cheapest check.
"""
prompt = ("You are the progress reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A-G shaped the contract; "
          "Gate 0 closed; the control and MaCro finished and MaCro is admitted at m=4; madasr and vaani are mid-run; the continuation to H is about "
          "to be committed. Today ten gate/selection defects were fixed, the admission test hardened, and the owner withdrew a data source and "
          "ordered its artefacts destroyed. Be concrete, numeric, adversarial. 'Keep as registered' is a valid and preferred answer over post-hoc "
          "changes. Below: the CONTRACT, then FACTS (measured, JSON), then the PRE-LAUNCH CARD, then the SURVEY HEAD, then QUESTIONS.\n\n"
          "# CONTRACT (experiments/p4/SPEC.md)\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) +
          "\n\n# PRE-LAUNCH CARD (experiments/p4/PRELAUNCH_CARD.md)\n" + card +
          "\n\n# SURVEY HEAD (experiments/p4/surveys/conversational_survey_2026-09-07.md, first 7,000 chars)\n" + survey_head + "\n" + q)
Path('/tmp/council_p4h').mkdir(exist_ok=True); Path('/tmp/council_p4h/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words;', len(prompt), 'chars')
