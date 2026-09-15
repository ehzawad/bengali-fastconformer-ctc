#!/usr/bin/env python3
# CORRECTED 2026-09-08 (SPEC 8w): this builder used to emit Phase 1's EarlyStopping
# improvement events labelled (i+1)*1000, i.e. the running-minimum envelope on a wrong
# x-axis. It now reads the true per-validation curve from the TensorBoard event files
# via P.phase1_val_curve(). The ARCHIVED prompt under experiments/p4/council/ is left as
# the historical record of what that council was actually told.
"""Compose council I: register a from-NVIDIA (init N) ablation against the Phase-1 warm start (init A)."""
import json, sys, datetime
from pathlib import Path
R = Path('/mnt/sdb/arafat/ehz/llm/bengali-asr-pipeline')
def vals(run):
    try: return [json.loads(l) for l in open(R / f'experiments/p4/runs/{run}/val.jsonl', encoding='utf-8')]
    except Exception: return []
def slim(r):
    p = r['panels']
    return {"update": r['update'], "S_pp": r['S_pp'], "D_frozen": round(p['D']['wer_frozen']*100, 3),
            "D_nemo_internal": round(p['D']['wer_nemo']*100, 3), "M": round(p['M']['wer_frozen']*100, 3),
            "R": round(p['R']['wer_frozen']*100, 3), "K": round(p['K']['wer_frozen']*100, 3)}
spec = (R / 'experiments/p4/SPEC.md').read_text(encoding='utf-8')
facts = {
 "as_of_utc": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
 "owner_request_verbatim": ("\"how could reusing an existing model would help? or it actually does that we don't know like starting with "
   "Nvidia vs Phase 1 out can you somehow verify which would work better and dataset completely ready and verified do an "
   "ablation test on it ans reconciled the codex council.\" Earlier: \"you shouldn't bound by GPU hours like it's unbounded... "
   "but it shouldn't be like six months\"."),
 "phase1_recipe_from_config_launch_yaml": {"pretrained_model": "nvidia/stt_en_fastconformer_ctc_large (English; CTC head re-initialised for the 1,024-piece Bengali tokenizer)",
   "tokenizer": "data/tokenizer_v1024_corpus (same as Phase 4)", "batch": "12 x 21 = 252 (same as Phase 4)", "sched": "CosineAnnealing",
   "lr_peak": 5e-4, "min_lr": 1e-5, "warmup_steps": 2000, "max_steps": 40000, "weight_decay": 1e-3, "corpus_hours": 984.13,
   "final": "dev frozen 20.348 % (S_pp 21.0296); NeMo-internal 20.979 % at zero-update restoration"},
 "phase1_cold_start_curve_nemo_internal_dev_wer_pct": P.phase1_val_curve(),   # CORRECTED, SPEC 8w: was the improvement-event envelope on a 1,000-grid x-axis
 "phase1_curve_note": "TRUE per-validation curve from the TensorBoard event files (tools/p4/p4_phase1_curve.py); NeMo-internal surface, NOT comparable to frozen numbers (offset +0.63 pp at zero updates, +1.46 pp at 8k on the control) - do not chain them",
 "phase4_warm_start_recipe": {"init": "Phase-1 final.nemo (61362a36), fresh AdamW", "lr_peak": 2e-4, "warmup": 1000, "H": 150000, "batch": 252},
 "warm_start_trajectories_frozen_surface": {k: [slim(r) for r in vals(v)] for k, v in
   [("O_ref_control_(P1_weights_on_P1_corpus_alone)", "O_A5000"), ("O+macro_filtered", "pilot_O+macro_filtered"), ("O+madasr", "pilot_O+madasr"), ("O+vaani", "pilot_O+vaani")]},
 "P1_baseline_frozen": {"S_pp": 21.0296, "D": 20.348},
 "final_corpus_candidates": {"O+macro_filtered": {"rows": 1084696, "hours": 1692.4}, "O+macro+madasr (most likely)": {"rows": 1179937, "hours": 1841.3}, "all three": {"rows": 1265884, "hours": 1958.7}},
 "decision_timeline": {"madasr_8k": "08 Sep ~01:40 UTC", "vaani_8k": "08 Sep ~02:40 UTC", "corpus_decided": "08 Sep ~03:00 UTC",
   "warm_start_continuation_resumes": "08 Sep ~03:00 (or ~13:00 if a union pilot is needed)", "second_card_free_from": "when the last pilot ends, subject to the co-tenant"},
 "cards": {"A5000": "24.5 GB, co-tenant holds 13.5 GB, we fit 10.1 GB", "A6000": "49 GB, co-tenant 33 GB, we fit 10.2 GB",
   "cross_card_status": "8k control agreement test NOT completed (O_A6000 reached 4,038; council E/H caveat stands); 200-update cross-card check: ids/LR identical, 20-step loss blocks differ <=1.58 % vs <=1.23 % same-card drift"},
 "budget_gpu_h": {"soft_ceiling": 300, "registered_plan_total": "≈250-260 (≈280 at 723 upd/h)", "init_N_full_to_150k": "≈175-190", "init_N_staged_to_40k": "≈50",
   "throughput_updates_per_h": "723-855 measured"},
 "one_shot_panels_constraint": "FLEURS test (920 rows, sealed) is decoded ONCE; holdout, Vaani validation/test, SPRING validation/test are one-shots after final selection. If the warm-start headline is decoded on schedule, no clean sealed panel remains for a later init-N comparison.",
 "same_data_order": "shard_seed 20260905 and the same manifest give both inits the SAME cut order (a paired design at the data level).",
}
q = """
# QUESTIONS (registering an init ablation; answer with picks, exact numbers and exact SPEC wording; 'do not run it' is a valid answer if you can show it is uninformative or unaffordable)

Q1 MECHANISM. State precisely what a warm start from Phase 1 (init A) buys and what it risks on THIS corpus: ~2x the data of Phase 1, same language and task, a decent starting model (20.35 dev), 30-35 nominal epochs at H=150k. Then predict the direction of init N (cold start from nvidia/stt_en_fastconformer_ctc_large with the Bengali tokenizer) vs init A at FULL budget, with your reasons and your confidence. The owner's exact words: "how could reusing an existing model would help? ... starting with Nvidia vs Phase 1 ... can you somehow verify which would work better".

Q2 RECIPE FOR INIT N. Phase 1's own proven cold-start recipe (peak 5e-4, warmup 2,000, cosine to 1e-5) or the Phase-4 recipe (2e-4, 1,000)? Using 2e-4 handicaps a cold start; using 5e-4 confounds init with LR. Rule, and say which question the chosen design answers ("best achievable from each init" vs "same hyperparameters, different init").

Q3 GATE FOR INIT N. A cold start cannot meet the warm start's 20k gate (S <= 19.5296; Phase 1 cold was S~21 at 40k on half the data). Propose init N's own registered progress gate: at which update, what threshold, terminating what. Phase 1's curve above is the only cold-start evidence in hand.

Q4 SELECTION ACROSS INITS AND THE ONE-SHOT PANELS. Options: (a) hold FLEURS and all one-shots until BOTH arms finish, select the final checkpoint by the registered rule (lowest eligible S, 0.05 pp tie band -> earlier) ACROSS both arms' eligible checkpoints, decode once on the winner; (b) decode the registered warm-start headline on schedule and treat init N as development-only evidence with no sealed confirmation; (c) something else. Rule, with council G's adaptive-analysis discipline in mind (SPEC 8m Q1) and the fact that FLEURS can be decoded exactly once.

Q5 CROSS-CARD CONFOUND. Init N would run on the OTHER card, concurrently with the warm-start continuation, so wall-clock does not double. The 8k cross-card control agreement was never completed. Given the effect sizes at stake (points, not tenths), is the card confound acceptable, and what minimal agreement check (if any) must precede or accompany it?

Q6 BUDGET AND SEQUENCING. Full parallel init N to 150k adds ~175-190 GPU-h (total ~430-440 vs the 300 soft ceiling; the owner said "unbounded ... not six months"). Alternatively a STAGED design: init N to 40,000 (~50 GPU-h), continue only if it is within a registered margin of init A at 40,000. Pick one and give exact stage gates, or show that neither is worth running.

Q7 WHAT DOES NOT RE-RUN. Confirm init N needs NO 8k admission against O_ref (the corpus is already admitted; init is not a data question) and NO union step; it trains on the SAME final manifest the warm start uses, decided at ~03:00 UTC. State what must be identical (manifest bytes, tokenizer, batch, SpecAug, panels, cadence, shard_seed) and what may differ (init, LR schedule if Q2 says so, warmup, BN buffers).

Q8 WHAT WOULD MAKE THE COMPARISON UNINTERPRETABLE. Name the confounds and the numbers that reveal them. Include: the internal-vs-frozen surface offset that is not constant; the plateau rule interacting differently with a cold start (anchor at 50,000 may bite a still-improving cold start); guard eligibility (a cold start is ineligible under the P1 guards for a long time - does that starve the selection?).

Q9 IF INIT N WINS. Consequences for the card, the claim, and how Phase 1 is described. If init A wins, what exactly may be claimed about warm starting.

Q10 ADVERSARIAL. The single most likely way this ablation reaches a WRONG conclusion, the number that reveals it, the cheapest check.
"""
prompt = ("You are the ablation-design reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A-H shaped the contract; "
          "MaCro is admitted; madasr and vaani finish their pilots within hours; the warm-start continuation resumes ~03:00 UTC. The OWNER has now "
          "asked, in the middle of that, whether the registered warm start from Phase 1 is actually better than a cold start from the NVIDIA "
          "English model, and has asked for an ablation to be registered and reconciled. Be concrete, numeric, adversarial. Below: the CONTRACT, "
          "then FACTS (measured, JSON), then QUESTIONS.\n\n# CONTRACT (experiments/p4/SPEC.md)\n" + spec +
          "\n\n# FACTS\n" + json.dumps(facts, indent=1, ensure_ascii=False) + "\n" + q)
Path('/tmp/council_p4i').mkdir(exist_ok=True); Path('/tmp/council_p4i/prompt.md').write_text(prompt, encoding='utf-8')
print(len(prompt.split()), 'words')
