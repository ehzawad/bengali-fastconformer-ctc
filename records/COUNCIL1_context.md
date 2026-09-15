You are an independent senior ML-systems reviewer (gpt-6-astra, xhigh). Audit and DESIGN Phase 4 of a Bengali ASR programme
before any GPU is spent. Be concrete, numeric, adversarial. The orchestrator (Claude) is the final judge but wants to be argued with.

# STANDING CONSTRAINTS (owner-set, not negotiable)
- One RTX A5000 24 GB (17.5 GiB practical line), shared box, co-tenant must never be disturbed. Hard cap for ALL of Phase 4: 200 A5000 GPU-h.
- Training data must be human-labelled. Model-FILTERED human labels are allowed (owner decision today); pseudo-labels and TTS audio are NOT.
- Architecture fixed by owner today: FastConformer-CTC large (nvidia/stt_en_fastconformer_ctc_large lineage, ~115 M, 8x subsampling), i.e. the Phase-1 recipe.
- FLEURS-bn official TEST (920 recordings, 349 distinct sentences) is the sealed headline; opened ONCE at the end with a paired sentence-clustered
  bootstrap vs frozen Phase-2 hypotheses. FLEURS validation is contaminated (88/150 sentences in fleurs_train) and is never used.
- Static manifests only (weighted Lhotse mux was rejected in Phase 1: it breaks audited epoch/resume semantics). No dataset ablations beyond what you pre-register.
- Pre-registered gates before long runs; no post-hoc reinterpretation; a failed gate = stop, not "extend".
- Publication: public gated NC Hugging Face card + private GitHub with the full pipeline. Claims must be defensible.

# WHAT EXISTS (all measured)
Phase 1 (2026-08-30/31): FastConformer-CTC large from the English checkpoint, tokenizer = 1,024-piece Bengali BPE trained on train-side text,
corpus 984.1 h / 552,932 utts (fleurs_train 7.2 h, openslr53 164.0, subakko 110.8, kathbath 82.0, commonvoice 39.7, indicvoices 580.5),
AdamW lr 5e-4 cosine to 1e-5, warmup 2,000, wd 1e-3, clip 1.0, eff. batch 252 (12 x 21), SpecAug 2x15 freq / 2x0.05 time, bf16, max 40,000 steps
(= ~18 epochs; 552,932/252 = 2,194 steps/epoch), val every 1,000 steps on a locked 5,385-row / 8.5 h dev panel (5 non-FLEURS sources).
Wall clock 36.4 h for 40k steps incl. validations (~1,100 steps/h). Dev val_wer curve (NeMo-internal, per 1k-ish): .566 .403 .398 .328 .295 .276
.262 .259 .255 .245 .237 .229 .226 .223 .219 .215 .213 .212 .208 — still creeping down at 40k (≈ -0.3 pp per 1k at the end).
Result: FLEURS-test 19.57 % (CI 18.76–20.38), speaker-disjoint holdout 19.55 %, conversational 24.8 %, 0 empty outputs; 2.11x faster than
hishab's conformer-large at equal batch. Frozen scorer: punctuation->space, NFC, whitespace collapse.
Phase 2: Qwen3-ASR-1.7B LoRA on the same corpus -> FLEURS-test 16.54 % (dev 17.03 %), 1,461 ms/clip. This is the number to beat.
Phase 3: Parakeet-TDT-0.6B on the same corpus failed its equal-exposure gate twice; stopped. Lesson: on this corpus the 115 M CTC learned fastest.
External reference: hishab/titu_stt_bn_conformer (Conformer-CTC large, trained on ~4,426 h INCLUDING FLEURS train+dev) = 14.37 % FLEURS-test
under our scorer, but 22.30 % on our dev panel.
Phase-1 data lessons: Shrutilipi (544 h raw, semi-supervised forced alignment) was HARD-EXCLUDED after an ablation lost by 24.6 pp with blank
collapse (its low-cps slice measured 91.8 % WER / 22.5 % empty); synthetic TTS (indictts, 18.8 h) measured 104 % WER and was dropped;
FULL natural mixture beat a capped IndicVoices mixture by 4.7 pp in a pre-registered 3k-step ablation.

# NEW DATA (downloaded today from the Kaggle 1st-place Bengali.AI 2023 release by tugstugi; his Whisper-medium won with WER-filtered human data)
Provenance of his filter: an initial Whisper trained on OpenSLR+TTS transcribed each corpus; rows with WER < 15 % vs the given transcript were
kept; "three rounds". His text normalisation: Bengali digits rows dropped, all punctuation removed. Duplicate rows exist in his CSVs
(165,873 exact duplicate lines); figures below are AFTER path-dedup. Zero conflicting transcripts per path.
| source (his dir)            | unique clips | hours | label origin                                                    | licence (to confirm) |
| macro (Bengali.AI MaCro/OOD-Speech "train" split)     | 550,921 | 736.8 | crowd-sourced READ speech of prompted sentences; competition says "only algorithmically cleaned"; then his WER<15 filter | CC-BY-4.0 (competition rules; no redistribution of data itself) |
| macro validated (competition "valid" split)           |  29,583 |  48.9 | "manually reviewed and corrected" by Bengali.AI annotators; disjoint from the filtered set | CC-BY-4.0 |
| madasr (RESPIN/MADASR-2023 Bengali train)             | 419,461 | 603.3 | read speech of expert-composed dialect-rich sentences, ~2,000 speakers; his WER<15 filter | RESPIN terms — UNKNOWN yet |
| madasr-dev (official dev)                             |   1,183 |   1.9 | official split | same |
| shrutilipi wer15                                      |  95,972 | 142.6 | AI4Bharat forced-aligned news; his WER<15 filter (we excluded ALL of it in Phase 1) | CC-BY-4.0 |
| openslr-asr (SLR53, his cleaned text)                 | 207,380 | 202.6 | human read speech; we already train on 166,585 of these; 9,363 shared rows differ only by punctuation/hyphen removal; he has 49,398 utts we dropped in cleaning/carving, we have 8,603 he dropped | CC-BY-SA-4.0 |
| kathbath-train                                        |  57,895 | 102.5 | human; we have 46,240 | CC-BY-4.0 |
| kathbath valid / test / test_known (official)         | 2,770 / 1,782 / 2,805 | 5.0 / 3.2 / 5.0 | official splits; 0 text overlap with our train | CC-BY-4.0 |
| EXCLUDED by owner: pseudo1-9 YouTube (~500 h wer10 / ~350 h wer5), bn-google-tts 441.8 h, "long" 362 h concatenations with added noise |
Ours to keep: indicvoices 580.5 h (human transcribed + verified), subakko 110.8 h, commonvoice 39.7 h, fleurs_train 7.2 h.
Candidate total before dedup/carving ≈ 2,570 h (≈ 1.42 M utts). Mean clip ≈ 5–6 s; his clips ≤ 22.7 s.
Text-overlap audit (normalised sentence strings): every one of his files has ZERO overlap with FLEURS test (349) and FLEURS validation (150).
Overlap with our locked dev panel / speaker-disjoint holdout exists for macro (449/422 sentences), macro-filtered (380/336), openslr (1,563/2,281),
kathbath-train (512/616) — because MaCro prompts and OpenSLR share sentence pools. Rows carrying a dev/holdout sentence must be removed from train.
MaCro has NO speaker ids in the release (competition never published them) — speaker-disjoint carving is impossible there; only transcript-group carving.
Audio: his files are 16 kHz mono WAV (to be verified on arrival; the 66 GB bundle is 77 % downloaded).
Owner decisions today: (1) human-labelled + model-filtered only; (2) MaCro-validated 48.9 h -> train minus a ~3–4 h transcript-group-disjoint slice
added to the dev panel; (3) FastConformer-CTC large; (4) 200 GPU-h cap. Owner: "high quality dataset matters, even the author admitted that";
"make sure nothing overlaps"; "his release might not be cleaned properly — you will do it"; "~2k hours would be better"; "more steps".

# QUESTIONS — answer each with numbers and a recommendation; say STOP where warranted
Q1 CORPUS. Which of the new sources go in, at what filtering, and why? Specifically: (a) is MaCro-train-after-Whisper-WER<15 acceptable as
   "human-labelled, model-filtered", and what selection bias does a Whisper-based filter import (easy/short utterances, Whisper-like spelling,
   digit-free text)? (b) Shrutilipi wer15 (142.6 h): in or out, given our Phase-1 blank-collapse evidence on the unfiltered set? (c) should we
   RE-FILTER with our own Phase-2 model (16.5 % FLEURS) rather than trust his 2023 Whisper filter — on all 1.4 M rows (cost: Qwen 1.46 s/clip
   = 570 GPU-h, impossible) or on a stratified sample to ESTIMATE label-noise per source and set per-source policy? Propose the exact cheap
   noise-estimation protocol (sample size per source, model, threshold, decision rule) that fits in ≤ 6 GPU-h with the Phase-1 model
   (~50 clips/s batched).
   (d) dedup rules: exact-path dedup, transcript-group carving vs dev/holdout, cross-source duplicate sentences (MaCro vs OpenSLR prompts),
   MAX_PER_TEXT cap (Phase 1 used 5 rows per transcript) — keep or change with 1.4 M rows? (e) digits: his text has no Bengali digits;
   ours has some — unify how? (f) a max-duration and cps policy for the new sources.
Q2 MIXTURE. Natural mixture would be ≈ 29 % MaCro, 23 % MADASR, 22 % IndicVoices, 8 % OpenSLR, 6 % Shrutilipi, 4 % SUBAK.KO, 4 % Kathbath,
   2 % CommonVoice, 0.3 % FLEURS-train (by hours). FLEURS is read Wikipedia-style sentences; MaCro/MADASR/OpenSLR are read prompts;
   IndicVoices is conversational/extempore. Phase 1 evidence: natural > capped. Any reason to cap or upweight now? Give the decision and
   the ONE pre-registered check that would overturn it.
Q3 INITIALISATION & TOKENIZER. Options: (A) warm-start ALL weights from Phase-1 final.nemo (keep its 1,024 tokenizer; the head already knows
   Bengali; new cosine cycle); (B) Phase-1 encoder + fresh head with a tokenizer retrained on the new 2.5 k h text; (C) English encoder again
   (Phase-1 procedure from scratch). Rank by expected FLEURS WER and GPU-h; state the warmup/peak-LR you would use for each (Phase 1 used
   5e-4 from the English init with a fresh head). Is there a catastrophic-forgetting or plasticity argument against (A)? Any tokenizer-coverage
   evidence you would demand before keeping the 1,024 model (e.g. UNK rate, tokens/s feasibility at 12.5 Hz frames, max_tps 11)?
Q4 HORIZON & BUDGET. With ≈ 1.4 M utts and eff. batch 252, one epoch ≈ 5,600 steps. Phase-1 ran 18 epochs on 984 h and was still improving
   slowly. Throughput ~1,100 steps/h incl. validation. Propose max_steps, warmup, peak LR, min LR, validation cadence, the plateau/early-stop
   rule, and the checkpoint-retention rule, all fitting: pilots + main run + eval ≤ 200 GPU-h with ≥ 20 GPU-h reserve. Give the arithmetic.
Q5 PRE-REGISTERED GATES. Design the cheapest pilot programme (≤ 25 GPU-h total) that de-risks the main run: what to compare (e.g. init A vs B
   at equal exposure; with/without Shrutilipi-wer15; natural vs capped), at how many steps, on which panel, with what numeric pass rule, and
   what "STOP Phase 4" looks like. Phase-1's own dev curve at 2k/3.5k/5k steps was 56.5/40.0/32.8 % — usable as an equal-exposure yardstick
   for init (C); NOT for warm starts.
Q6 EVALUATION SURFACE. Sealed FLEURS test (paired vs Phase 2's 16.54 % AND vs Phase 1's 19.57 %); locked dev panel (must stay comparable
   with Phases 1–3 — but its sentences are now being removed from train, as they were before); new panels: MaCro-validated slice, MADASR
   official dev (1.9 h), Kathbath official valid+test (8.2 h), Phase-1 speaker-disjoint holdout. Which are selection instruments vs one-shot
   reports? What overlap/leakage audits (text AND audio fingerprint) must run before training and what is the pass rule?
Q7 CLAIMS & LICENCES. MaCro is CC-BY-4.0 but the competition rules forbid redistributing the data (training a model is "any purpose");
   RESPIN/MADASR licence is not yet found; OpenSLR-53 is CC-BY-SA (model shipped CC-BY-SA-4.0 in Phase 1). What may the card claim
   ("fine-tuned on ≈ X h human-labelled Bengali"; "model-filtered" must be disclosed), what must it disclose about the Kaggle-derived filtering,
   and is there any licence that BLOCKS use until resolved?
Q8 What is the single most likely way Phase 4 fails to beat 16.54 % or embarrasses the owner, and the one check that prevents it?
Output: Q1–Q8, then a one-page REGISTERED PLAN (corpus rules, mixture, init, schedule, gates with numbers, eval surface, budget table),
then explicit STOP conditions. Do not hedge with "it depends" — pick.
