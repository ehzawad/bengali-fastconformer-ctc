You are an ADVERSARIAL reviewer (gpt-6-astra, xhigh). Your job is to find every way Phase 4 as sketched below goes wrong: leakage, licence blockers, label-noise traps, optimisation traps, budget overruns, unfalsifiable gates, overclaiming. Attack Phase 4 of a Bengali ASR programme
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

# YOUR TASK
Produce a ranked RED-TEAM list (≥ 12 items) of concrete failure modes for this plan, each with: mechanism, evidence from the context that makes it
plausible, probability (%), cost if it happens (pp WER on FLEURS-test or GPU-h or reputational), and the CHEAPEST pre-registered check or rule that
neutralises it (with exact numbers). Then answer: (i) is there any reason the Whisper-WER<15-filtered MaCro/MADASR data could make FLEURS-test WORSE
than Phase 1's 984 h corpus did? (ii) is warm-starting from the Phase-1 checkpoint a trap (forgetting, LR mismatch, BN stats, tokenizer lock-in)?
(iii) which licence, if any, blocks publication until resolved, and what exactly must the HF card say about Kaggle-derived filtering and CC-BY-SA
inheritance? (iv) what leakage audit would you insist on beyond sentence-string matching (audio fingerprints across MaCro/OpenSLR/CommonVoice
prompt pools, FLEURS sentence sources), with pass rules? (v) name the ONE thing you would forbid outright. Do not hedge; pick.
