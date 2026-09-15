You are an independent senior ML-systems reviewer (gpt-6-astra, xhigh) doing an INTERIM audit of Phase 4 of a Bengali ASR programme. You reviewed the design earlier today (councils A and B); the orchestrator has since built the pipeline and measured things. Be concrete, numeric, adversarial; say STOP where warranted. The full registered contract follows, then MEASURED FACTS, then QUESTIONS.

# CONTRACT (experiments/p4/SPEC.md, current)
# Phase 4 — FastConformer-CTC large, continued training on an expanded human-labelled Bengali corpus

Registered 2026-09-05 (UTC) before any GPU work. Reconciled from owner decisions (2026-09-05), council A
(`/tmp/council_p4a`, gpt-6-astra xhigh, design) and council B (`/tmp/council_p4b`, gpt-6-astra xhigh, red team).
Where A and B disagree the orchestrator's ruling is stated and marked **RULING**. This file is the contract; a
number that is not here is not registered.

## 0. Owner decisions (verbatim intent)
- Quality over quantity: "even if you get only 500 high-quality hours, keep it"; "useless YouTube … should not be there".
- Admissible labels: human-labelled; **model-filtered human labels are allowed**; pseudo-labels and TTS audio are **not**.
- Architecture: FastConformer-CTC large (Phase-1 lineage). Warm start allowed.
- Steps scale with the data; GPU-hours are not a hard cap ("unbounded … but not six months"). A soft plan of ~200 A5000 GPU-h
  is registered so that budget is still accounted; the owner may raise it, never the orchestrator.
- MaCro human-validated split: train minus a 3–4 h transcript-group-disjoint dev slice.
- Model licence: **CC-BY-NC-SA-4.0** (owner decision, against council advice; the card discloses the OpenSLR-53 CC-BY-SA and
  Phase-1 CC-BY-SA-4.0 lineage exactly as the Phase-2 card does). **Both the HF model repo and the GitHub repo stay PRIVATE** (owner, 2026-09-05: "keep this one private … later I will make it public and still it will be NC"); only the owner flips visibility.
- Human label audit: owner adjudicates ~100 MaCro-filtered clips (+20 flagged) on a listening board.
- A6000 off-limits; A5000 only (`CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567`); co-tenant never disturbed;
  process anonymisation ("6mn lmu"); hourly loss+WER reporting; self-authored commits.

## 1. Sources and admission (allowlist; nothing else is read)
| bucket | origin of labels | rows / hours (pre-carve) | status |
|---|---|---|---|
| Phase-1 corpus `data2/train_corpus_ready.json` (fleurs_train, openslr53, subakko, kathbath, commonvoice, indicvoices) | as Phase 1 | 552,932 / 984.1 h | IN, **re-audited 2026-09-05 (owner: "don't take the previous pipeline for granted")** — the G0.9 rules now apply to it too (see §3); every Phase-1 exclusion stays excluded |
| `macro_filtered` — Bengali.AI competition `train` split ∩ tugstugi WER<15 list | human-written prompt read by crowd speaker, "algorithmically cleaned" by Bengali.AI, then Whisper-WER<15 filtered (3 rounds) | 550,921 / 736.8 h | IN **iff** the audit in §3 passes |
| `macro_validated` — competition `valid` split (= tugstugi's validated list, 29,583 = all but 5 rows) | manually reviewed and corrected by Bengali.AI annotators | 29,583 / 48.9 h | IN; 3.5 h → new dev panel `M` |
| `madasr` — IISc RESPIN-S1.0 Bengali `train_clean` (2025 release), **whole** | read speech of expert-composed dialect-rich sentences, IISc "clean" slab; CC-BY-4.0 (IISc Terms of Use incorporate CC-BY-4.0: `experiments/p4/licences/RESPIN_Data_TAC.pdf`) | TBD from `meta_train_bn.json` (~87 GB tar) | IN iff audit §3 passes. **AMENDMENT 2026-09-05:** tugstugi's 2023 MADASR id list does not map onto this release (numeric ids; 7/1,183 dev texts match), so no upstream Whisper filter applies to this bucket — disclosed on the card. Official `dev` (1,500 clips / 2.27 h / 100 speakers / 5 dialects) → panel `R` |
| Kathbath official valid (HF `ai4bharat/Kathbath` bengali/valid, converted by `tools/p4/p4_ingest_kathbath_valid.py`) | AI4Bharat | 2,236 / 4.05 h / 17 speakers | panel `K` (selection); Kathbath official test is not on HF → no test one-shot; speaker ids of K are excluded from kathbath training rows |
| `vaani` — ARTPARK-IISc Vaani transcription part, Bengali (owner-approved 2026-09-05) | manually transcribed spontaneous image-description speech (West Bengal districts), transcribers from the same district; CC-BY-4.0, HF gate accepted. Annotation policy (registered after measuring 14,598 rows: 52 % `<noise>` spans, 7 % `<talking>`, 64 % `{brace}`, 5.5 % partial markers): `<span>` tags mark the background condition around the PRIMARY speaker's words → tags deleted, words kept; `[event]` brackets deleted; `{brace}` is an English gloss of the preceding Bengali-script loanword ("লেগ {leg}") → deleted, following case suffix re-attached; rows with `[unintelligible]`/`[inaudible]` DROPPED (partial labels teach deletion, Phase-1 lesson). Residual risk (noise-heavy audio, span semantics) is what the `O+vaani` ablation arm and the 50-clip listening slice test | ~113 h reported | IN iff audit §3 passes (own 50-clip listening slice + Phase-1 disagreement); its official validation/test splits are protected, never trained on |
| `spring_inx` — SPRINGLab SPRING_INX_Bengali_R1 + R2 (owner-approved 2026-09-05) | manually transcribed mock conversations/monologues, vendor-collected with preliminary QC (IIT Madras, NLTM); **CC BY 4.0** (portal text archived: `experiments/p4/licences/SPRING_INX_licence_record.md`) | train 374.73 h + 272.98 h (dev 70 h, eval 10 h protected) | IN iff audit §3 passes (own ablation arm) |
| Shrutilipi (any subset, incl. WER<15) | forced alignment | — | **OUT, 0 rows, 0 GPU-h** (Phase-1 −24.6 pp ablation; both councils) |
| tugstugi YouTube pseudo1–9, bn-google-tts, "long" concatenations | pseudo / synthetic / derived | — | **OUT** |
| tugstugi OpenSLR / Kathbath "extra" rows | rows Phase 1 excluded or never downloaded | — | **OUT** (historical exclusions stand; no provenance for re-admission) |

Text targets: the **original human reference** (competition `train.csv` sentence; RESPIN transcript; Phase-1 manifest text) —
never tugstugi's cleaned CSV text (used only as the membership list of filtered ids). Every target is transformed with the
repo's frozen surface `metric_surface(NFC(text), score_punctuation=False)` + whitespace collapse (punctuation → space, Latin
lower-cased, numerals preserved). **RULING:** this is applied to the old corpus too, so pilot arms share one target convention.

Cleaning for new rows (registered limits, publish removal counts per reason): duration 0.5–23.0 s; 2–35 NFC non-space code
points per second; decodable, finite, non-silent audio; non-empty target; no `<unk>` after tokenisation; CTC feasibility
`L + R ≤ T` and `(L+R)/T ≤ 0.90` with the model's real frame count (12.5 Hz), where R = adjacent equal-token pairs.
If any single new bucket loses >1 % of rows to feasibility, STOP and report (do not silently drop a dialect/text type).

Old-corpus re-audit (2026-09-05, text + 500-file/source header sample): 0 duplicate paths; all audio 16 kHz mono (FLEURS-train files are 32-bit FLOAT WAV and
IndicVoices files are FLAC despite the .wav suffix — Python's `wave` cannot read either, `soundfile` can; every audio tool must use soundfile); durations
match headers. Defects found and now handled by G0.9 for ALL buckets: fleurs_train 84 rows with Latin letters (usgs, a1gp) + 11 with
¥/other symbols; subakko 262 Latin rows ("ভিটামিন d") + 450 math-expression rows ("১+৩+৫+৭=১৬": after punctuation→space the target is
a digit soup nobody spoke) + 68.6 % non-NFC raw text (already NFC'd by the surface); openslr53 3,311 rows below 2 chars/s (3.9 h;
one-word prompts inside long silence — legitimate but outside the registered density band, dropped for uniformity). Numerals: the
FLEURS-test references write 18.3 % of sentences with ASCII digits while every training source writes Bengali digits (or none: MaCro,
RESPIN have no digit rows) — a scorer-level convention mismatch Phase 1 also carried; **registered policy: keep digit-bearing rows as
written, drop math expressions, and report at the end the share of FLEURS-test errors that are digit tokens** (diagnostic only; the
scorer is frozen).

**Provenance finding 2026-09-05 (audio audit):** decoded-PCM hashes show MaCro recordings byte-identical to Common Voice Bengali
recordings in our dev panel D and holdout (362 matches in the first 46k MaCro files hashed) — Bengali.AI's crowd-sourced MaCro
INCLUDES Common Voice Bengali audio. Consequences (registered): (a) every training row whose PCM hash equals a protected-panel
recording is removed; (b) training rows are de-duplicated by PCM hash across ALL sources (MaCro ∩ CommonVoice), keeping one copy
by deterministic hash order; (c) the card must state that MaCro and Common Voice overlap and that the count of unique hours is
after hash de-duplication. **Kathbath panel K:** the HF `valid` split shares speaker ids with Kathbath train (27,230 of 46,240 train
rows share a speaker) — K is a KNOWN-SPEAKER, text-disjoint panel and is labelled so; no training rows are removed for it.

Dedup and carving order: (1) canonical id dedup; (2) decoded-PCM hash dedup; (3) fingerprint dedup across all corpora
(calibrated, §4); (4) remove every training row whose sentence group or audio matches a protected panel; (5) global
`MAX_PER_TEXT = 5` distinct recordings per normalised sentence across all sources (deterministic hash order, never by ASR WER).

## 2. Panels and protected sets (no training row may share a sentence or audio with any of these)
| panel | rows | role |
|---|---|---|
| `D` locked dev `data2/val_corpus.json` | 5,385 | selection + historical comparison (P2 17.03 / P1 20.35 / hishab 22.30, frozen surface) — reported separately, always |
| `M` MaCro validated slice | ≈3.5 h, ≥500 sentence groups, ≤3 recordings/group, groups chosen by seed 20260905 **from sentences with no historical exposure** (not in any Phase-1/2/3 train manifest or tokenizer text) | selection |
| `R` RESPIN official dev (`data4/respin/manifest_madasr_dev.jsonl`) | 1,500 (494 distinct sentences) | selection (only if MADASR admitted) |
| `K` Kathbath official valid (`data4/kathbath/manifest_valid.jsonl`) | 2,236 | selection |
| Kathbath official test; `data2/holdout_corpus.json` (speaker-disjoint IndicVoices-R); conversational slice of D | — | one-shot reports |
| **FLEURS official test** (`bench_assets/eval_fleurs_bn`, split==test, 920 recordings / 349 sentences) | 920 | sealed headline, decoded once |
| FLEURS validation | — | never used (contaminated) |

Selection score (frozen-surface corpus WER, pp): `S = 0.60·D + 0.20·M + 0.10·R + 0.10·K` if MADASR is admitted, else
`S = 0.60·D + 0.30·M + 0.10·K`. Regression guards: no D source worse than P1 by >1.0 pp; conversational slice of D worse by
≤0.5 pp; empty-output rate ≤0.1 %. **Definitions fixed 2026-09-05 (trainer workstream):** conversational slice = D rows with source indicvoices and scenario "Conversation" (774 rows, 6,628 reference words); Phase-1 baselines measured through the new evaluator and stored in `experiments/p4/baseline_p1.json` (D 20.35 frozen; per source commonvoice 19.21 / indicvoices 16.57 / kathbath 12.48 / openslr53 18.10 / subakko 36.83; conversational 21.58; K 14.67; R 32.06). NeMo-internal surface numbers are NOT comparable with the Phase-1 in-training curve (different dataloader filters); only the frozen surface is used for gates. Measured throughput on a smoke manifest: ~745 steps/h (SPEC planning figure 1,000 was optimistic; re-measure on the real arm manifests).

## 3. Pre-GPU gates (Gate 0, CPU only)
- G0.1 Provenance: 100 % of admitted rows trace to an original human reference + allowlisted transformation; counts by source.
- G0.2 Licence record for every source archived under `experiments/p4/licences/`; MADASR admitted only with the IISc terms file.
- G0.3 Leakage audit (text): frozen surface + audit keys (punctuation-deleted, digit-script-folded, joiner-stripped) + fuzzy
  candidates (char similarity ≥0.90 for strings ≥20 chars) against every protected panel → **zero unresolved matches**.
- G0.4 Leakage audit (audio): decoded-PCM SHA-256 + spectral fingerprint (calibrated on 200 train-only clips × 6 transforms,
  ≥99 % recovery per transform, 0 false confirmations on 10,000 unrelated + 1,000 same-prompt pairs) train × protected panels
  → zero confirmed or unresolved matches. Test-set matching is run as a fixed audit returning only ids/counts.
- G0.5 Tokenizer: Phase-1 `data/tokenizer_v1024_corpus` (sha256 6288cf8e…) unchanged; zero `<unk>` in retained targets;
  tokens/s p50/p95/p99 by source; feasibility rule of §1.
- G0.6 Human audit (owner): listening board of 100 uniformly random `macro_filtered` clips + 20 worst by Phase-1
  disagreement (+ 50 random `madasr` clips). Material error = >10 % WER of the reference vs what is heard; catastrophic =
  wrong pairing / non-speech / gross truncation. **Pass: ≤5 material per 100 clips of the uniformly-random stratum (the 20 worst-by-disagreement and the 50 madasr clips count only towards the 0-catastrophic clause) and 0 catastrophic anywhere.** A failed bucket
  is excluded whole. Machine disagreement (Phase-1 greedy, ≤0.5 GPU-h) is reported as *ASR-disagreement*, not label noise.
- G0.9 **Uniformity gate per bucket (owner principle 2026-09-05: "if a dataset is not uniform, precise and clean after the full
  process, jettison it")**. After ingestion and the frozen-surface transform, every bucket is profiled and must satisfy ALL:
  (a) ≥99 % of rows contain only Bengali script, ASCII/Bengali digits and spaces — rows with Latin letters or other scripts are
  removed (FLEURS references write foreign names in Bengali script; Latin targets are a convention mismatch), and a bucket losing
  >5 % of its rows to this rule is flagged for owner review; (b) empty/duplicate-audio/decode failures <0.5 %; (c) duration-vs-metadata
  mismatches <1 %; (d) the tokens/s p99 ≤ 11 and feasibility loss ≤1 %; (e) a 200-row random sample of targets read by the orchestrator
  shows one consistent transcription convention (no residual tags, no transliteration/gloss doubling, punctuation removed uniformly).
  A bucket that fails any clause is **excluded whole for this phase** — no row-level repair beyond the registered transforms, no
  threshold tuning. Passing G0.9 is necessary, not sufficient: the human audit (G0.6) and the bucket's ablation arm (§5) still apply.
- G0.7 Resource: profile the 21 longest admitted clips at microbatch 12; peak reserved ≤16.5 GiB; watchdog 17.0 GiB.
- G0.8 Admission: ≥500 non-duplicate new hours after all of the above; otherwise STOP Phase 4 and report.

## 4. Initialisation and optimisation (registered)
- Init **A**: all weights of Phase-1 `experiments/fastconformer_ctc_bn_1kh/final.nemo` (sha256 61362a36…; 18 layers, d 512,
  dw-striding 8×, conv BatchNorm, kernel 9; head 1,024 + blank), tokenizer unchanged, BN buffers preserved; **fresh AdamW state,
  fresh scheduler**. Zero-update restoration check: dev WER of the loaded model equals Phase-1's recorded 20.35 % (frozen) ±0.05.
- AdamW (0.9, 0.98) wd 1e-3, clip 1.0, bf16, SpecAug 2×15 / 2×0.05 (Phase-1 values), effective batch 252 = 12 × 21.
- Schedule: warmup 1,000 steps 1e-5 → **peak 2e-4** (RULING between A's 2e-4 and B's 1e-4: Phase 1 tolerated 5e-4 from a
  fresh head; 2e-4 is already a 2.5× reduction), cosine to 1e-5 at **H = 150,000** steps (RULING between A 150k and B 100k:
  ≈27 epochs of 1.4 M rows; owner asked for steps to scale with data; plateau rule below stops earlier if warranted).
- Validation every 2,000 steps (all panels, frozen surface + NeMo-internal for D). Checkpoints: last 2 resumable, best 3 by S,
  immutable milestones (pilot end, 20k, 40k, 80k, 120k, H).
- Progress gate at 20,000: S ≤ S(P1) − 1.5 pp and D ≤ D(P1) − 1.0 pp and guards pass; else STOP.
- Plateau: from 30,000, stop after 5 consecutive validations (10k steps) without ≥0.20 pp improvement of the best S.
- Final checkpoint = lowest eligible S; ties within 0.05 pp → earlier. No averaging, no decoder search.
- Resume semantics: audited — interrupted vs uninterrupted 1,000-step replay across an epoch boundary must give identical
  sample ids and LRs (Phase-1 kill-drill tooling).

## 5. Pilots — per-bucket ablation (owner directive 2026-09-05: "check all the new data we will be adding")
All arms start from init A with the same schedule prefix; **8,000 steps each = 2.016 M presentations** (equal exposure); validation
at 2k/4k/6k/8k on D, M, K, R (+ Vaani/SPRING official validation splits as one-shot reports, never selection).
| arm | corpus |
|---|---|
| O control | Phase-1 corpus only (re-normalised targets) |
| O+macro_filtered | O + `macro_filtered` |
| O+macro_validated | O + `macro_validated` (train part, M carved out) |
| O+madasr | O + `madasr` |
| O+vaani | O + `vaani` (if downloaded and audited in time; else deferred to a later registered arm) |
| O+spring_inx | O + `spring_inx` (only if licence confirmed; else not run) |
| N_final | O + every bucket that passed its arm; its own 8k point is compared with O and with the best single-bucket arm |
Stop an arm at any validation for non-finite loss, >1 % empties, or D worse than P1 by >5 pp.
**Bucket pass rule at 8,000** (all required): S ≤ S(O) − 0.3 pp with paired sentence-clustered 95 % CI upper bound < 0 (10,000 draws,
seed 20260905); D ≤ D(O) + 0.3 pp; no D source worse than O by >1.0 pp; conversational slice worse by ≤0.5 pp; empties ≤0.1 %.
A bucket that fails is **excluded from N_final** (no re-tuning, no partial admission). N_final is admitted to the long run iff it
passes the same rule against O **and** S(N_final) ≤ S(best single arm) + 0.2 pp (adding buckets must not hurt). If N_final fails but
a single arm passed, the long run uses that single arm's corpus (registered fallback). If nothing passes → **STOP Phase 4**; O is not
promoted; FLEURS stays sealed. The chosen arm resumes from its own 8k checkpoint (optimizer, sampler, RNG) to H with no second warmup.
Capped-mixture arm C (council A) is dropped in favour of this design; a 30 % cap may only be registered later as a new arm.
Pilot cost ≈ 7 × 8 GPU-h ≈ 56 GPU-h at 1,000 steps/h (owner: GPU-hours are soft).

## 6. Headline (one event)
Freeze checkpoint, tokenizer, greedy CTC decoding and scorer; decode 920 FLEURS-test recordings once through the benchmark's
`asr_core` staging; paired sentence-cluster bootstrap (349 clusters, 20,000 draws, seed 20260905, corpus WER = Σedits/Σref words)
against frozen Phase-2 (16.54 %) and Phase-1 (**19.67 %** under the frozen scorer on the 920 test rows — the 19.57 % quoted earlier was a different harness; corrected 2026-09-05 after re-scoring `outputs_a5000/ehzawad_fastconformer/predictions.json`, which the benchmark summary also records as 0.19672) hypotheses. **Success: WER < 16.54 % and the 95 % CI of (P4 − P2) lies
entirely below 0.** "Strong": ≤16.04 %. Report CER, per-panel one-shots, latency (same protocol as Phase 3 SPEC Q4). No second decode.

## 7. Budget ledger (soft; owner may raise)
| item | GPU-h |
|---|---|
| Gate-0 GPU work (disagreement decode, restoration check, profiling) | ≤ 6 |
| Pilots ≤7 × 8k (per-bucket ablation) | ≤ 60 |
| Main run to H incl. validation | ≤ 150 |
| Final eval, export, latency | ≤ 8 |
| Reserve | 12 |
| **Total plan** | **~236 (soft; owner: unbounded but not months)** |

## 8. Claims discipline (card)
"Continued fine-tuning of the Bengali FastConformer-CTC checkpoint on X unique hours of human-labelled Bengali speech; Y hours were
selected upstream by ASR/reference agreement (Whisper, WER<15, 3 rounds, digit rows dropped upstream); no pseudo-label targets or
TTS audio." Disclose: per-source table (creator, version, licence, hours, exclusions), the human audit result, MaCro speaker
separation unknown, FLEURS test history (scored in Phases 1–3), single Phase-4 scoring event, private pipeline, NC-SA vs CC-BY-SA
tension (owner decision). Never: "fully manually verified", "speaker-disjoint everywhere", "SOTA", "reproducible from the public repo".

## 9. Log
- 2026-09-05: owner decisions recorded; councils A/B run; MaCro competition bundle (24.7 GB, 963,636 mp3, 32 kHz mono 48 kbps)
  and IISc RESPIN `train_bn_clean` (87.1 GB) downloads started; tugstugi CSV inventory and 0-overlap-with-FLEURS audit done
  (`data4/tugstugi/csv`). tugstugi's `validated` list == competition `valid` split (29,583 of 29,588); his filtered list ⊂ `train`.

# MEASURED FACTS SINCE THE DESIGN COUNCILS (all from tools run today)
- Old Phase-1 corpus re-audit (owner insisted nothing be taken for granted): 0 duplicate paths; audio all 16 kHz mono (FLEURS-train 32-bit float WAV, IndicVoices FLAC in .wav); durations match headers on 500-file/source samples. Text defects: fleurs_train 73 Latin-letter rows + 8 symbol rows + 20 "math" rows; subakko 253 Latin ("ভিটামিন d"), 370 symbol rows, 87 math-expression rows ("১+৩+৫+৭=১৬"), 68.6 % non-NFC raw; openslr53 3,317 rows under 2 chars/s (one-word prompts in long silence; 3.9 h), 102 math; commonvoice 30 math; indicvoices 251 under 2 cps. FLEURS-test references: 18.3 % of sentences contain ASCII digits, 5.4 % Latin letters; every training source writes Bengali digits or has no digits (MaCro and RESPIN contain NO digit rows at all).
- Applying G0.9 to the old corpus: O = 544,031 rows / 972.6 h (from 552,932 / 984.1); removals per source in the builder report; MAX_PER_TEXT=5 removed 387 openslr + 2,778 indicvoices rows ("হুম ঠিক আছে" x103).
- Text leakage audit (agent tool): exact + punctuation/digit-folded keys + fuzzy (char-5-gram bound, similarity ≥0.90, ≥20 chars) over old train and macro_validated vs D, holdout, K, R, FLEURS test/validation. Fuzzy near-duplicates found in OLD train vs D: 163 rows, vs holdout 721, vs K 17 (Phase 1 carved exact matches only). FLEURS test: 0 matches on every key. 1,752 training rows listed as candidates; the builder now removes ALL listed rows (no adjudication).
- Audio leakage audit: landmark-constellation fingerprint calibrated on 200 clips × 6 transforms: recovery 100 % on 5 transforms and 99.5 % on speed 1.05; 0 false confirmations on 39,800 unrelated + 1,000 same-prompt pairs. Decoded-PCM SHA-256 over the first 46,672 files: **362 MaCro-validated recordings are byte-identical to Common Voice recordings sitting in our dev panel D (192) and holdout (170)** — Bengali.AI's MaCro contains Common Voice Bengali audio. Builder now removes any training row whose hash equals a protected-panel recording and de-duplicates training rows by hash across sources.
- Kathbath: the HF `valid` split shares speaker ids with train (27,230 of 46,240 Kathbath train rows) → panel K relabelled "known-speaker, text-disjoint"; no training rows removed for it.
- MaCro validated ingested: 29,462 / 48.9 h (109 duration mismatches vs tugstugi's CSV, 12 density outliers removed). MaCro filtered: converting (rate limited by a saturated disk). RESPIN train_clean (87 GB) queued. RESPIN dev = 1,500 clips / 2.27 h / 100 speakers / 5 dialects / 494 distinct sentences (panel R). Kathbath valid = 2,236 / 4.05 h / 17 speakers (panel K).
- tugstugi's MADASR id list does NOT map to the 2025 RESPIN release (7/1,183 dev texts match; his ids are numeric) → RESPIN goes in WHOLE (IISc "clean" slab), no upstream filter.
- Vaani Bengali (ARTPARK-IISc, CC-BY-4.0, ~113–154 h transcribed, image-description spontaneous speech from West Bengal districts) downloaded. Transcript annotation measured on 14,598 rows: 52 % `<noise>` spans, 7 % `<talking>`, 64 % `{brace}`, 5.5 % `[unintelligible]/[inaudible]`. Inspection: spans wrap the primary speaker's words and name the background condition; `{brace}` is an ENGLISH GLOSS of the preceding Bengali-script loanword ("লেগ {leg} পিস {Piece} -এর"), not a spoken word. Registered policy: tags deleted/words kept, braces deleted with suffix re-attached, partial-marker rows dropped.
- SPRING-INX Bengali R1+R2 (IIT Madras, CC BY 4.0 confirmed from the portal text; manually transcribed mock conversations/monologues, vendor-collected with "preliminary" QC; train 374.7 + 273.0 h; text contains plain Latin code-switch words like "doggy", "pt") downloading; owner approved as a bucket with its own ablation arm.
- Owner directives added today: per-bucket ablation arms ("check all the new data"), jettison any bucket that is not uniform/clean after processing (G0.9), GPU-hours soft, HF+GitHub private, licence NC-SA anyway.
- Trainer built and verified: zero-update restoration on D = 20.346 % frozen (target 20.35 ± 0.05); BN buffers preserved and unchanged by eval; 40-step interrupted-vs-uninterrupted replay identical (cut ids, LRs, loss Δ 0.0); watchdog fault-injection passes; peak reserved 8.49 GiB at microbatch 12 on the 21 longest clips; **measured throughput ≈ 745 steps/h** (SPEC planned 1,000) → each 8k-step arm ≈ 10.7 GPU-h, 7 arms ≈ 75 GPU-h, main run 142k steps ≈ 190 GPU-h.
- Eval tools verified: P1 hyps re-score to 19.67 % on FLEURS test (SPEC corrected from 19.57); P2 to 16.54 %; paired bootstrap fixtures pass; listening board built.

# QUESTIONS
Q1 Old-corpus G0.9 changes: is dropping openslr rows <2 cps (3.9 h of one-word prompts) and Latin/symbol/math rows from the CONTROL arm O acceptable, given O must stay comparable with Phase 1? Would you keep O = literal Phase-1 corpus and apply G0.9 only to new buckets? Pick.
Q2 Numerals: FLEURS-test writes 18 % of sentences with ASCII digits; training data writes Bengali digits or none. Options: (a) keep as registered (digits kept as written, diagnostic only); (b) drop all digit-bearing training rows (precision principle); (c) map Bengali digits → ASCII digits in training targets (a reversible, deterministic convention change that matches the scorer's reference convention). Pick, with the expected effect on FLEURS WER.
Q3 MaCro ⊇ Common Voice: beyond hash removal against panels and hash dedup, what else must be checked (e.g. MaCro filtered vs our CommonVoice TRAIN rows — same audio twice with possibly different transcripts; which transcript wins)? Should MaCro rows that duplicate Common Voice audio be dropped in favour of the CV row or vice versa?
Q4 Kathbath K known-speaker: keep in S at 0.10, or drop K from S and keep it as a one-shot report? Pick.
Q5 SPRING-INX plain-Latin code-switch words vs G0.9 (a) "rows with Latin letters removed": estimate the loss and say whether the rule should stay or SPRING should be jettisoned if >5 % rows are lost.
Q6 Per-bucket ablation design (7 arms × 8k) at 745 steps/h ≈ 75 GPU-h: keep, or shorten arms to 6k / drop O+macro_validated (45 h, small) into N_final directly? Any statistical flaw in "bucket passes iff S ≤ S(O) − 0.3 pp with paired CI < 0" given arms differ in corpus size (more data = more steps per epoch... no — equal steps, so fewer epochs over a larger corpus)? Say how to interpret a bucket that is neutral (CI includes 0).
Q7 Near-duplicate removal: 1,752 training rows dropped on fuzzy candidates without adjudication — acceptable, or must candidates be adjudicated? Anything the text audit still misses (transliteration variants, split/merged sentences)?
Q8 Name the single most likely remaining defect in this pipeline and the check for it. Output: Q1–Q8 with picks, then a list of REQUIRED changes to SPEC (exact wording) before the pilots start.
