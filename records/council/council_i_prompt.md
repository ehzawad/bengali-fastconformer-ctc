You are the ablation-design reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A-H shaped the contract; MaCro is admitted; madasr and vaani finish their pilots within hours; the warm-start continuation resumes ~03:00 UTC. The OWNER has now asked, in the middle of that, whether the registered warm start from Phase 1 is actually better than a cold start from the NVIDIA English model, and has asked for an ablation to be registered and reconciled. Be concrete, numeric, adversarial. Below: the CONTRACT, then FACTS (measured, JSON), then QUESTIONS.

# CONTRACT (experiments/p4/SPEC.md)
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
- **GPU policy amended 2026-09-06 (owner):** the main run stays on the A5000 (`CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567`); pilot/ablation arms MAY also run on the A6000 with a **20 GB (decimal = 18.6 GiB) cap; watchdog 18.0 GiB**; foreign-compute polling every ≤60 s with a resumable checkpoint + card release within 5 minutes of detection, one arm per card, only while no foreign compute process is on that card (checked before launch and by the hourly watcher; a foreign process appearing = our arm pauses at the next checkpoint, never the other way round); co-tenant never disturbed;
  process anonymisation ("6mn lmu"); hourly loss+WER reporting; self-authored commits.

## 1. Sources and admission (allowlist; nothing else is read)
| bucket | origin of labels | rows / hours (pre-carve) | status |
|---|---|---|---|
| Phase-1 corpus `data2/train_corpus_ready.json` (fleurs_train, openslr53, subakko, kathbath, commonvoice, indicvoices) | as Phase 1 | 552,932 / 984.1 h | IN, **re-audited 2026-09-05 (owner: "don't take the previous pipeline for granted")** — the G0.9 rules now apply to it too (see §3); every Phase-1 exclusion stays excluded |
| `macro_filtered` — Bengali.AI competition `train` split ∩ tugstugi WER<15 list | human-written prompt read by crowd speaker, "algorithmically cleaned" by Bengali.AI, then Whisper-WER<15 filtered (3 rounds) | 550,921 / 736.8 h | IN **iff** the audit in §3 passes |
| `macro_validated` — competition `valid` split (= tugstugi's validated list, 29,583 = all but 5 rows) | manually reviewed and corrected by Bengali.AI annotators | 29,583 / 48.9 h | **EXCLUDED FROM TRAINING (owner ruling 2026-09-05, strict G0.9(b): 362/29,583 = 1.22 % of its clips are byte-identical to Common Voice recordings in D/holdout, above the 0.5 % duplicate ceiling).** Used ONLY as the source of dev panel `M` (carved from historically unexposed, hash-clean sentence groups) |
| `madasr` — IISc RESPIN-S1.0 Bengali `train_clean` (2025 release), **whole** | read speech of expert-composed dialect-rich sentences, IISc "clean" slab; CC-BY-4.0 (IISc Terms of Use incorporate CC-BY-4.0: `experiments/p4/licences/RESPIN_Data_TAC.pdf`) | **645,722 / 894.2 h**, 1,791 speakers, 5 dialects, 20,038 distinct sentences (ingested 2026-09-06; 49 rows outside the duration band) | IN iff audit §3 passes. **AMENDMENT 2026-09-05:** tugstugi's 2023 MADASR id list does not map onto this release (numeric ids; 7/1,183 dev texts match), so no upstream Whisper filter applies to this bucket — disclosed on the card. Official `dev` (1,500 clips / 2.27 h / 100 speakers / 5 dialects) → panel `R` |
| Kathbath official valid (HF `ai4bharat/Kathbath` bengali/valid, converted by `tools/p4/p4_ingest_kathbath_valid.py`) | AI4Bharat | 2,236 / 4.05 h / 17 speakers | panel `K` (selection, weight 0.10) — a KNOWN-SPEAKER, text-disjoint panel (shares speakers with Kathbath train); no rows are removed SOLELY for shared speaker identity (all protected sentence/audio exclusions still apply); Kathbath official test is not on HF → no test one-shot |
| `vaani` — ARTPARK-IISc Vaani transcription part, Bengali (owner-approved 2026-09-05) | manually transcribed spontaneous image-description speech (West Bengal districts), transcribers from the same district; CC-BY-4.0, HF gate accepted. Annotation policy (registered after measuring 14,598 rows: 52 % `<noise>` spans, 7 % `<talking>`, 64 % `{brace}`, 5.5 % partial markers): `<span>` tags mark the background condition around the PRIMARY speaker's words → tags deleted, words kept; `[event]` brackets deleted; `{brace}` is an English gloss of the preceding Bengali-script loanword ("লেগ {leg}") → deleted, following case suffix re-attached; rows with `[unintelligible]`/`[inaudible]` DROPPED (partial labels teach deletion, Phase-1 lesson). Residual risk (noise-heavy audio, span semantics) is what the `O+vaani` ablation arm and the 50-clip listening slice test | **train 86,861 / 118.1 h ingested (117.4 h retained after carving/caps)** (3,953 partial-marker rows dropped); validation 10,105 / 13.6 h and test 12,695 / 16.1 h are PROTECTED one-shot panels | G0.6 PASS after the owner's clarification (verdicts had targeted the Phase-1 hint); arm runs | IN iff audit §3 passes (own 50-clip listening slice + Phase-1 disagreement); its official validation/test splits are protected, never trained on |
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
RESPIN have no digit rows) — a scorer-level convention mismatch Phase 1 also carried; **registered policy (council C #2, #11): keep digit-bearing rows as
written in every arm, drop math expressions, no further headline-driven target or scorer changes, and report after the single
scoring event the share of FLEURS-test errors that are digit tokens** (diagnostic only). Vaani/SPRING one-shot results are produced
only after final checkpoint selection and never influence admission or selection.

**Provenance finding 2026-09-05 (audio audit):** decoded-PCM hashes show MaCro recordings byte-identical to Common Voice Bengali
recordings in our dev panel D and holdout (362 matches in the first 46k MaCro files hashed) — Bengali.AI's crowd-sourced MaCro
INCLUDES Common Voice Bengali audio. Consequences (registered): (a) every training row whose PCM hash equals a protected-panel
recording is removed; (b) training rows are de-duplicated by PCM hash across ALL sources (MaCro ∩ CommonVoice), keeping one copy
by deterministic hash order; (c) the card must state that MaCro and Common Voice overlap and that the count of unique hours is
after hash de-duplication. **Kathbath panel K:** the HF `valid` split shares speaker ids with Kathbath train (27,230 of 46,240 train
rows share a speaker) — K is a KNOWN-SPEAKER, text-disjoint panel and is labelled so; no training rows are removed for it.

**Freeze and precedence (council C #1, #2, registered):** before any pilot update, the source inventories, protected-panel manifests,
transformation code, exclusion ledger and final arm manifests are frozen with checksums. Deduplication and MAX_PER_TEXT reserve
O's retained recordings FIRST; new rows occupy only remaining slots, so every O+bucket arm contains exactly the frozen O rows and
targets. For each audio component (rows sharing a decoded-PCM hash or a confirmed fingerprint match): every alias and every original
reference is checked against protected sets and historical exclusions; any protected match excludes every member; if the
frozen-surface references agree, the already-admissible O row is kept and new-source copies count zero new hours; if they disagree
materially the whole component is excluded for this phase (ASR agreement never selects the reference). G0.9 admission rates use
the ingested candidate count as denominator and are reported both pre-removal and retained. Arithmetic expressions are detected on
the NFC original before punctuation removal. Reported O/N counts are provisional until every audit finishes.

Dedup and carving order: (1) canonical id dedup; (2) decoded-PCM hash dedup; (3) fingerprint dedup across all corpora
(calibrated, §4); (4) remove every training row whose sentence group or audio matches a protected panel; (5) global
`MAX_PER_TEXT = 5` distinct recordings per normalised sentence across all sources (deterministic hash order, never by ASR WER), O's retained rows reserved first. **RESPIN exception (council D Q2, registered before the build):** `madasr` may add at most 5 distinct speakers per (sentence, official dialect) cell, with ≤25 retained recordings per sentence across all sources; missing dialect metadata gets no expanded allowance; the sentence×dialect occupancy table is published.

## 2. Panels and protected sets (no training row may share a sentence or audio with any of these)
| panel | rows | role |
|---|---|---|
| `D` locked dev `data2/val_corpus.json` | 5,385 | selection + historical comparison (P2 17.03 / P1 20.35 / hishab 22.30, frozen surface) — reported separately, always |
| `M` MaCro validated slice | ≈3.5 h, ≥500 sentence groups, ≤3 recordings/group, groups chosen by seed 20260905 **from sentences with no historical exposure** (not in any Phase-1/2/3 train manifest or tokenizer text) | selection |
| `R` RESPIN official dev (`data4/respin/manifest_madasr_dev.jsonl`) | 1,500 (494 distinct sentences) | selection (only if MADASR admitted) |
| `K` Kathbath official valid (`data4/kathbath/manifest_valid.jsonl`) | 2,236 | selection |
| `data2/holdout_corpus.json` (speaker-disjoint IndicVoices-R — read as *disjoint from* IndicVoices-R; it contains none, see §8n(v)); Vaani official validation (10,105) / test (12,695); SPRING R1/R2 validation and test | — | one-shot reports after final checkpoint selection (manifest hashes recorded at freeze) |
| conversational slice of D (indicvoices, scenario Conversation, 774 rows) | 774 | repeatedly evaluated selection GUARD (not a one-shot) |
| **FLEURS official test** (`bench_assets/eval_fleurs_bn`, split==test, 920 recordings / 349 sentences) | 920 | sealed headline, decoded once |
| FLEURS validation | — | never used (contaminated) |

Selection score (frozen-surface corpus WER, pp): `S = 0.60·D + 0.20·M + 0.10·R + 0.10·K` if MADASR passes Gate 0, else
`S = 0.60·D + 0.30·M + 0.10·K`. **The weight vector is fixed once after Gate 0 and used for every arm and checkpoint thereafter, even if MADASR later fails its arm.** Bootstrap: paired resamples across systems; sentence clusters shared across panels receive coupled resampling weights before S is formed. Regression guards: no D source worse than P1 by >1.0 pp; conversational slice of D worse by
≤0.5 pp; empty-output rate ≤0.1 %. **Definitions fixed 2026-09-05 (trainer workstream):** conversational slice = D rows with source indicvoices and scenario "Conversation" (774 rows, 6,628 reference words); Phase-1 baselines measured through the new evaluator and stored in `experiments/p4/baseline_p1.json` (D 20.35 frozen; per source commonvoice 19.21 / indicvoices 16.57 / kathbath 12.48 / openslr53 18.10 / subakko 36.83; conversational 21.58; K 14.67; R 32.06). NeMo-internal surface numbers are NOT comparable with the Phase-1 in-training curve (different dataloader filters); only the frozen surface is used for gates. Measured throughput on a smoke manifest: ~745 steps/h (SPEC planning figure 1,000 was optimistic; re-measure on the real arm manifests).

## 3. Pre-pilot gates (Gate 0: CPU audits + the explicitly listed GPU qualification: restoration check, disagreement decode, longest-clip profiling, cross-card check, 1,000-update replay)
- G0.1 Provenance: 100 % of admitted rows trace to an original human reference + allowlisted transformation; counts by source.
- G0.2 Licence record for every source archived under `experiments/p4/licences/`; MADASR admitted only with the IISc terms file.
- G0.3 Leakage audit (text): frozen surface + audit keys (punctuation-deleted, digit-script-folded, joiner-stripped) + fuzzy
  candidates (char similarity ≥0.90 for strings ≥20 chars) against every protected panel → **zero unresolved matches**.
- G0.3/G0.4 coverage (council C #5): audits run over EVERY candidate bucket (incl. Vaani/SPRING protected splits) and candidate M is
  audited against the complete Phase-1/2/3 training audio (PCM + fingerprint) and sentence/tokenizer text; M must still meet its
  hours/group minimum afterwards, else STOP. Historical exposure is recorded separately from current-training overlap.
- G0.4 fingerprint rule (registered 2026-09-06 before the full match completed, from calibration alone): frozen thresholds
  MIN_HITS 12, confirmed ≥24 hits with ≥2 s span; calibration negatives (39,800 unrelated + 1,000 same-prompt + 20,000
  operational) never exceeded 11 hits. Any training row with a match ≥12 hits to any protected panel is EXCLUDED (confirmed
  or unresolved); a panel-M recording matched by an old-corpus query is removed from M as historically exposed; M is then
  FIXED to the indexed set (`panel_M.indexed_by_fingerprint.jsonl`) minus those rows and is not re-selected.
- G0.4 Leakage audit (audio): decoded-PCM SHA-256 + spectral fingerprint (calibrated on 200 train-only clips × 6 transforms,
  ≥99 % recovery per transform, 0 false confirmations on 10,000 unrelated + 1,000 same-prompt pairs) train × protected panels
  → zero confirmed or unresolved matches. Test-set matching is run as a fixed audit returning only ids/counts.
- G0.5 Tokenizer: Phase-1 `data/tokenizer_v1024_corpus` (sha256 6288cf8e…) unchanged; zero `<unk>` in retained targets;
  tokens/s p50/p95/p99 by source; feasibility rule of §1.
- G0.6 **AMENDED by the owner 2026-09-06** after reviewing the 230 board clips informally ("dataset is kinda okay"; no
  verdict export): the human verdict gate is replaced by a MACHINE CONSENSUS audit (`tools/p4/p4_machine_audit.py`) using two
  independent recognisers (Phase-1 FastConformer and hishab/titu_stt_bn_conformer_large): a clip is a consensus disagreement
  when both are >30 % WER from the reference but <15 % WER from each other; catastrophic-like when both are >80 % CER from it.
  Registered rule (before results): a new bucket passes iff its consensus-disagreement rate ≤ max(2× old-corpus rate, old + 2 pp)
  and its catastrophic-like rate ≤ old + 0.5 pp, over 1,000 random retained clips per bucket + the 230 board clips, with 500 clips
  per old source as the baseline. The card must state that label quality was machine-audited and owner-spot-checked, not
  human-verified. Original G0.6 text kept below for the record.
- G0.6 (superseded) Human audit (owner): listening board of 100 uniformly random `macro_filtered` clips + 20 worst by Phase-1
  disagreement, PLUS 50 uniformly sampled clips from the final retained manifest of EVERY other new bucket (madasr, vaani,
  spring_inx); each 50-clip stratum passes with ≤2 material errors and 0 catastrophic (council C #7). Judgments are published
  before the bucket's pilot. Machine disagreement is never a substitute for the human audit. Material error = >10 % WER of the reference vs what is heard; catastrophic =
  wrong pairing / non-speech / gross truncation. **Pass: macro random stratum ≤5 material / 100; every other new bucket ≤2 material / 50; 0 catastrophic anywhere, including the 20 flagged macro clips (council D Q7 — no exemption).** A failed bucket
  is excluded whole. Machine disagreement (Phase-1 greedy, ≤0.5 GPU-h) is reported as *ASR-disagreement*, not label noise.
- G0.9(a) **ruling (council D Q4):** script violations are counted on the incoming candidate rows; >1 % = the bucket FAILS (the >5 % owner-review flag adds a report, it does not override the failure). Measured: SPRING-INX R1 48.3 % of rows carry Latin-script code-switch words → **SPRING-INX is EXCLUDED for this phase** (owner may amend explicitly with a replacement threshold and the stated bias). Rows are never partially repaired (no deletion of Latin words, no transliteration).
- G0.9(b) **ruling (owner 2026-09-05): the duplicate rate is the INCOMING rate — duplicates against protected panels AND against other sources count, denominator = ingested candidate rows — and cross-corpus copies are NOT exempt.** Consequence already applied: macro_validated excluded from training; macro_filtered, madasr, vaani, spring_inx are judged by the same rule once fully hashed.
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
- Plateau (council D Q5): at step 50,000 initialise an anchor = best eligible S so far; reset anchor and counter whenever an eligible checkpoint improves the anchor by ≥0.20 pp cumulatively, else increment at each validation; stop after 10 consecutive non-reset validations (20k-step patience, earliest stop 70k). H stays 150,000. **The parenthetical that stood here, "≈16–18 epochs of the final corpus; Phase 1 ran 18", is SUPERSEDED by §8f and kept only for the record: at effective batch 252 the true figure is 29.9 nominal epochs of N_final (1,265,884 rows / 5,023 updates per epoch) and 34.8 if MaCro alone wins (1,084,696 rows / 4,304 per epoch). H and every stopping rule are unchanged.**
- Final checkpoint = lowest eligible S; ties within 0.05 pp → earlier. No averaging, no decoder search.
- Resume semantics: audited — interrupted vs uninterrupted **1,000-step replay across an epoch boundary** must give identical
  sample ids and LRs (the 40-step smoke replay passed on 2026-09-05 but does NOT satisfy this; the 1,000-step replay is a Gate-0
  GPU item, ≈1.5 GPU-h, charged to Gate 0). Resource profiling is repeated on the final admitted manifests' longest clips.

## 5. Pilots — per-bucket ablation (owner directive 2026-09-05: "check all the new data we will be adding")
All arms start from init A with the same schedule prefix; **8,000 steps each = 2.016 M presentations** (equal exposure); validation
at 2k/4k/6k/8k on D, M, K, R. Official Vaani/SPRING splits are decoded only after final checkpoint selection.
| arm | corpus |
|---|---|
| O_ref control (run id `O_A5000`) | Phase-1 corpus only (re-normalised targets); trained on the A5000 to update 7,774 and on the A6000 to 8,000 after a co-tenant took the A5000 — the completed mixed-card reference control used by every admission comparison |
| O_A6000 control | identical O on the A6000 (council D Q1): a 200-update cross-card check first (identical cut order, target hashes, LRs; ≤1 % relative difference of 20-step mean losses), then at 8k the two controls must agree within 0.10 pp S / 0.20 pp D / 0.30 pp per source; an A6000-trained bucket arm must pass admission against BOTH controls; if the controls disagree, decision pilots run on the A5000 only |
| O+macro_filtered | O + `macro_filtered` |
| O+madasr | O + `madasr` |
| O+vaani | O + `vaani` — RESTORED 2026-09-06: the owner's board verdicts had judged the Phase-1 hypothesis shown as a hint, not the reference (owner: "reference is okay ... it is phase 1 some are catastrophic"); verified by transcribing the board-embedded audio with hishab (12/12 flagged MaCro clips at 0–18 % WER vs reference, identical to the original files). Vaani references are verbatim-ish spontaneous speech (some omit false starts) — its arm decides |
| O+spring_inx | NOT RUN (SPRING-INX failed G0.9(a): 48.3 % Latin-script rows) |
| N_final | O + every bucket that passed its arm, on the A5000; its own 8k point is compared with O_A5000 and with the best single-bucket arm; an A6000 winner's 8k state is transferred to the A5000 (path verified beforehand) |
Stop an arm at any validation for non-finite loss, >1 % empties, or D worse than P1 by >5 pp.
**Bucket pass rule at 8,000** (all required): S ≤ S(O) − 0.3 pp with paired sentence-clustered 95 % CI upper bound < 0 (10,000 draws,
seed 20260905); D ≤ D(O) + 0.3 pp; no D source worse than O by >1.0 pp; conversational slice worse by ≤0.5 pp; empties ≤0.1 %.
**Simultaneous bound (council C #8):** with m predeclared single-bucket arms (m=3 after G0.6/G0.9: macro_filtered, madasr, vaani), admission additionally requires the paired-bootstrap upper quantile at 1−0.025/m (≈99.4th percentile for m=4) to be below zero. Bootstrap per panel corpus WER first, then form S; report marginal CIs too. An inconclusive bucket (CI includes 0) is excluded with NO claim of equivalence or bad labels. N_final's interval is a conditional selection statistic, not independent confirmation. A bucket that fails is **excluded from N_final** (no re-tuning, no partial admission). N_final is admitted to the long run iff it
passes the same rule against O **and** S(N_final) ≤ S(best single arm) + 0.2 pp (adding buckets must not hurt). If N_final fails but
a single arm passed, the long run uses that single arm's corpus (registered fallback) — **only if G0.8 (≥500 unique new hours after all exclusions) still holds for that corpus**; otherwise STOP unless the owner amends the minimum (council C #9). If nothing passes → **STOP Phase 4**; O is not
promoted; FLEURS stays sealed. The chosen arm resumes from its own 8k checkpoint (optimizer, sampler, RNG) to H with no second warmup.
Capped-mixture arm C (council A) is dropped in favour of this design; a 30 % cap may only be registered later as a new arm.
Pilot cost at the MEASURED 745 steps/h: 6 × 10.7 ≈ 64 GPU-h; continuation 8k→150k ≈ 191 GPU-h; with Gate-0 GPU work, eval and reserve ≈ 292 GPU-h total (council C #12; owner ruled GPU-hours soft, 'not six months' ≈ 12 days of A5000 time). Re-measure throughput on the final arm manifests before the ledger is trusted.

## 6. Headline (one event)
Freeze checkpoint, tokenizer, greedy CTC decoding and scorer; decode 920 FLEURS-test recordings once through the benchmark's
`asr_core` staging; paired sentence-cluster bootstrap (349 clusters, 20,000 draws, seed 20260905, corpus WER = Σedits/Σref words)
against frozen Phase-2 (16.54 %) and Phase-1 (**19.67 %** under the frozen scorer on the 920 test rows — the 19.57 % quoted earlier was a different harness; corrected 2026-09-05 after re-scoring `outputs_a5000/ehzawad_fastconformer/predictions.json`, which the benchmark summary also records as 0.19672) hypotheses. **Success: WER < 16.54 % and the 95 % CI of (P4 − P2) lies
entirely below 0.** "Strong": ≤16.04 %. Report CER, per-panel one-shots, latency (same protocol as Phase 3 SPEC Q4). No second decode.

## 7. Budget ledger (soft; owner may raise)
| item | GPU-h |
|---|---|
| Gate-0 GPU work (disagreement decode, restoration check, profiling) | ≤ 6 |
| Pilots: 2 controls + 4 buckets + N_final, 8k each incl. validation | ≈ 76 |
| Continuation 8k→150k incl. 71 validations | ≈ 194 |
| Final eval, export, latency | ≤ 8 |
| Reserve | 12 |
| **Total plan** | **≈ 298 (soft; owner: unbounded but not months)** |

## 8. Claims discipline (card)
"Continued fine-tuning of the Bengali FastConformer-CTC checkpoint on X unique hours of human-labelled Bengali speech; Y hours were
selected upstream by ASR/reference agreement (Whisper, WER<15, 3 rounds, digit rows dropped upstream); no pseudo-label targets or
TTS audio." Disclose: per-source table (creator, version, licence, hours, exclusions), the human audit result, MaCro speaker
separation unknown, FLEURS test history (scored in Phases 1–3), single Phase-4 scoring event, private pipeline, NC-SA vs CC-BY-SA
tension (owner decision). Never: "fully manually verified", "speaker-disjoint everywhere", "SOTA", "reproducible from the public repo".

## 8b. Gate-0 measurements (2026-09-06, after the full hash + text audits)
- Hashing complete: 1,882,804 rows; 2,019 old-corpus files (data/wavs) decode as silent → removed as audio defects; 10,278 exact
  duplicate groups within training (MaCro ∩ Common Voice); 380 train×panel hash matches (MaCro vs D 204 / holdout 176), all removed.
- Text audit over every bucket: FLEURS test 0 matches on every key. macro_filtered shares 749 sentences with D / 645 with holdout;
  madasr shares 126 sentences with R exactly and 4,443 fuzzy (all removed; 50,723 madasr rows carried a protected sentence).
- Panel M: 1,964 groups / 3.50 h, 0 recordings hash-identical to any Phase-1 training recording; **P1 baseline M = 20.59 % frozen
  (NeMo-internal 30.56 %), S(P1) = 20.998 pp with weights D .6 / M .2 / R .1 / K .1** (`experiments/p4/baseline_p1.json`).
- G0.9(a): macro_filtered 0 %, madasr 0 %, vaani 1.31 % BEFORE the "|" ruling (1,119 of its 1,135 flagged rows carried Vaani's
  "|" sentence delimiter, a punctuation mark Unicode classes as Sm; registered: "|" → space for every bucket) → vaani ≈ 0.02 %;
  spring_inx 46.2 % → FAIL (excluded). G0.9(b) incoming: macro_filtered 0.19 % (1,047 CV copies) pass; madasr 0 %; vaani 0 %;
  macro_validated 31.3 % (9,208 of its clips are Common Voice training recordings) — panel-only as ruled.
- RESPIN under the sentence×dialect cap: 95,243 rows / 149.0 h retained (dialects use disjoint prompt inventories, so the cap
  behaves like 5 per sentence; council D warned exactly this). Not raised (registered: "do not raise the cap again").
- Provisional arms: O 543,776 / 972.5 h; O+macro_filtered 1,088,679 / 1,701 h; O+madasr 639,019 / 1,121 h; N 1,850 h (+878 new h ≥ 500 ✓).
  Rebuild with the "|" ruling and audio-defect removal in progress; fingerprint match (train × panels incl. M) running.

## 8c. Storage finding (2026-09-06) — registered mitigation
`/mnt/sdb` is a single rotational 16 TB HDD (~135 random reads/s). Training reads 252 clips per update; Phase 1's 984 h fit
the 96 GB page cache (1,100 steps/h) but arm N (227 GB PCM16) does not: the Gate-0 drills ran at ~250 steps/h with the GPU at
20–40 % and the disk at 100 %. Mitigation: every training clip of arm N is stored losslessly as FLAC on the NVMe root disk
(`/tmp/p4cache`, ~125 GB of 279 GB free; `tools/p4_nvme_cache.py`, sample-exact round-trip verified per file, PCM-head
digest recorded). Arm manifests used for training are verbatim path-remapped copies (`*.nvme.jsonl`, checksums recorded);
the frozen `/mnt/sdb` manifests remain the registered artefacts. The cache is regenerable and deleted after Phase 4.
Consequence: the 745 steps/h figure was measured on a page-cached 2,000-row smoke and is NOT the HDD rate; throughput is
re-measured from the cache before the ledger is trusted. The fingerprint match runs on the HDD while pilots read the NVMe.

## 8d. Gate-0 GPU qualification results (2026-09-06, from the NVMe cache)
- Throughput from the cache (median per-update wall time, updates 20–120): A5000 891 steps/h, A6000 887 steps/h (peak 9.2 GiB).
  Worker-side tokenisation (pretokenize=False) gave 863 steps/h — no gain — so the registered value stays pretokenize=True.
  Ledger at ~880 steps/h: pilot arm 8k ≈ 9.1 GPU-h; continuation 142k ≈ 161 GPU-h.
- Cross-card 200-update check (`experiments/p4/cross_card_check.json`): cut order and LR sequence IDENTICAL at every update;
  all losses/gradients finite; 20-update mean-loss relative difference worst block 1.58 % (max |Δloss| 0.075) — above the
  proposed 1 % tripwire → **tripwire FAILS**. Over the tested 200-update probe the same-card replay was bit-identical (Δloss 0.0); the 1,000-update replay later showed
  same-card divergence from update 285 (max |Δloss| 0.022), so determinism is asserted only for short probes. **RULING (orchestrator):** the tripwire's 1 % was council
  D's proposed policy tolerance, not a measured bound; the decisive registered rule remains the 8k control agreement
  (O_A6000 vs O_A5000 within 0.10 pp S / 0.20 pp D / 0.30 pp per source). O_A6000 runs; if that agreement fails, every
  A6000-trained arm is discarded and decision pilots run on the A5000 only. Disclosed on the card either way.

- 1,000-update interrupted-vs-uninterrupted replay (`experiments/p4/replay_1000_check.json`, cut at 400, epoch boundary at
  540→541 inside the resumed leg): cut ids and LR identical at all 1,000 updates, resume handshake OK, both runs finite.
  Losses are bit-identical for updates 1–284 and then drift (max |Δloss| 0.022, worst 20-update block 1.23 %) in BOTH the
  uninterrupted and the interrupted run — the divergence begins 116 updates BEFORE the cut, so it is run-to-run CUDA
  nondeterminism (non-deterministic kernels in CTC loss / convolutions), not a resume defect. **RULING:** resume semantics
  PASS on the registered criteria (ids, LR, handshake, epoch boundary); numerical drift of this size is the same-card noise
  floor and is disclosed; it also reframes the cross-card 1.58 % block difference as mostly run-to-run noise rather than a
  card effect (the 8k control-agreement rule remains the binding cross-card test). Deterministic algorithms are NOT switched
  on (they would change the registered kernels and throughput mid-phase).

- Deviations disclosed at Gate 0 (council F Q7): no train×train fingerprint dedup (1.8 M × 1.8 M infeasible; exact PCM hashing covers cross-source duplicates) and no fingerprint index for the Vaani protected splits (text + exact PCM hash only). The original §1/§3 wording ("fingerprint dedup across all corpora", "EVERY candidate bucket") is therefore NOT fully executed as written.
- hishab exposure: its 4.3–4.8 % WER on MaCro/RESPIN raises a training-exposure concern; provenance is unverified. Card wording: "suspected training exposure; independence unverified".
- 2026-09-06 G0.6 outcome (final): the owner reviewed all 230 board clips and first exported 65 + 6 "catastrophic" verdicts, then
  clarified they had been judging the Phase-1 hypothesis hint, not the reference: "Vaani and MaCro-filtered and RESPIN reference is
  okay". Verification: hishab transcribes the board-embedded audio of 12/12 flagged MaCro clips to the reference (0–18 % WER; board
  audio identical to the originals). **G0.6 = PASS for macro_filtered, madasr and vaani (references OK per the owner).** The machine
  consensus audit is the quantitative record. Lesson recorded: a listening board must not show a model hypothesis next to the
  reference by default — it biases the listener toward judging the model.

- Machine consensus audit result (`experiments/p4/audit/machine_audit.json`, 2026-09-06): old-corpus baseline 2.13 % consensus
  disagreement / 0.20 % catastrophic-like (n=3,000; note Phase 1 trained on these clips, so this baseline is biased LOW → the
  rule is stricter for new data). macro_filtered 0.20 % / 0 % → PASS (limit 4.27 % / 0.7 %); madasr 0.60 % / 0 % → PASS; vaani
  1.80 % / 0 % would pass the machine rule but is excluded by the owner's listening verdict (6 catastrophic), which outranks it.
  Board clips (170 MaCro+RESPIN): 0 consensus disagreements. Caveat for the card: hishab's conformer-large scores 4.8 % WER on
  MaCro-filtered and 4.3 % on RESPIN (vs 13–24 % on our old sources), i.e. it has almost certainly trained on those corpora —
  it is an agreement witness, not an independent judge, for those two buckets; the Phase-1 model is the independent one.
  Side finding: old OpenSLR-53 rows show 7.4 % consensus disagreement (both models agree against the reference) — the highest
  of any source. O stays frozen as registered (no ASR-agreement-based row selection); disclosed as a known label-noise source
  and a candidate for a future phase.

## 8e. FREEZE (2026-09-06 16:0x UTC) — Gate 0 closed
- Fingerprint match over 1,836,132 training clips × 19,174 panel clips: 296 confirmed + 7,937 unresolved (≥12 hits) → 5,296
  training rows excluded; 135 panel-M recordings historically exposed (old-corpus audio matches) → removed; **M = 1,829 groups /
  3.24 h**. Retained-corpus verification (`experiments/p4/corpus/retained_leakage_verification.json`): 0 retained rows in any
  hash, fingerprint or text match; 0 retained PCM identical to any panel or FLEURS file; 0 M rows with old-corpus matches.
- Final frozen arms (`FREEZE.sha256`, 22 entries incl. code, config, tokenizer, panels): O 541,276 / 965.9 h; O+macro_filtered
  1,084,696 / 1,692.4 h; O+madasr 636,517 / 1,114.9 h; O+vaani 627,223 / 1,083.3 h; N 1,265,884 / 1,958.7 h (+992.8 new h ≥ 500).
  Training reads the NVMe copies (`*.nvme.jsonl`, sha256 O f5b8e716…, +macro 00d6506b…, +madasr f92be7d7…, +vaani cb1469c6…, N ba723cc9…).
- Launched: O_A5000 and O_A6000 controls (8k, val every 2k), A6000 guarded by `experiments/p4/a6000_guard.sh` (foreign process →
  graceful stop at a checkpoint → re-queue on the A5000; owner rule "continue on one card").

- Controls started 16:02 UTC. Determinism finding: a 5-update probe on the A6000 reproduces O_A6000's losses BIT-EXACTLY
  (same-card runs are deterministic from update 1), while O_A5000 and O_A6000 consume identical cut ids/tokens yet differ by
  ~10 % in per-update loss from update 1. PROPOSED mechanism (not traced in this environment): CUDA dropout/mask kernels may derive Philox offsets from the launch
  grid, which depends on SM count (A5000 64, A6000 84), giving different dropout/SpecAugment realisations. Treat as a hypothesis. Consequence: the two controls
  double as a run-to-run (seed-like) noise estimate; the 8k control-agreement rule is the binding cross-card test as registered.
  Council C/D's "≤1 % 20-update block" tripwire is recorded as not applicable across cards.

## 8f. Council E (pre-launch) closeout — items certified after its packet (2026-09-06)
- Retained-corpus leakage certification: `experiments/p4/corpus/retained_leakage_verification.json` PASS (0 retained rows in any
  hash/fingerprint/text match; 0 retained PCM identical to any panel or FLEURS file, all 1,322 FLEURS files hashed). The incoming
  `leakage_report.json` (PASS=false) is preserved: its counts are findings against INCOMING rows that the builder excluded. The
  665 `totals_fleurs` text matches are fleurs_train rows echoing the FLEURS **validation** split (contaminated, never used); the
  test split has 0 matches on every key.
- Fingerprint index coverage: 19,174 = D 5,385 + holdout 6,767 + K 2,236 + R 1,500 + M 1,964 + FLEURS 1,322 (test+validation).
  Vaani validation/test were NOT fingerprint-indexed; they are protected by text and by exact PCM hash (22,800 files, 0 identical
  to Vaani training audio, `vaani_protected_split_hash_check.json`) and are decoded only after final selection. SPRING is excluded
  entirely (its splits are not evaluation instruments in this phase). Train×train fingerprint dedup was NOT run (1.8 M × 1.8 M);
  cross-source duplicates are handled by exact PCM hashing (10,278 groups) and the known MaCro ⊃ Common Voice provenance — disclosed
  as a deviation from §1 dedup step 3.
- P1 baseline on the FINAL M (1,829 rows): M 20.74 % frozen, **S(P1) = 21.030** (D 20.35 / K 14.67 / R 32.06 unchanged).
- Tokenizer on the frozen N: 0 `<unk>` rows in every source; tokens/s p99 ≤ 8.73 (max 11.16 madasr) — within the ≤11 p99 rule;
  longest clips are the old-corpus 29.98 s rows already profiled at microbatch 12 (8.49 GiB); new rows ≤ 22.8 s.
- Candidate-count reconciliation: macro_filtered 550,921 ids → 550,617 ingested (304 density rejects) → 543,420 retained;
  macro_validated 29,583 → 29,462 ingested (109 duration mismatches + 12 density) → panel only.
- G0.6 (council E wording adopted): G0.6 uses the owner-amended machine-consensus rule including each bucket's random sample and
  assigned board clips, both strata reported separately (`machine_audit_verdicts_with_board.json`: macro_filtered 0.18 % / 0 %,
  madasr 0.57 % / 0 %, vaani 1.89 % / 0.09 % (board 1/60 catastrophic-like reported separately) — all pass). The earlier
  hypothesis-directed owner verdicts are withdrawn; the corrected owner assessment is an informal reference spot-check.
- Cross-card amendment (council E wording adopted): the registered 200-update loss-agreement threshold of 1 % was exceeded
  (1.582 %) and the failure is preserved; before pilot training the loss-block agreement is a diagnostic; identical cut ids, LRs
  and finite losses remain mandatory; A6000 decision eligibility requires the 8k controls to agree within 0.10 pp S / 0.20 pp D /
  0.30 pp per source; A6000 arms pass against both controls; if agreement fails, decision arms run on the A5000.
- N_final candidate unions frozen (`nfinal_candidates.json`): macro 726.5 new h; macro+madasr 875.4; macro+vaani 843.9; all three
  992.9; any union without MaCro ≤ 266 h < 500 → MaCro must pass for Phase 4 to continue. If the admitted union equals a tested
  single arm, that arm's 8k checkpoint is reused; a distinct union trains from init A with its own schedule prefix.
- Ledger corrected: six 8k runs ≈ 54.5 GPU-h at 880 steps/h; continuation ≈ 161 GPU-h; H=150k ≈ 29.9 nominal epochs of N (H and
  the stopping rules unchanged; the earlier "16–18 epochs" text is superseded).
- Wave plan (council E Q8) adopted: wave 1 O_A5000 + O_A6000 (running); wave 2 O+macro_filtered (A5000) + O+madasr (A6000);
  wave 3 O+vaani (A5000); wave 4 N_final (A5000, if distinct); admission tests exactly as council E's table (m=3, 99.1667th
  percentile), stop rules: non-finite / empties >1 % / D > 25.35 %.
- Reporting cadence amended by the owner: an update after EVERY validation (each 2,000 steps) instead of hourly.

## 8g. Robustness policy (owner 2026-09-06: "training must complete on its own; prefer the A5000, switch to the A6000 if busy;
survive terminal or process death; always resume")
- Every run is driven by a systemd user unit `p4-sup-<run>` (`experiments/p4/run_supervised.sh`) that survives the terminal and the
  session: it picks the preferred card if no OTHER user's compute process is on it and ≥19.5 GB is free, otherwise the other card,
  otherwise waits; launches the worker unit (`run_arm.sh`, `--resume auto` whenever a checkpoint exists, KillMode=mixed so a stop
  reaches only the trainer, Restart=no — the supervisor is the sole restarter); polls every 60 s; a foreign process on our card →
  graceful stop (the trainer writes a resumable checkpoint at the next update and exits 4) → immediate re-queue on a free card;
  a crash → re-queue after 11 min (past any unit auto-restart) with resume; completion = last logged update ≥ target.
- Resumable checkpoints every 500 updates (rotated, last 2) in addition to validation/milestone checkpoints, so any loss is ≤500
  updates (~35 min). Cross-card resume is allowed (init weights, optimizer, sampler cursor and RNG are card-independent; the two
  cards draw different dropout masks after a hand-over — accepted and disclosed).
- The two controls launched before this policy were ADOPTED by their supervisors at updates 740 / 254; they still checkpoint only
  at validations (every 2,000) because the periodic rule entered the trainer after their start.
- Wave-2/3 arms are queued under supervisors (macro_filtered → A5000, madasr → A6000, vaani → A5000 preferences) and take a card
  only when it is free of our own and any foreign process.

## 8h. Owner ruling 2026-09-06 22:3x UTC: A5000 ONLY, serial
The co-tenant's 33 GB job occupies the A6000. The owner ruled: rely on the A5000 and complete the work serially. Consequences:
the O_A6000 control is abandoned at update 4,038 (its partial validations at 2k/4k are kept as a cross-card observation:
S within 0.08 pp of O_A5000); no A6000 arm exists, so the cross-card admissibility rule is moot; all arms compare with O_A5000;
order on the A5000: O_A5000 → O+macro_filtered → O+madasr → O+vaani → N_final → continuation. Supervisors run in only-preferred
mode (never migrate). The A6000 amendment (§0) is withdrawn for the rest of Phase 4 unless the owner re-enables it.

## 8i. Share-a-card policy (owner 2026-09-07 "work smart", supersedes 8h's whole-card assumption)
Measured footprint over 7,774 updates: peak reserved 9.21 GiB median / 9.23 GiB max, peak allocated 8.72 GiB max, worst case a
microbatch of twelve 30 s clips. Therefore a run no longer waits for an EMPTY card: it takes whichever card has ≥12,000 MiB free
(9.23 GiB + ~2.7 GiB cushion), sharing with a co-tenant's idle notebook; it yields (graceful checkpoint, re-queue) only if free
memory on its card falls below 2,500 MiB, i.e. when the co-tenant actually needs the space. Watchdog lowered 17.0 → 13.0 GiB so
our process stops long before it can squeeze anyone. Card assignment is recorded per run; arms may therefore run on either card,
and the ruling of §8d applies: the 8k cross-card agreement test is UNCOMPLETED (the A6000 control was abandoned at 4,038; its 2k/4k S agreed within 0.08 pp,
which does not certify the missing 8k D and per-source thresholds — continuing is a disclosed protocol deviation), so **every arm is compared with O_A5000 and the card of each run is disclosed**; the two cards'
partial controls agreed within 0.08 pp S at 2k and 4k, which is the only cross-card evidence on record.

## 8j. Arm results
- **Control O (8,000 updates, A5000 then A6000):** S 21.073 (P1 21.030). D 20.32 / M 20.57 / K 14.93 / R 32.76, conversational
  21.47, guards pass. Reading: 8k more updates on the old corpus alone changed S by **+0.043 pp vs P1**. Its S was still
  DECLINING (21.54 → 21.32 → 21.17 → 21.07), so this does NOT establish saturation (council F Q7); it establishes the
  comparator every arm is measured against, with training/hardware variation not separated.
- **O+macro_filtered (8,000 updates, A6000): ADMITTED.** Authoritative record `experiments/p4/admission_macro_filtered.json`:
  S 20.531 vs O_ref 21.073, **ΔS = −0.542 pp, paired 95 % CI [−0.736, −0.352], simultaneous upper quantile −0.314 pp** at
  1−0.025/3 (9,727 sentence clusters, 10,000 draws, seed 20260905, clusters coupled across panels; the tool's per-panel WERs
  reproduce the trainer's to 3 decimals from val_preds + manifests, verifying hypothesis ordering). Guards pass: D −0.11,
  conversational +0.42 (limit +0.50), worst source indicvoices +0.79 (limit +1.00), empties 0.019 %. Panels: D 20.21 / M 18.03 /
  K 14.90 / R 33.07. Trajectory S: 21.22 → 21.18 → 20.70 → 20.53 (the conversational guard failed at 2k/4k/6k, cleared at 8k).
  **CORRECTION (council F Q7, 2026-09-07): the first version of this entry recorded Δ −0.517, CI [−0.62, −0.42], upper −0.44.
  Those numbers were transcribed wrongly by the orchestrator and never existed in any artifact; the JSON above is authoritative
  and the superseded line is preserved here.**
- **Honest characterisation of that result (council F Q6):** at equal 8,000-update budgets the MaCro arm improved the registered
  composite by 0.542 pp, driven predominantly by a 2.54 pp improvement on held-out MaCro (M contributes ≈93 % of ΔS: D −0.066,
  M −0.508, R +0.031, K −0.003 weighted pp), while D improved 0.11 pp and conversational WER worsened 0.42 pp within tolerance.
  It supports admission under the registered objective; it does NOT demonstrate broad improvement, label superiority, or a
  predicted FLEURS gain. Equal updates also do not hold old-data exposure constant: adding MaCro roughly halves O's share of
  presentations (MaCro is 50.1 % of presented rows in O+macro_filtered, 42.9 % in N), so the comparison measures the whole
  corpus-mixture intervention.

## 8k. Council F (mid-ablation, 2026-09-07) — rulings adopted
- **Q1 selector: KEEP S FROZEN** for admission AND final checkpoint selection (0.60 D / 0.20 M / 0.10 R / 0.10 K, ties to the
  earlier checkpoint by 0.05 pp). Switching to D after seeing that MaCro's benefit concentrates on M would be post-hoc selector
  shopping; S measures the registered panel mixture and does not claim FLEURS optimality.
- **Q2 conversational: NO abort rule added.** A three-failure rule would have killed the MaCro arm at 6k before its registered
  8k pass. Checkpoint INELIGIBILITY is the protection, provided (i) ineligible checkpoints never reset plateau patience and
  (ii) the selected eligible checkpoint is preserved independently of rotation. **Continuation/final guards use the frozen P1
  baseline, not O_ref**: every D source ≤ P1 +1.0 pp; conversational ≤ P1 +0.5 pp = **22.075 %** (exact P1 21.575, reconciled;
  the "21.58" elsewhere was rounding). Admission against O_ref does NOT certify continuation eligibility. MaCro's 8k
  conversational 21.892 is eligible under the P1 rule.
- **Q3 N_final:** point threshold is the tighter of S ≤ 20.773 (O_ref −0.30) and S ≤ S(best passing single) + 0.20 = **20.731**
  if MaCro stays best. Fallback if N_final fails = the lowest-S passing single arm **whose own frozen corpus has ≥500 unique new
  hours** — with the frozen inventories only O+macro_filtered (726.5 h) qualifies (madasr 149.0, vaani 117.4, both together 266).
  No further union search, no partial bucket admission. m stays 3 even if an arm is skipped by owner authorisation.
  Permitted claim: "Among the frozen corpus candidates evaluated at 8,000 updates, this corpus was selected by the registered
  weighted-dev, regression-guard and minimum-hours rules; the selected-union comparison is conditional on those development
  results and does not independently establish an optimal corpus or transfer to FLEURS."
- **Q4 horizon: KEEP H=150,000** and the plateau rule (anchor 50k, ≥0.20 pp cumulative reset, 10 non-reset validations,
  earliest stop 70k). Nominal epochs: O+MaCro 34.85, all three 29.86 — monitored, not a reason to change a registered schedule
  after seeing pilots. The verified 8k prefix (optimizer, sampler, RNG, scheduler) is preserved: no second warmup, no LR reset.
- **Q5:** finish O+madasr and O+vaani (~18–25 GPU-h) unless the owner declares a deadline; skipping records them as UNTESTED
  and excluded, with m unchanged.
- **Q8 the 20k progress gate is the decisive cheap check:** S ≤ 19.530 AND D ≤ 19.35 AND all main-run guards, else Phase 4
  TERMINATES. From MaCro's 8k that demands a further **1.00 pp S and 0.86 pp D** — improving M alone cannot satisfy it. Cost to
  reach the decision ≈ 15–19 GPU-h; the failure it prevents ≈ 157–210 GPU-h.
- **Terminal stop semantics (implemented 2026-09-07 in `run_supervised.sh`):** a registered STOP (progress gate, plateau, pilot
  abort, watchdog) or an ABORT terminates the run; the supervisor exits WITHOUT restarting and no FLEURS decode follows. Only
  `reason: graceful_stop` (card hand-over) and crashes are resumable. Previously the supervisor would have resurrected a run the
  protocol had ended — the defect council F named in Q8.
  **AMENDED 2026-09-07 by §8p (owner-delegated ruling): the phrase "and no FLEURS decode follows" is SUPERSEDED for the plateau
  stop.** The supervisor semantics above are unchanged — no STOP is ever restarted — but a plateau STOP is a *normal ending*
  that proceeds to §4 checkpoint selection and the §6 headline, exactly as reaching H does. The 20k gate failure, pilot aborts,
  the watchdog and every ABORT remain terminal with no headline. The struck words are kept here verbatim for the record.

## 8l. New bucket `cv17` — Common Voice 17.0 Bengali (owner-authorised 2026-09-07, registered BEFORE any of it is used)
**OWNER RULING 2026-09-07 (verbatim intent: "that unconfirmed tier is the 1,186 hours, the part I would call bad. Yes,
fucking remove that. No need any testing on those portion.") — the `other` split is PERMANENTLY EXCLUDED, not deferred.**
The registered WER/CER quality filter below is STRUCK: it will not be run, in this phase or a later one, and `other` is not a
candidate for any future registration. `invalidated` was already never ingestible. **SUPERSEDED SAME DAY by the owner's second ruling below**, which removed the whole release rather than one split; the
sentence is kept because it records what the first deletion covered: the 25 tar files of `other` (26.42 GB) and the
`invalidated` tar (296 MB), deleted from `data4/cv17/audio/bn/` on 2026-09-07.

**OWNER RULING 2026-09-07, second (verbatim intent: "the fully human validated data set we took from Mozilla ... Remove
all seventeen version items") — EVERY release-17 artefact is deleted, not just the unconfirmed split.** Removed:
`data4/cv17/` in full (3.578 GB, 7,780 files: the dev/test/train tars, 7,682 converted WAVs, 602 MB of transcript TSVs and
the ingest logs), the Hugging Face cache stubs `datasets--mozilla-foundation--common_voice_17_0` and
`datasets--fsicoli--common_voice_17_0` with their lock directories, and the 29 raw tar member listings under
`experiments/p4/cv17_recon/` in full (30 MB, listings and analysis). No arm or panel manifest referenced any of it; nothing in the training
pipeline depended on it. **Nothing of release 17 remains.** The 888 KB analysis record (`GO_NO_GO.md`, six result JSONs, the PCM hash table and
their scripts) was deleted in the same ruling, and every claim that rested on it is STRUCK in §8n(iii) rather than left
standing without evidence. The project makes no citable measurement about release 17.
**Why the whole release goes and not only the bad split:** §8n-bis. Release 17.0 is nine releases older than the 26.0 we
actually train on, so none of it was ever a candidate. Keeping a stale release beside the live one is only an invitation
to confuse the two.
**No Phase-4 training row has ever come from any Common Voice unconfirmed tier, and this is now verified at the source rather
than inferred.** `tools/convert_commonvoice.py` — the Phase-1 ingester — reads `validated.tsv` ONLY, from Common Voice **26.0**
(`cv-corpus-26.0-2026-06-12/bn`), and additionally excludes that release's own `dev.tsv` and `test.tsv` so the public benchmark
split stays uncontaminated. Its docstring is explicit: "Keeps ONLY rows in validated.tsv (community-validated, core S0/S1 tier)".
(A sentence that stood here about "425 `other` clips matched by an identity join" cited a deleted release-17 measurement and is
STRUCK; see §8n(iii).) What stands without it: every Common Voice row in arm_O came from release 26.0's validated tier, and
removing any of them would cost the entire freeze — arm_O is hashed in FREEZE.sha256 and is the base of every arm and of O_ref.
**Why, from measurement not intuition.** Dev-panel error decomposition at the MaCro arm's 8k checkpoint: IndicVoices 45.3 %
of all errors (17.30 % WER), SUBAK.KO 33.6 % (36.03 %), OpenSLR 7.8 %, CommonVoice 7.1 %, Kathbath 6.1 %. SUBAK.KO's failure
was tested against four hypotheses and none explains it: duration (uniformly bad in every band, 24–56 %), numerals (38.40 %
on digit-bearing vs 35.48 % digit-free), compound-writing convention (1.2 % of its errors), truncated references (0 detected);
median hypothesis/reference length ratio is 1.000, so the model emits the right number of words and the wrong words — genuine
lexical difficulty in formal/literary Bengali. The registered headline (FLEURS) is read encyclopedic prose, the same register.
Every bucket added so far is the WRONG register for it (MaCro everyday prompts, RESPIN agri/banking prompts, Vaani image
descriptions), which is consistent with MaCro moving its own panel 2.54 pp and the locked panel 0.11 pp.
**Source and licence.** `fsicoli/common_voice_17_0`, config bn, **CC0-1.0**; Bengali prompts are largely Wikipedia-derived.
Splits: `validated` 44,121 clips / 73.7 h (2+ listeners confirmed the recording matches the prompt), `other` 993,696 /
1,181.4 h (not yet confirmed), `invalidated` 7,811 (community REJECTED — never ingested), dev/test/train ⊂ validated.
We currently train on 41.5 h from an older CV release.
**SUPERSEDED 2026-09-07 — the two split figures on the line above are wrong and are kept verbatim for the record.**
Measured: `validated` 44,121 rows / 73.723 h listed but only 39,882 clips / 66.663 h downloadable (4,239 rows have audio
in no tar); `other` 997,561 clips / 1,185.797 h. Both splits are NO-GO / deferred; see §8n(iii) for the reason, which is
not the inventory: release 17.0 is nine releases older than the 26.0 we train on, so it was never new data (§8n-bis).
**Leakage, measured on transcripts BEFORE downloading audio (`data4/cv17/leakage_text.json`): FLEURS test 0 sentences,
FLEURS validation 0** — the headline is unaffected. D 495 / holdout 463 / M 1,825 sentences do appear in CV, because
Bengali.AI ran the Common Voice Bangla campaign and MaCro shares its prompt pool; those rows are removed by the builder's
protected-sentence rule exactly as MaCro's were, which is what keeps panel M valid. 33,819 of 39,875 validated and 350,435 of
474,987 `other` sentences are already in our training text; `other` still carries **124,552 unseen sentences in 244,284 clips**,
and MAX_PER_TEXT=5 is nowhere near binding (625,283 of our sentences have a single recording; 3.18 M free slots).
**Registered quality filter for `other` — STRUCK 2026-09-07 by the owner ruling above; kept verbatim for the record, never to be run.** `validated` is human-confirmed and takes the ordinary
cleaning only. For `other`: decode once with the Phase-1 FastConformer — NOT hishab, which is suspected to have trained on
Common Voice and would confirm these clips trivially — and **drop a clip iff WER > 0.60 AND CER > 0.40** against its
frozen-surface reference. The AND is deliberate: gross defects (silence, wrong prompt read, truncation, non-speech) fail both
surfaces, while a hard-but-correct clip with one rare word wrong fails at most one. This is a defect filter, not a
select-the-easy filter like the upstream Whisper WER<15 % rule; the retained/removed rates and the removed-clip examples are
published. Everything else is the registered pipeline: G0.9 uniformity, PCM-hash dedup across all sources, fingerprint audit
against every protected panel, MAX_PER_TEXT, and its own 8,000-update ablation arm.
**Multiplicity consequence:** predeclaring a fourth single-bucket arm makes **m = 4**; MaCro's admission is re-tested at m=4
(quantile 1−0.025/4) and must still clear, otherwise the earlier m=3 verdict is reported as superseded.

## 8m. Council G (expansion review, 2026-09-07) — CV17 DEFERRED out of Phase 4; rulings adopted verbatim
- **CV17 status:** "CV17 was proposed after inspection of Phase-4 development results. Its Phase-4 ingestion/filtering/ablation
  plan is deferred. No CV17 recordings or targets enter Phase-4 training, and no CV17 ASR filtering or pilot GPU work is charged
  to this phase. CPU admission reconnaissance may proceed without delaying the registered runs or changing their frozen
  artifacts. A failed Phase-4 progress gate does not activate a CV17 rescue run." (The 27 GB download and the CC0 licence
  record are kept; the `other` ingest was NOT started; the partial `train` ingest is reconnaissance only.)
- **Multiplicity:** m stays **4** because CV17's slot was announced in §8l, even though its arm is deferred; deferring or
  skipping an arm never reduces m. Every admission requires the bootstrap upper quantile at 1−0.025/4 = **0.99375** below zero.
  **MaCro re-tested at m=4: ΔS −0.542, quantile −0.306 < 0, all point/guard criteria hold → still ADMITTED**
  (`experiments/p4/admission_macro_filtered_m4.json`). The m=3 record is historical.
- **Adaptive-discovery disclosure (registered):** "The CV17 proposal was generated from error analysis of the same development
  panels used for admission and checkpoint selection. It was not prospectively chosen independently of those results. No CV17
  data were used in the selected Phase-4 model. Any subsequent CV17 evaluation on these panels constitutes adaptive development
  evidence; increasing the number-of-comparisons correction alone does not make that evaluation independent confirmation."
- **Claims corrections (my overstatements, retracted):** (i) the SUBAK.KO analysis weakens several explanations but does not
  establish a unique cause — equal median word counts do not exclude compensating insertions/deletions; (ii) transfer from
  Common Voice prompts to SUBAK.KO or FLEURS is a hypothesis, not a finding; (iii) **Vaani is spontaneous image-description
  speech, so "every candidate bucket is read speech" was false**; (iv) the proposed WER>0.60 AND CER>0.40 rule is an
  ASR-disagreement filter over human prompts that may reject correct difficult recordings and retain defective pairs — calling
  it "a defect filter, not a select-the-easy filter" was too strong; (v) zero CV17/FLEURS transcript matches does not certify
  absence of audio leakage; (vi) the mirror exposes train/dev/test/other/invalidated and NOT a `validated` split, so the
  44,121-clip figure from validated.tsv is not reconciled against available audio.
- **Marginal-value arithmetic that reframes the gate (council G Q5):** panel D's reference words are 52.9 % IndicVoices and
  18.9 % SUBAK.KO, so the outstanding 0.87 pp of D needs ≈ **1.64 pp on IndicVoices alone** or **4.62 pp on SUBAK.KO alone**
  (≈480 fewer edits). Conversational coverage is therefore central, and sampling weights stay frozen — upweighting IndicVoices
  is a separate intervention needing its own arm.
- **Execution order adopted:** finish MADASR and Vaani → build exactly ONE union U = O ∪ (all passing buckets) under the frozen
  dedup/precedence/cap rules → reuse MaCro's 8k state if U equals it, else one 8k run of U → admission vs O_ref plus
  S(U) ≤ S(best passing single) + 0.20 (= 20.731 if MaCro stays best) → resume to the unchanged 20k gate at ≈78–96 cumulative
  GPU-h → on failure TERMINATE with no FLEURS decode. No subset search, no second union, no mixture rescue.
- **Budget ceiling 300 GPU-h** including the ~45 spent; CV17 receives **0 further Phase-4 GPU-h**; only the owner raises it.
- **Yield-floor reconciliation:** the registered 2,500 MiB free-memory floor was replaced 2026-09-07 by a co-tenant-DEMAND rule
  (yield only if a new foreign process appears or foreign memory grows > 2,000 MiB), because the free-memory floor stopped a
  healthy arm six times in eleven minutes — free memory is low precisely because our own 9.2 GiB is allocated.

## 8n-bis. Release-choice error on the `cv17` registration (owner question 2026-09-07: "why aren't you using the latest common voice 26?")

**The owner is right and this is my error, recorded rather than quietly fixed.** We already train on Common Voice **26.0**,
the NEWEST release that exists. `tools/convert_commonvoice.py` reads `data2/raw/commonvoice/cv-corpus-26.0-2026-06-12/bn`,
`validated.tsv` only, minus that release's own `dev.tsv`/`test.tsv`, keeping 25,935 rows / 41.51 h over 2,383 speakers
(24,712 rows / 39.5 h survive Phase-4 dedup and caps). §8l then registered `fsicoli/common_voice_17_0` as a *new bucket*.
**Release 17.0 is NINE releases older than 26.0**, so it was never new data: it is a stale snapshot of a corpus we had
already ingested at a newer release.

Verified against Mozilla's own version registry (`common-voice/cv-dataset`, `datasets/scripted-speech`, 54 releases
listed): 21.0 2025-03-14 · 22.0 2025-06-20 · 23.0 2025-09-05 · 24.0 2025-12-05 · 25.0 2026-03-09 · **26.0 2026-06-12 is
the newest; there is no 27.0.** The Hugging Face ecosystem is what misled the registration: `mozilla-foundation` publishes
up to 17.0 and `fsicoli` up to 22.0, so the newest release reachable on HF is four behind what we already had on disk.

**Consequence for §8n(iii).** That section's recycling measurements are STRUCK, their artefacts having been deleted on
the owner's ruling. The reasoning that replaces them needs no measurement: Common Voice is a cumulative corpus, so an
older release is contained in the newer one we already trained on, by construction. The decoder-blindness of PCM-sha256
survives as a separately registered limitation on the *instrument*; it is why a hash could not have seen such an overlap,
never why one existed.

**Consequence for any future Common Voice work: there is nothing material to gain, and this is now measured from an
external source that survives the deletion.** Mozilla's own release registry (`common-voice/cv-dataset`,
`datasets/scripted-speech/cv-corpus-<v>.json`) gives Bengali per release:

| release | date | validated clips | validated h | `other` | `invalidated` | all clips |
|---|---|---:|---:|---:|---:|---:|
| 17.0 | 2024-03-15 | 44,121 | 53.51 | 997,561 | 7,811 | 1,049,493 |
| 20.0 | 2024-12-06 | 44,351 | 53.81 | 998,046 | 7,860 | 1,050,257 |
| 23.0 | 2025-09-05 | 44,634 | 54.18 | 999,488 | 7,926 | 1,052,048 |
| **26.0** | **2026-06-12** | **44,739** | **54.32** | 999,500 | 7,939 | 1,052,178 |

Every bucket's COUNT grows across these releases and Bengali gained a **net 618 validated clips (+0.81 h) in 27 months
and nine releases**. **Council H caveat, adopted:** Mozilla documents release-time exclusion of problematic clips and
partitions that can change between releases, so growing aggregate counts do NOT prove a clip-level subset relation; "618
new recordings" is a net figure, not an identifier-level one. The cumulative-release description is Mozilla's own; it is
not used here as a proof of strict containment.

**Phase 1 already holds every usable validated clip of the newest release.** 44,739 validated − 9,402 dev − 9,402 test =
**25,935**, which is exactly the row count `data2/commonvoice/manifest.jsonl` contains. Not one clip was lost to the
duration or empty-text filters. Re-downloading 26.0 would yield **zero** new rows. No forecast is made about a future release (council H). No further
Common Voice ingestion is planned. **Residual, not closed:** the identity assertion `ingested IDs = validated IDs \\ (dev ∪ test)`
has not been verified at the identifier level — the Phase-1 manifest keeps only `client_id[:16]`, not the clip path, and the
26.0 raw tree is gone — so the 25,935 = 44,739 − 9,402 − 9,402 identity is a row-count match, not an ID match.

## 8n. Main-run readiness workstreams (2026-09-07, four parallel agents; every figure below re-verified against the raw files by the orchestrator before it was written here)

**Nothing in this section changes the freeze, the arms, the panels, the selection score, the guards, the 20k gate or the
budget.** It fixes code that implements them, corrects two factual errors in this document, and records one measured
instrument limitation.

### (i) Gate and selection code — eleven defect IDs (D1–D11), all fixed, all covered by executable tests
`experiments/p4/tests/test_main_run_gate.py` (`results.json` records the sha256 of `p4_train.py`, `p4_common.py` and
`run_supervised.sh` it was run against). The one test that was SKIPPED at first run — `retention_defect_reproduced_on_the_superseded_code`,
which needs the pre-fix trainer — now runs against `experiments/p4/tests/superseded/p4_train_pre_D4_2026-09-07.py` (sha256 f0ae17de…)
and passes: the D4 defect is demonstrated on every run of the suite, not asserted. Current suite: see §8o.
- **D1** `p4_common.py` still held the SUPERSEDED plateau rule (`PLATEAU_FROM=30000`, 5 validations). Council F Q4 (§8k)
  re-ruled it to anchor 50k / 10 non-reset / earliest stop 70k. Fixed, with an assertion that 50,000 + 10×2,000 = 70,000.
- **D2** the plateau reset was measured against a MOVING best, so it was not cumulative: a run improving 0.15 pp every
  validation forever would never have reset. Fixed with a sticky anchor.
- **D3** an INELIGIBLE checkpoint (guards failing) reset plateau patience and became the best — council F Q2 condition (i)
  says it must not. Fixed: only guard-passing validations touch the anchor or the best; an ineligible one is a non-reset.
- **D4** retention had no notion of eligibility, so `best/` was best-3-by-S over ALL checkpoints. Three ineligible lower-S
  checkpoints evict the best eligible one, and with `keep_last=2` the unlink of its hard link **deletes the only copy of the
  model §4 says must be selected**. Reproduced on the pre-fix code. Fixed, plus a post-condition that ABORTs if the current
  selection is ever not on disk. Cost: 1.39 GB/checkpoint, 9.5 TB free. **Bound corrected by council H:** every eligible checkpoint inside the band is
  retained, and all 71 continuation validations could in principle fall within 0.05 pp, so the worst case is ≈99 GB
  (75 validations from 2,000 → ≈104 GB); affordable, and the correct bound is stated rather than "at most one".
- **D5** the sort implemented no tie band: S 20.47 @80,000 beat S 20.50 @60,000, where §4 gives ties within 0.05 pp to the
  EARLIER checkpoint. Fixed (band swept from the best, never chained).
- **D6** the 20k gate could be silently disabled by `--val-every`. Fixed: a validation at 20,000 is forced whenever the run
  is going that far.
- **D7** the horizon/effective-batch resume refusal was a bare `assert` — stripped by `python -O`, and an AssertionError is a
  crash, which the supervisor treats as RESUMABLE: an infinite relaunch loop against a bad checkpoint. Fixed as a recorded
  ABORT (exit 3).
- **D8/D9/D10** the watchdog message and three docstrings still said 17.0 GiB after §8i lowered the ceiling to 13.0; the
  external backstop `watch_p4.sh` defaulted to 17.0, i.e. 4 GiB **above** the trainer's own limit, so it could not fire for
  any breach the trainer would not already have caught; and its A5000 UUID was hard-coded, so on the A6000 — where O_ref's
  tail and the whole MaCro arm ran — our usage always read 0. That is not "no breach", it is no measurement. All fixed.
- **D11** (found by the gate workstream, fixed by the orchestrator, who owns the file) `run_supervised.sh` reached
  `rm -f runs/$RUN/STOP` on the launch path before any worker had exited, so a **fresh invocation** deleted the record of a
  terminal outcome and resumed a run the protocol had ended — the resurrection council F #5 legislated against, one shell
  command away. The in-loop classifier could never see it, because it only runs after a worker exits. Fixed: a startup guard
  refuses to start (exit 5) on any STOP whose reason is not `graceful_stop`, on an unparsable STOP, and on any ABORT that is
  not one of the registered infrastructure strings; the launch path now deletes a STOP only when its reason is
  `graceful_stop`. `P4_OVERRIDE_TERMINAL_STOP=1` exists and requires an owner decision recorded here. Eight new end-to-end
  cases drive the real script with an impossible memory requirement so no trainer can launch.
- **Not a defect:** the continuation comparator was already the frozen P1 baseline (`baseline_p1.json`, init sha256
  `61362a36…`, S 21.0296, conversational 21.575 → limit 22.075). It was hardened, not changed: the baseline is refused if it
  lacks init A's provenance, and the 20k gate stops rather than proceed with a comparator it cannot vouch for.
- **Operative gate constant.** The S threshold is evaluated **relatively** as S(P1) − 1.5 = **19.5296**, cross-checked
  against the 19.530 printed by council F to 0.001 pp. The relative form governs; 19.530 is its rounded presentation.

### (ii) Admission-test audit — two defects fixed, the ordering proof added
`p4_admission_test.py`: `--m` defaulted to **3**, contradicting §8m, which fixes m at 4 permanently — anyone re-running it
without the flag would have produced a non-registered verdict at the 0.99167 quantile that looked authoritative. Now
`M_DEFAULT = 4`, the 0.99375 quantile asserted against §8m, achieved quantile written into the JSON. And `val.jsonl` was read
as `readlines()[-1]`, so on a run that validated past the step under test it would have scored step-8000 hypotheses against a
later checkpoint's S and guards without complaint; it now requires exactly one record with `update == step`.
**Four things were already right and were left untouched:** the simultaneous bound was already the empirical percentile of
the realised draws (no normal approximation anywhere), clusters were already frozen-surface reference strings sharing one id
table across all four panels, corpus WER was already summed-edits / summed-reference-words inside each draw, and the guards
already compared against O_ref. The **ordering proof** (recomputed per-panel WER must match the trainer's own `val.jsonl` to
0.005 pp, plus D's empty count, list length, and arm/control agreement on reference strings and word counts) did not exist
and is the substantive addition: worst live disagreement 0.00048 pp; negative tests refuse on a two-hypothesis swap
(0.0793 pp) and on a dropped hypothesis. Cross-panel cluster coupling proved with a fixture: coupled Δ sd 0.09361 pp vs
uncoupled 0.07011 pp. `p4_union_build.py` reproduces all five frozen manifests **byte-identically** against `FREEZE.sha256`
and refuses when U equals a tested single arm.

### (iii) Common Voice release 17 — WITHDRAWN 2026-09-07, artefacts deleted, claims struck

**This section formerly recorded a CPU reconnaissance of release 17.0 and its measurements. The owner ordered every
release-17 artefact deleted, including the 888 KB analysis record that backed those figures. Claims whose evidence no
longer exists are STRUCK rather than left standing.** Struck: the corrected split inventory; the PCM-sha256 sample and
its zero-duplicate result; the identifier-join recycling rates; the 120-pair waveform cross-correlation; the per-split
incoming-defect ratios; the usable-hours upper bounds; the panel-overlap counts against release 17. None of those may be
cited anywhere in this project again.

**Two findings survive because their evidence never depended on the deleted artefacts, and both are re-checkable today.**

1. **The registration was a release-choice error, not a data finding.** See §8n-bis. Release 17.0 is nine releases older
   than the 26.0 the project already trains on, so it was never new data. Evidence: `tools/convert_commonvoice.py` in this
   repository, and Mozilla's public version registry.
2. **Instrument limitation, wording fixed by council H (kept as a rule; it follows from what a hash is, not from any
   destroyed measurement):** "PCM SHA-256 detects equality of the canonical decoded sample representation. It does not
   establish recording independence: the same underlying recording can produce unequal PCM representations after decoding,
   re-encoding, resampling or other transformations. This is an instrument limitation, not an empirical finding about
   Common Voice 17. Future overlap audits must combine source-identity checks and calibrated audio fingerprinting with exact
   PCM hashing. No cross-decoder miss rate is claimed." The MaCro ∩ CommonVoice 1,047 exact copies (§8b) remain positive
   evidence backed by the surviving audit; no claim is made about WHY they were catchable.

**What this does NOT put in doubt.** No release-17 row ever entered an arm or a panel — verified while deleting, by
grepping every frozen manifest. The panels are protected by the protected-sentence rule, which operates on text and is
unaffected. `arm_O+vaani` contains only Vaani `wavs_train` rows (85,947 of 86,861). **Correction (council H):** Vaani validation and test
are not selection panels but they ARE protected one-shot evaluation sets (§2), and their absence from the fingerprint index
remains a DISCLOSED coverage limitation on the one-shot reports, not a non-issue.

### (iv) Conversational survey (research only; nothing admitted, nothing may be admitted without its own registration)
`experiments/p4/surveys/conversational_survey_2026-09-07.md`. Headline: **there is no large, clean, unencumbered pool of
Bengali conversational speech that we overlooked.** Ranked by admissible-after-our-cleaning hours × label quality × licence:
1. **SPRING-INX R1+R2, Bengali-script-only subset, ≈280 h** (estimate from the character share 40.3 %/47.3 %, not the row
   share 53 %, because code-switched rows are longer; a row-share estimate would have overstated it by ≈65 h). Genuinely
   dialogue, human transcripts with vendor QC, CC BY 4.0 with the licence record already archived in-repo, ungated, 16 kHz.
   We excluded it on a **script** rule (G0.9(a), 48.3 % Latin rows), not a quality one. Admitting it needs an owner amendment
   with the threshold fixed BEFORE the retained rate is known, and a registered disclosure that this is a **content** filter
   removing exactly the code-switched conversation, not a defect filter.
2. **IndicVoices upstream delta, 0–116 h, UNRESOLVED** — the cheapest open question in the survey (a CPU-only diff of the
   Phase-1 raw ingest count against today's 335,367 rows). Highest value per hour if real: D is 52.9 % IndicVoices reference
   words. No claim of "+116 h available" is made.
3. **Ben-10, ≈35 h** after the marker rule; CC0-1.0, linguist-validated, but likely fails G0.6 as written.
   Everything else is read speech in the wrong register, forced-alignment or subtitle-derived, synthetic TTS, or has an
   unverifiable licence, and is a NO. Arithmetic anchor: closing D's outstanding 0.87 pp needs ≈1.64 pp on IndicVoices alone
   or ≈4.62 pp on SUBAK.KO alone.

### (v) Correction to §2
`data2/holdout_corpus.json` is described there as "speaker-disjoint IndicVoices-R". Counted this session it is **6,767 rows /
10.72 h** across commonvoice 464, indicvoices 2,493, kathbath 626, openslr53 2,413 and subakko 771 — it contains **no
IndicVoices-R rows at all**. Read the parenthetical as "disjoint from IndicVoices-R". No result changes; the phrase was
load-bearing for the IndicVoices-R question in the survey and is corrected rather than left to be re-derived.

## 8o. Council H (progress review before the continuation, 2026-09-07 ~20:40 UTC) — rulings adopted

Packet: `experiments/p4/council/council_h_prompt.md` (contract + measured facts + pre-launch card + survey head + 11 questions);
verdict: `council_h_progress.md`. **Headline: "Keep the registered science."** No hyperparameter changes; finish both pilots;
one distinct union if required; existing admission, selection, horizon and stopping rules retained. Council found factual
errors in the packet (corrected in place, listed below) and ONE uncovered failure boundary (Q11) that turned out to be a real
defect when checked against the code. Everything below is now implemented and tested unless marked OWNER.

- **Q11 — REAL DEFECT, fixed under 18 new tests.** The loop writes `step_20000.pt` BEFORE validating and deciding the gate; a
  crash in between leaves a checkpoint that owes its decision, the resume starts at 20,001, and `if update == 20000` never
  fires again — the gate would be silently skipped for the rest of the run. The PASS was also only a `print`. Fix, three
  parts, all in `p4_train.py`: (1) a PASS is now durable — `runs/<run>/GATE20K` JSON plus `sel["gate20k"]="pass"` carried in
  every later checkpoint; (2) **resume replays any owed validation** — a resume FROM a validation-step checkpoint reuses that
  step's `val.jsonl` record if it exists (it is the evaluation of exactly that saved state) or decodes the resumed state if
  not, then replays the gate/plateau decision and the retention; a checkpoint written later (U+500) owes nothing because
  the process that wrote it had already completed them; (3) **invariant** — a resume at any update > 20,000 in a run heading
  past 20,000 with no recorded PASS is refused as a terminal ABORT, which the supervisor does not retry. Tests cover: reuse
  without decode, decode when no record, replayed FAIL is terminal and a fresh supervisor stays stopped, nothing owed at a
  non-validation step, a lost plateau tick is restored, the invariant refuses/accepts/is inert in each case, duplicate
  records are refused, and council's named boundary **S = 19.5300 FAILS, 19.5297 FAILS, 19.5296 passes** against
  S(P1) − 1.5 = 19.5296.
- **Q1 — every fix implements the contract; none needed an owner ruling.** D5 interpretation adopted verbatim: find the
  minimum UNROUNDED eligible S, then select the earliest checkpoint within 0.05 pp of it. Plateau anchoring uses the minimum
  eligible S independently of the tie-selected checkpoint. Tie-band retention bound corrected (§8n(i)). "Ten" → eleven defect
  IDs. The skipped test now runs against the preserved pre-fix trainer. `P4_OVERRIDE_TERMINAL_STOP=1` may exist; USING it
  requires an owner ruling recorded here.
- **Q2 — union sequence confirmed; comparator ≠ fallback.** After both pilots: full admission test vs O_ref at 8,000, m=4,
  quantile 0.99375. Form ONE union U = O ∪ passing buckets. If U is not a tested single arm it trains its own 8,000 prefix
  from init A (≈9.5–11.1 GPU-h at 846–723 upd/h; never splice trained states). **U's point threshold is
  S(U) ≤ min{ S(O_ref) − 0.30, min over PASSING singles of S + 0.20 }** — so if madasr's 8k S ends below MaCro's 20.531,
  madasr sets the +0.20 bar even though it can never be the fallback. **If U fails, the fallback is MaCro's own 8k state**,
  the only passing single whose corpus carries ≥500 new hours (MaCro 726.5 ✓, madasr 149.0 ✗, vaani 117.4 ✗). No second
  union, no subset search. Note: madasr at 4k is NOT yet a pass on the O_ref guards it must clear at 8k.
- **Q3 — in-domain-dominated admission is legitimate under the registered objective; the packet's "D moved −0.09" was
  WRONG.** At matched 4,000 updates, madasr − control: **D +0.076, M +0.444, R −9.547, K +0.608 → S −0.7595; R contributes
  125.7 % of the net**, i.e. every other panel is slightly worse. Card wording adopted verbatim: *"At 4,000 updates,
  O+MADASR improved the registered composite by 0.7595 percentage points relative to the control at the same update. The
  improvement was concentrated on RESPIN dev: R improved by 9.547 points, while D worsened by 0.076, M by 0.444 and K by
  0.608. These are interim development results; the registered admission decision occurs at 8,000 updates."* And on any
  admission: *"Admission establishes improvement under the registered weighted-development objective and regression
  tolerances. It does not establish broad improvement, superior labels, or transfer to FLEURS. Bootstrap intervals
  condition on the evaluated trained systems and do not quantify training-run or hardware variability; the registered
  8,000-update cross-card agreement test was not completed."* Both guard systems are required and neither substitutes:
  admission guards vs O_ref (each D source ≤ +1.0, conversational ≤ +0.5), continuation guards vs P1 (conv ≤ 22.075).
- **Q4 — vaani runs to 8,000** (≈6.0 GPU-h remaining, not 5). No registered early stop applies. If it fails it is excluded
  whole; a capped mixture or conversational up-weighting is a NEW registration, never a rescue.
- **Q5 — relative thresholds govern:** S ≤ 21.0296 − 1.5 = 19.5296, D ≤ 19.35; the 0.001 pp check is a consistency check
  against council F's printed constants, never gate slack. S = 19.5300 fails. (Implementation note: a float-representation
  tolerance of 1e-9 pp is applied to both comparisons so that a value equal to the limit at the registered 4-dp precision
  is not failed by binary representation; 1e-9 is five orders below the panel's WER quantum and cannot pass 19.5297.)
- **Q6 — PCM-instrument rule kept with council's narrower wording** (§8n(iii) item 2 replaced verbatim). No CV17 experiment is
  recreated. Vaani validation/test correctly described as protected one-shots with a disclosed fingerprint-coverage gap.
- **Q7 — G0.9(b) unchanged; MaCro ∩ CommonVoice 1,047 copies unchanged; "counts grow ⇒ strict containment" rejected** and
  softened in §8n-bis. Future re-releases of already-held corpora: screen publisher, lineage, release, canonical recording
  IDs and historical exclusions BEFORE audio ingestion; register the novel delta prospectively against the full inventory.
- **Q8 — change NO hyperparameter.** Epoch arithmetic corrected: MaCro 34.85, MaCro+madasr **32.04** (not 30.6), all three
  29.86. The exposure is an accepted risk under the registered horizon + plateau + eligible-checkpoint selection. Resume
  carries weights AND buffers, optimizer, scheduler position, sampler and RNG; a continuation LR reset would break the
  literal-prefix design. Card corrected: max 1,958.7 h is **41.3 h short of 2,000**; what MaCro satisfies is G0.8's ≥500 new h.
- **Q9 — budget fits:** ≈248 GPU-h (reuse MaCro) / ≈258 (distinct union) with 8 h eval allowance; at 723 upd/h throughout
  the union path is ≈280. Council's CPU preflight (<30 min, 0 GPU-h) adopted as the pre-continuation checklist: verify code
  hashes against the passing test record; fresh-launch the continuation so it loads the corrected code; re-issue admission
  records with the hardened tool (m=4, quantile 0.99375, exactly the 8k record); verify corpus/checkpoint association and
  full resume state; run the Q11 fault-injection test.
- **Q10 — NEXT registration only, nothing admitted now.** (a) IndicVoices: pre-register a CPU-only identity + exclusion-ledger
  diff whose OUTCOME is the measured unique admissible delta in hours; zero closes it; no invented cutoff. (b) SPRING-INX
  amendment text prepared, **REQUIRES OWNER AUTHORISATION**, with the disclosure that it is informed by observed script rates:
  *"For the next registration only, SPRING-INX R1/R2 may form a Bengali-script subset. After the frozen registered surface
  transformation, retain a row only if it contains zero disallowed characters under the existing script predicate. Reject
  the entire offending row; do not delete spoken words or transliterate. The whole-source script-loss ceiling is waived for
  this explicitly defined subset. Report exclusions against the full source inventory. This is a content filter that removes
  code-switched speech. All other admission requirements remain applicable."* ≈280 h remains an estimate. (c) Ben-10 stays in
  the research queue.
- **OWNER DECISION REQUIRED (not blocking the continuation; cannot fire before update 70,000): plateau STOP and the
  headline.** §8k's terminal-stop sentence lists a plateau STOP among the outcomes after which "no FLEURS decode follows".
  Read literally, the run's intended good ending — convergence — would deny the headline, and only reaching H = 150,000
  would produce one. Council H: if the intended behaviour is to score the selected checkpoint after a plateau stop, that is an
  **owner amendment**, not an implementation fix. Orchestrator's recommendation: amend §8k so that a plateau STOP and reaching
  H are both *normal endings* that proceed to §4 checkpoint selection and the §6 headline, while the 20k gate failure, pilot
  aborts, the watchdog and numerical ABORTs remain terminal without a headline. Pending the owner's word, the literal text stands.

## 8p. Ruling: the plateau stop is a normal ending, not a failure (owner-delegated, 2026-09-07 ~21:10 UTC)

**Owner, verbatim:** "Honestly, I cannot decide here. What you think, do ultra think, and you decide is up to you." Council H
had ruled that changing §8k's literal text is an owner amendment, not an implementation fix. The owner delegated the amendment to
the orchestrator. This section is that amendment and its reasoning; it is recorded as a RULING, not as a correction.

**The ruling.** A run has two classes of ending.
- **Normal endings → §4 checkpoint selection → §6 headline, decoded once:** (a) reaching H = 150,000; (b) a **plateau STOP**
  at update ≥ 70,000 under the council F Q4 rule (anchor at 50,000; reset on ≥ 0.20 pp cumulative eligible improvement;
  stop after 10 consecutive non-reset validations).
- **Terminal endings → NO headline; a human looks; any further step needs an owner ruling recorded here:** (a) the 20,000
  progress-gate failure — Phase 4 terminates by rule and is NOT overridable; (b) the §5 pilot aborts (non-finite loss, >1 %
  empties, D > P1 + 5 pp); (c) the watchdog STOP (an infrastructure event on a shared card: fix the cause, then an owner
  ruling may resume the run under `P4_OVERRIDE_TERMINAL_STOP=1`, or may decode the best eligible pre-event checkpoint);
  (d) any ABORT (numerical divergence, a resume refusal, a violated premise: same treatment as (c)).
The STOP record now carries `ending: "normal" | "terminal"` (`p4_train.stop`), so the post-run procedure reads the class
instead of parsing the reason string. The supervisor's behaviour is unchanged: it restarts NEITHER class, which is right for
both, because training does not resume after either.

**Why this is the correct reading and not a loosening.**
1. The plateau rule is a *convergence* criterion — the ordinary early-stopping rule — registered by council D Q5 to save
   GPU-hours once the model has stopped improving. It is not a failure signal. It fires only when ten consecutive validations
   show no 0.20 pp cumulative gain on the development panels, i.e. precisely when further training would not have changed
   the selected checkpoint. The checkpoint it stops at is the same checkpoint the horizon path would have selected, reached
   at lower cost. Denying it a headline would make the plateau rule a self-destruct and would put the orchestrator under
   pressure to *not* let it fire — the kind of pressure a pre-registered rule exists to remove.
2. It adds no degree of freedom to the headline. The headline's validity rests on FLEURS being sealed and decoded once
   (§6), not on when training stopped. Checkpoint selection already chooses the minimum-S eligible checkpoint among the
   ~71 validations on the development panels; the plateau rule, defined prospectively on those same panels, chooses nothing
   the selection rule was not already choosing. FLEURS is never consulted by either. Council G's adaptive-analysis concern
   (§8m Q1) is about *what enters the corpus*, and is untouched.
3. The struck phrase was written on 2026-09-07 to implement council F Q8 — "the supervisor must not resurrect a run the
   protocol ended" — and it correctly lists the plateau stop among the outcomes the supervisor must not restart. Its
   authors bundled "no FLEURS decode" onto that list without distinguishing the run that ended because it *converged* from
   the run that ended because it *failed*. Read literally it would mean the only way to earn a headline is to train to
   150,000 regardless of convergence, which contradicts §4 ("Final checkpoint = lowest eligible S") and the purpose of the
   rule that §4 itself registers two lines earlier.
4. The terminal class is kept strict where strictness protects the science: a 20k-gate failure means the model is not the
   1.5 pp improvement the phase exists to produce, and the one FLEURS decode must not be spent on it; a divergence or a
   watchdog breach means the run's own premises broke and a human must look before anything is scored.

**What this does NOT change.** H stays 150,000; the plateau constants stay (50,000 / 0.20 pp / 10 / 70,000); the selection
rule stays; the headline protocol stays one decode with the registered success criterion (WER < 16.54 % and the 95 % CI of
P4 − P2 below 0); the supervisor stays non-restarting for every STOP. Earliest possible plateau stop remains update 70,000,
around 11 September; nothing here can fire before then.

**Also recorded here, on the owner's question of lineage.** The weights Phase 4 starts from are the **output of Phase 1**
(`experiments/fastconformer_ctc_bn_1kh/final.nemo`, sha256 61362a36), not the NVIDIA checkpoint directly. Phase 1 itself was
restored from **`nvidia/stt_en_fastconformer_ctc_large`** (the English FastConformer-CTC large; NeMo log 2026-08-30 02:39:54)
and trained 40,000 updates on the ≈984 h Bengali corpus with a fresh 1,024-token Bengali tokenizer. So the lineage is NVIDIA
English → Phase 1 Bengali → Phase 4, with only the weights carried at each step and a fresh optimizer state at the Phase-4
warm start (§4).

## 9. Log
- 2026-09-07 ~21:10: §8p — owner delegated the plateau-stop question ("you decide"); RULED: plateau STOP is a NORMAL ending → selection → headline; 20k gate failure, pilot aborts, watchdog and ABORTs stay terminal; STOP records now carry `ending`. Lineage recorded: nvidia/stt_en_fastconformer_ctc_large → Phase 1 → Phase 4.
- 2026-09-07 ~21:00: council H (progress review) adopted (§8o). Its Q11 hypothesis — a crash between the 20k checkpoint and the gate decision lets the resume skip the gate forever — was REAL in the code and is fixed with a durable PASS record, resume-time replay of any owed validation, and a past-20k invariant, under 18 new tests (79 pass, 0 fail, 0 skipped). Packet errors corrected in place. Plateau-STOP-vs-headline put to the owner.
- 2026-09-07 (owner caught): registering `cv17` was a RELEASE-CHOICE ERROR — we already train on Common Voice 26.0, the newest release in existence, and 17.0 is nine releases older. Recorded in §8n-bis with Mozilla's version registry as evidence. Second ruling the same day removed every release-17 artefact including the analysis record, so §8n(iii)'s measurements are struck.
- 2026-09-07 (owner): Common Voice's unconfirmed `other` tier PERMANENTLY EXCLUDED, its quality filter struck, 26.7 GB of `other`+`invalidated` audio deleted; verified at the source that Phase 1 ingested `validated.tsv` only from CV 26.0 minus that release's dev/test, so no unconfirmed clip has ever been trained on.
- 2026-09-07: four-agent main-run readiness workstream (§8n): 10 gate/selection defects fixed under 61 executable tests (incl. a retention bug that could DELETE the selected checkpoint, and the supervisor resurrecting a terminal STOP); admission test's m default corrected to 4 and an ordering proof added; CV17 reconnaissance filed, later WITHDRAWN the same day with its artefacts deleted and its measurements struck (§8n(iii)); conversational survey filed (SPRING-INX ≈280 h the only large candidate, needs an owner amendment to G0.9(a)).
- 2026-09-06: council D pre-pilot reconciliation (`experiments/p4/council/council_d_prepilot.md`): STOP pilots until Gate 0 closes; two-card controls, RESPIN sentence×dialect cap, Vaani arm runs, SPRING excluded (48.3 % Latin rows), plateau anchor rule, clause fixes adopted; launch checklist = council D Q8. Ledger ≈ 298 GPU-h (soft).
- 2026-09-05 (later): council C interim audit (`experiments/p4/council/council_c_interim_audit.md`) → STOP pilots until Gate 0 closes;
  12 required changes adopted above; G0.9(b) reading for MaCro∩CommonVoice duplicates put to the owner.
- 2026-09-05: owner decisions recorded; councils A/B run; MaCro competition bundle (24.7 GB, 963,636 mp3, 32 kHz mono 48 kbps)
  and IISc RESPIN `train_bn_clean` (87.1 GB) downloads started; tugstugi CSV inventory and 0-overlap-with-FLEURS audit done
  (`data4/tugstugi/csv`). tugstugi's `validated` list == competition `valid` split (29,583 of 29,588); his filtered list ⊂ `train`.


# FACTS
{
 "as_of_utc": "2026-09-07T21:12Z",
 "owner_request_verbatim": "\"how could reusing an existing model would help? or it actually does that we don't know like starting with Nvidia vs Phase 1 out can you somehow verify which would work better and dataset completely ready and verified do an ablation test on it ans reconciled the codex council.\" Earlier: \"you shouldn't bound by GPU hours like it's unbounded... but it shouldn't be like six months\".",
 "phase1_recipe_from_config_launch_yaml": {
  "pretrained_model": "nvidia/stt_en_fastconformer_ctc_large (English; CTC head re-initialised for the 1,024-piece Bengali tokenizer)",
  "tokenizer": "data/tokenizer_v1024_corpus (same as Phase 4)",
  "batch": "12 x 21 = 252 (same as Phase 4)",
  "sched": "CosineAnnealing",
  "lr_peak": 0.0005,
  "min_lr": 1e-05,
  "warmup_steps": 2000,
  "max_steps": 40000,
  "weight_decay": 0.001,
  "corpus_hours": 984.13,
  "final": "dev frozen 20.348 % (S_pp 21.0296); NeMo-internal 20.979 % at zero-update restoration"
 },
 "phase1_cold_start_curve_nemo_internal_dev_wer_pct": {
  "1000": 56.6,
  "2000": 40.3,
  "3000": 39.8,
  "4000": 32.8,
  "5000": 29.5,
  "6000": 27.6,
  "7000": 26.2,
  "8000": 25.9,
  "9000": 25.5,
  "10000": 24.5,
  "11000": 23.7,
  "12000": 22.9,
  "13000": 22.6,
  "14000": 22.3,
  "15000": 21.9,
  "16000": 21.5,
  "17000": 21.3,
  "18000": 21.2,
  "19000": 20.8,
  "note": "from lightning_logs.txt, val every 1,000 steps, NeMo-internal surface; NOT comparable to frozen-surface numbers (offset +0.63 pp at zero updates, +1.46 pp at 8k on the control) - do not chain them"
 },
 "phase4_warm_start_recipe": {
  "init": "Phase-1 final.nemo (61362a36), fresh AdamW",
  "lr_peak": 0.0002,
  "warmup": 1000,
  "H": 150000,
  "batch": 252
 },
 "warm_start_trajectories_frozen_surface": {
  "O_ref_control_(P1_weights_on_P1_corpus_alone)": [
   {
    "update": 2000,
    "S_pp": 21.5397,
    "D_frozen": 20.835,
    "D_nemo_internal": 22.29,
    "M": 21.435,
    "R": 32.243,
    "K": 15.274
   },
   {
    "update": 4000,
    "S_pp": 21.3181,
    "D_frozen": 20.578,
    "D_nemo_internal": 22.045,
    "M": 21.051,
    "R": 32.666,
    "K": 14.945
   },
   {
    "update": 6000,
    "S_pp": 21.1746,
    "D_frozen": 20.496,
    "D_nemo_internal": 21.998,
    "M": 20.721,
    "R": 32.479,
    "K": 14.849
   },
   {
    "update": 8000,
    "S_pp": 21.0735,
    "D_frozen": 20.319,
    "D_nemo_internal": 21.784,
    "M": 20.565,
    "R": 32.763,
    "K": 14.928
   }
  ],
  "O+macro_filtered": [
   {
    "update": 2000,
    "S_pp": 21.221,
    "D_frozen": 20.744,
    "D_nemo_internal": 22.225,
    "M": 19.659,
    "R": 33.075,
    "K": 15.353
   },
   {
    "update": 4000,
    "S_pp": 21.175,
    "D_frozen": 20.8,
    "D_nemo_internal": 22.272,
    "M": 19.329,
    "R": 32.964,
    "K": 15.328
   },
   {
    "update": 6000,
    "S_pp": 20.7045,
    "D_frozen": 20.409,
    "D_nemo_internal": 21.904,
    "M": 18.495,
    "R": 32.694,
    "K": 14.907
   },
   {
    "update": 8000,
    "S_pp": 20.531,
    "D_frozen": 20.213,
    "D_nemo_internal": 21.74,
    "M": 18.027,
    "R": 33.075,
    "K": 14.903
   }
  ],
  "O+madasr": [
   {
    "update": 2000,
    "S_pp": 21.0309,
    "D_frozen": 20.804,
    "D_nemo_internal": 22.254,
    "M": 21.381,
    "R": 27.345,
    "K": 15.378
   },
   {
    "update": 4000,
    "S_pp": 20.5586,
    "D_frozen": 20.654,
    "D_nemo_internal": 22.127,
    "M": 21.495,
    "R": 23.119,
    "K": 15.553
   }
  ],
  "O+vaani": [
   {
    "update": 2000,
    "S_pp": 21.7049,
    "D_frozen": 21.069,
    "D_nemo_internal": 22.488,
    "M": 21.339,
    "R": 32.687,
    "K": 15.27
   }
  ]
 },
 "P1_baseline_frozen": {
  "S_pp": 21.0296,
  "D": 20.348
 },
 "final_corpus_candidates": {
  "O+macro_filtered": {
   "rows": 1084696,
   "hours": 1692.4
  },
  "O+macro+madasr (most likely)": {
   "rows": 1179937,
   "hours": 1841.3
  },
  "all three": {
   "rows": 1265884,
   "hours": 1958.7
  }
 },
 "decision_timeline": {
  "madasr_8k": "08 Sep ~01:40 UTC",
  "vaani_8k": "08 Sep ~02:40 UTC",
  "corpus_decided": "08 Sep ~03:00 UTC",
  "warm_start_continuation_resumes": "08 Sep ~03:00 (or ~13:00 if a union pilot is needed)",
  "second_card_free_from": "when the last pilot ends, subject to the co-tenant"
 },
 "cards": {
  "A5000": "24.5 GB, co-tenant holds 13.5 GB, we fit 10.1 GB",
  "A6000": "49 GB, co-tenant 33 GB, we fit 10.2 GB",
  "cross_card_status": "8k control agreement test NOT completed (O_A6000 reached 4,038; council E/H caveat stands); 200-update cross-card check: ids/LR identical, 20-step loss blocks differ <=1.58 % vs <=1.23 % same-card drift"
 },
 "budget_gpu_h": {
  "soft_ceiling": 300,
  "registered_plan_total": "≈250-260 (≈280 at 723 upd/h)",
  "init_N_full_to_150k": "≈175-190",
  "init_N_staged_to_40k": "≈50",
  "throughput_updates_per_h": "723-855 measured"
 },
 "one_shot_panels_constraint": "FLEURS test (920 rows, sealed) is decoded ONCE; holdout, Vaani validation/test, SPRING validation/test are one-shots after final selection. If the warm-start headline is decoded on schedule, no clean sealed panel remains for a later init-N comparison.",
 "same_data_order": "shard_seed 20260905 and the same manifest give both inits the SAME cut order (a paired design at the data level)."
}

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
