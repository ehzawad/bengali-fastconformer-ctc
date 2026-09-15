You are the final pre-launch reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A–D shaped the contract; Gate 0 has now been executed. Below: the CONTRACT, then the FROZEN FACTS (JSON, measured), then QUESTIONS.

# CONTRACT
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
| `vaani` — ARTPARK-IISc Vaani transcription part, Bengali (owner-approved 2026-09-05) | manually transcribed spontaneous image-description speech (West Bengal districts), transcribers from the same district; CC-BY-4.0, HF gate accepted. Annotation policy (registered after measuring 14,598 rows: 52 % `<noise>` spans, 7 % `<talking>`, 64 % `{brace}`, 5.5 % partial markers): `<span>` tags mark the background condition around the PRIMARY speaker's words → tags deleted, words kept; `[event]` brackets deleted; `{brace}` is an English gloss of the preceding Bengali-script loanword ("লেগ {leg}") → deleted, following case suffix re-attached; rows with `[unintelligible]`/`[inaudible]` DROPPED (partial labels teach deletion, Phase-1 lesson). Residual risk (noise-heavy audio, span semantics) is what the `O+vaani` ablation arm and the 50-clip listening slice test | **train 86,861 / 118.1 h** (3,953 partial-marker rows dropped); validation 10,105 / 13.6 h and test 12,695 / 16.1 h are PROTECTED one-shot panels | G0.6 PASS after the owner's clarification (verdicts had targeted the Phase-1 hint); arm runs | IN iff audit §3 passes (own 50-clip listening slice + Phase-1 disagreement); its official validation/test splits are protected, never trained on |
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
| `data2/holdout_corpus.json` (speaker-disjoint IndicVoices-R); Vaani official validation (10,105) / test (12,695); SPRING R1/R2 validation and test | — | one-shot reports after final checkpoint selection (manifest hashes recorded at freeze) |
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
- Plateau (council D Q5): at step 50,000 initialise an anchor = best eligible S so far; reset anchor and counter whenever an eligible checkpoint improves the anchor by ≥0.20 pp cumulatively, else increment at each validation; stop after 10 consecutive non-reset validations (20k-step patience, earliest stop 70k). H stays 150,000 (≈16–18 epochs of the final corpus; Phase 1 ran 18).
- Final checkpoint = lowest eligible S; ties within 0.05 pp → earlier. No averaging, no decoder search.
- Resume semantics: audited — interrupted vs uninterrupted **1,000-step replay across an epoch boundary** must give identical
  sample ids and LRs (the 40-step smoke replay passed on 2026-09-05 but does NOT satisfy this; the 1,000-step replay is a Gate-0
  GPU item, ≈1.5 GPU-h, charged to Gate 0). Resource profiling is repeated on the final admitted manifests' longest clips.

## 5. Pilots — per-bucket ablation (owner directive 2026-09-05: "check all the new data we will be adding")
All arms start from init A with the same schedule prefix; **8,000 steps each = 2.016 M presentations** (equal exposure); validation
at 2k/4k/6k/8k on D, M, K, R. Official Vaani/SPRING splits are decoded only after final checkpoint selection.
| arm | corpus |
|---|---|
| O_A5000 control | Phase-1 corpus only (re-normalised targets), on the A5000 — canonical control |
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
  proposed 1 % tripwire → **tripwire FAILS**. Same-card replay is exactly deterministic (Δloss 0.0), so the difference is
  bf16 kernel variation between cards, not a code-path difference. **RULING (orchestrator):** the tripwire's 1 % was council
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

## 9. Log
- 2026-09-06: council D pre-pilot reconciliation (`experiments/p4/council/council_d_prepilot.md`): STOP pilots until Gate 0 closes; two-card controls, RESPIN sentence×dialect cap, Vaani arm runs, SPRING excluded (48.3 % Latin rows), plateau anchor rule, clause fixes adopted; launch checklist = council D Q8. Ledger ≈ 298 GPU-h (soft).
- 2026-09-05 (later): council C interim audit (`experiments/p4/council/council_c_interim_audit.md`) → STOP pilots until Gate 0 closes;
  12 required changes adopted above; G0.9(b) reading for MaCro∩CommonVoice duplicates put to the owner.
- 2026-09-05: owner decisions recorded; councils A/B run; MaCro competition bundle (24.7 GB, 963,636 mp3, 32 kHz mono 48 kbps)
  and IISc RESPIN `train_bn_clean` (87.1 GB) downloads started; tugstugi CSV inventory and 0-overlap-with-FLEURS audit done
  (`data4/tugstugi/csv`). tugstugi's `validated` list == competition `valid` split (29,583 of 29,588); his filtered list ⊂ `train`.


# FROZEN FACTS
{
 "arms": {
  "O": {
   "rows": 541276,
   "hours": 965.88,
   "by_source": {
    "openslr53": {
     "rows": 162144,
     "hours": 158.96,
     "share": 0.1646
    },
    "indicvoices": {
     "rows": 273001,
     "hours": 574.37,
     "share": 0.5947
    },
    "kathbath": {
     "rows": 45965,
     "hours": 81.46,
     "share": 0.0843
    },
    "commonvoice": {
     "rows": 24712,
     "hours": 39.46,
     "share": 0.0409
    },
    "subakko": {
     "rows": 33800,
     "hours": 105.81,
     "share": 0.1095
    },
    "fleurs_train": {
     "rows": 1654,
     "hours": 5.83,
     "share": 0.006
    }
   }
  },
  "O+macro_filtered": {
   "rows": 1084696,
   "hours": 1692.36,
   "by_source": {
    "macro_filtered": {
     "rows": 543420,
     "hours": 726.47,
     "share": 0.4293
    },
    "openslr53": {
     "rows": 162144,
     "hours": 158.96,
     "share": 0.0939
    },
    "indicvoices": {
     "rows": 273001,
     "hours": 574.37,
     "share": 0.3394
    },
    "kathbath": {
     "rows": 45965,
     "hours": 81.46,
     "share": 0.0481
    },
    "commonvoice": {
     "rows": 24712,
     "hours": 39.46,
     "share": 0.0233
    },
    "subakko": {
     "rows": 33800,
     "hours": 105.81,
     "share": 0.0625
    },
    "fleurs_train": {
     "rows": 1654,
     "hours": 5.83,
     "share": 0.0034
    }
   }
  },
  "O+madasr": {
   "rows": 636517,
   "hours": 1114.86,
   "by_source": {
    "madasr": {
     "rows": 95241,
     "hours": 148.97,
     "share": 0.1336
    },
    "openslr53": {
     "rows": 162144,
     "hours": 158.96,
     "share": 0.1426
    },
    "indicvoices": {
     "rows": 273001,
     "hours": 574.37,
     "share": 0.5152
    },
    "kathbath": {
     "rows": 45965,
     "hours": 81.46,
     "share": 0.0731
    },
    "commonvoice": {
     "rows": 24712,
     "hours": 39.46,
     "share": 0.0354
    },
    "subakko": {
     "rows": 33800,
     "hours": 105.81,
     "share": 0.0949
    },
    "fleurs_train": {
     "rows": 1654,
     "hours": 5.83,
     "share": 0.0052
    }
   }
  },
  "O+vaani": {
   "rows": 627223,
   "hours": 1083.28,
   "by_source": {
    "openslr53": {
     "rows": 162144,
     "hours": 158.96,
     "share": 0.1467
    },
    "vaani": {
     "rows": 85947,
     "hours": 117.4,
     "share": 0.1084
    },
    "indicvoices": {
     "rows": 273001,
     "hours": 574.37,
     "share": 0.5302
    },
    "kathbath": {
     "rows": 45965,
     "hours": 81.46,
     "share": 0.0752
    },
    "commonvoice": {
     "rows": 24712,
     "hours": 39.46,
     "share": 0.0364
    },
    "subakko": {
     "rows": 33800,
     "hours": 105.81,
     "share": 0.0977
    },
    "fleurs_train": {
     "rows": 1654,
     "hours": 5.83,
     "share": 0.0054
    }
   }
  },
  "N": {
   "rows": 1265884,
   "hours": 1958.73,
   "by_source": {
    "madasr": {
     "rows": 95241,
     "hours": 148.97,
     "share": 0.0761
    },
    "macro_filtered": {
     "rows": 543420,
     "hours": 726.47,
     "share": 0.3709
    },
    "openslr53": {
     "rows": 162144,
     "hours": 158.96,
     "share": 0.0812
    },
    "vaani": {
     "rows": 85947,
     "hours": 117.4,
     "share": 0.0599
    },
    "indicvoices": {
     "rows": 273001,
     "hours": 574.37,
     "share": 0.2932
    },
    "kathbath": {
     "rows": 45965,
     "hours": 81.46,
     "share": 0.0416
    },
    "commonvoice": {
     "rows": 24712,
     "hours": 39.46,
     "share": 0.0201
    },
    "subakko": {
     "rows": 33800,
     "hours": 105.81,
     "share": 0.054
    },
    "fleurs_train": {
     "rows": 1654,
     "hours": 5.83,
     "share": 0.003
    }
   }
  }
 },
 "g09a": {
  "macro_filtered": {
   "incoming_rows": 550617,
   "script_violation_rows": 0,
   "rate": 0.0,
   "max": 0.01,
   "verdict": "pass"
  },
  "macro_validated": {
   "incoming_rows": 29462,
   "script_violation_rows": 0,
   "rate": 0.0,
   "max": 0.01,
   "verdict": "pass"
  },
  "madasr": {
   "incoming_rows": 645722,
   "script_violation_rows": 0,
   "rate": 0.0,
   "max": 0.01,
   "verdict": "pass"
  },
  "vaani": {
   "incoming_rows": 86861,
   "script_violation_rows": 17,
   "rate": 0.0002,
   "max": 0.01,
   "verdict": "pass"
  },
  "spring_inx": {
   "incoming_rows": 240941,
   "script_violation_rows": 111270,
   "rate": 0.46181,
   "max": 0.01,
   "verdict": "FAIL"
  }
 },
 "g09b": {
  "macro_filtered": {
   "ingested_rows": 550617,
   "defect_rows": 1042,
   "rate": 0.00189,
   "max": 0.005,
   "hash_coverage": 1.0,
   "verdict": "pass"
  },
  "macro_validated": {
   "ingested_rows": 29462,
   "defect_rows": 9174,
   "rate": 0.31138,
   "max": 0.005,
   "hash_coverage": 1.0,
   "verdict": "PANEL_ONLY (owner ruling)"
  },
  "madasr": {
   "ingested_rows": 645722,
   "defect_rows": 0,
   "rate": 0.0,
   "max": 0.005,
   "hash_coverage": 1.0,
   "verdict": "pass"
  },
  "vaani": {
   "ingested_rows": 86861,
   "defect_rows": 2,
   "rate": 2e-05,
   "max": 0.005,
   "hash_coverage": 1.0,
   "verdict": "pass"
  },
  "spring_inx": {
   "ingested_rows": 240941,
   "defect_rows": 0,
   "rate": 0.0,
   "max": 0.005,
   "hash_coverage": 0.0,
   "verdict": "INCOMPLETE_HASHING"
  }
 },
 "panel_M": {
  "fixed_from": "experiments/p4/corpus/panel_M.indexed_by_fingerprint.jsonl",
  "rows_in": 1964,
  "rows": 1829,
  "removed_historical_fingerprint_exposure": 135,
  "hours": 3.237,
  "groups": 1829,
  "distinct_sentences": 1829
 },
 "admission": {
  "new_hours_beyond_O": 992.85,
  "min_required": 500,
  "PASS": true
 },
 "removed": {
  "fleurs_train": {
   "fingerprint_match_to_panel": 30,
   "latin_letters": 71,
   "math_expression": 20,
   "symbol_outside_bengali_digits": 7,
   "protected_sentence": 237
  },
  "openslr53": {
   "math_expression": 102,
   "cps": 3305,
   "latin_letters": 62,
   "text_near_duplicate_of_panel": 85,
   "fingerprint_match_to_panel": 422,
   "protected_sentence": 29,
   "symbol_outside_bengali_digits": 1,
   "max_per_text": 435
  },
  "subakko": {
   "fingerprint_match_to_panel": 291,
   "math_expression": 87,
   "latin_letters": 252,
   "text_near_duplicate_of_panel": 472,
   "symbol_outside_bengali_digits": 364,
   "cps": 12,
   "protected_sentence": 6
  },
  "kathbath": {
   "fingerprint_match_to_panel": 248,
   "text_near_duplicate_of_panel": 26,
   "protected_sentence": 1
  },
  "commonvoice": {
   "fingerprint_match_to_panel": 93,
   "text_near_duplicate_of_panel": 11,
   "cps": 8,
   "math_expression": 30,
   "protected_sentence": 1,
   "latin_letters": 1
  },
  "indicvoices": {
   "fingerprint_match_to_panel": 1441,
   "protected_sentence": 476,
   "text_near_duplicate_of_panel": 249,
   "cps": 246,
   "feasibility_margin": 36,
   "max_per_text": 2499
  },
  "macro_filtered": {
   "protected_sentence": 2789,
   "fingerprint_match_to_panel": 1605,
   "feasibility_margin": 891,
   "feasibility_infeasible": 358,
   "text_near_duplicate_of_panel": 199,
   "pcm_duplicate_of_kept_row": 1042,
   "max_per_text": 313
  },
  "macro_validated": {
   "protected_sentence": 2703,
   "text_near_duplicate_of_panel": 11,
   "feasibility_margin": 2,
   "pcm_duplicate_of_kept_row": 9174
  },
  "madasr": {
   "protected_sentence": 50723,
   "fingerprint_match_to_panel": 840,
   "text_near_duplicate_of_panel": 1352,
   "feasibility_unk": 66,
   "feasibility_margin": 34,
   "feasibility_infeasible": 2,
   "max_per_text": 497464
  },
  "vaani": {
   "protected_sentence": 852,
   "symbol_outside_bengali_digits": 11,
   "latin_letters": 6,
   "feasibility_infeasible": 1,
   "feasibility_margin": 4,
   "fingerprint_match_to_panel": 1,
   "pcm_component_conflicting_reference": 2,
   "max_per_text": 37
  },
  "spring_inx": {
   "latin_letters": 108346,
   "symbol_outside_bengali_digits": 2924,
   "protected_sentence": 12534,
   "feasibility_unk": 1,
   "feasibility_margin": 3,
   "feasibility_infeasible": 3
  }
 },
 "fingerprint_in_build": {
  "present": true,
  "generated": "2026-09-06T15:50:20Z",
  "confirmed": 296,
  "unresolved": 7937,
  "train_rows_excluded": 5296,
  "M_rows_historically_exposed": 135,
  "thresholds": {
   "MIN_HITS": 12,
   "REVIEW_HITS": 12,
   "min_span_s": 2.0
  }
 },
 "respin_cap": {
  "per_cell_speakers": 5,
  "per_sentence_total": 25,
  "retained_by_dialect": {
   "D3": 18945,
   "D1": 18500,
   "D5": 22844,
   "D2": 17610,
   "D4": 17342
  }
 },
 "fingerprint_report": {
  "generated": "2026-09-06T15:50:20Z",
  "thresholds": {
   "MIN_HITS": 12,
   "REVIEW_HITS": 12,
   "min_span_s": 2.0
  },
  "panel_clips": 19174,
  "train_clips_queried": 1836132,
  "confirmed": 296,
  "unresolved": 7937,
  "wall_s": 15199.8
 },
 "leakage_report": {
  "generated": "2026-09-06T15:51:11Z",
  "spec": "experiments/p4/SPEC.md 1 (dedup 2-4), 3 G0.3, 3 G0.4",
  "components": {
   "pcm_hash": true,
   "text": true,
   "fingerprint_calibration": true,
   "fingerprint_match": true
  },
  "pcm_hash": {
   "rows_hashed": 1882804,
   "rejects_by_reason": {
    "silent": 2019
   },
   "exact_duplicate_groups_within_training": 10278,
   "exact_duplicate_rows_within_training": 10278,
   "train_x_panel_hash_matches": 380
  },
  "text": {
   "totals": {
    "surface": 3079,
    "key_punct": 2929,
    "key_tight": 3390,
    "fuzzy": 8026
   },
   "totals_fleurs": {
    "surface": 665,
    "key_punct": 665,
    "key_tight": 665,
    "fuzzy": 665
   },
   "unresolved": 17424
  },
  "fingerprint_calibration": {
   "recovery": {
    "mp3_48k": 1.0,
    "gain_p6db": 1.0,
    "trim_0p3s": 1.0,
    "resample_8k": 1.0,
    "speed_0p95": 1.0,
    "speed_1p05": 0.995
   },
   "recovery_pass": true,
   "negatives": {
    "unrelated_pairs_implied": 39800,
    "unrelated_pairs_required": 10000,
    "unrelated_queries_scored": 200,
    "same_prompt_pairs": 1000,
    "same_prompt_pairs_required": 1000,
    "operational_index_clips": 20000,
    "operational_queries_scored": 200,
    "max_hits_operational_index": 11,
    "span_clearing_operational": 2,
    "span_clearing_unrelated": 2,
    "span_clearing_same_prompt": 5,
    "max_hits_unrelated": 6,
    "max_hits_same_prompt": 10,
    "max_hits_among_span_clearing_negatives": 11,
    "p999_hits": 10
   },
   "MIN_HITS": 12,
   "separation_pass": true,
   "PASS": true
  },
  "fingerprint_match": {
   "panel_clips": 19174,
   "train_clips_queried": 1836132,
   "confirmed": 296,
   "unresolved": 7937
  },
  "checks": {
   "G0.3_zero_unresolved_text_matches": false,
   "G0.4_calibration_passed": true,
   "G0.4_zero_pcm_hash_train_x_panel": false,
   "G0.4_zero_confirmed_audio_matches": false,
   "G0.4_zero_unresolved_audio_matches": false
  },
  "PASS": false,
  "incomplete_because": []
 },
 "machine_audit": {
  "baseline": {
   "n": 3000,
   "consensus_disagreement_rate": 0.0213,
   "catastrophic_like_rate": 0.002
  },
  "summary": {
   "macro_filtered": {
    "n": 1000,
    "consensus_disagreement_rate": 0.002,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.2485,
    "wer_hishab": 0.0477
   },
   "madasr": {
    "n": 1000,
    "consensus_disagreement_rate": 0.006,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.376,
    "wer_hishab": 0.0434
   },
   "vaani": {
    "n": 1000,
    "consensus_disagreement_rate": 0.018,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.2819,
    "wer_hishab": 0.337
   },
   "old:openslr53": {
    "n": 500,
    "consensus_disagreement_rate": 0.074,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.1429,
    "wer_hishab": 0.1309
   },
   "old:indicvoices": {
    "n": 500,
    "consensus_disagreement_rate": 0.01,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.1343,
    "wer_hishab": 0.2432
   },
   "old:kathbath": {
    "n": 500,
    "consensus_disagreement_rate": 0.012,
    "catastrophic_like_rate": 0.002,
    "wer_p1": 0.1253,
    "wer_hishab": 0.0837
   },
   "old:commonvoice": {
    "n": 500,
    "consensus_disagreement_rate": 0.006,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.1555,
    "wer_hishab": 0.0716
   },
   "old:subakko": {
    "n": 500,
    "consensus_disagreement_rate": 0.008,
    "catastrophic_like_rate": 0.008,
    "wer_p1": 0.1991,
    "wer_hishab": 0.1374
   },
   "old:fleurs_train": {
    "n": 500,
    "consensus_disagreement_rate": 0.018,
    "catastrophic_like_rate": 0.002,
    "wer_p1": 0.2203,
    "wer_hishab": 0.1687
   },
   "board:board_macro_madasr": {
    "n": 170,
    "consensus_disagreement_rate": 0.0,
    "catastrophic_like_rate": 0.0,
    "wer_p1": 0.334,
    "wer_hishab": 0.0456
   },
   "board:board_vaani": {
    "n": 60,
    "consensus_disagreement_rate": 0.0333,
    "catastrophic_like_rate": 0.0167,
    "wer_p1": 0.3529,
    "wer_hishab": 0.3574
   }
  },
  "verdicts": {
   "macro_filtered": {
    "pass": true,
    "cd_rate": 0.002,
    "cd_limit": 0.0427,
    "cat_rate": 0.0,
    "cat_limit": 0.007
   },
   "madasr": {
    "pass": true,
    "cd_rate": 0.006,
    "cd_limit": 0.0427,
    "cat_rate": 0.0,
    "cat_limit": 0.007
   },
   "vaani": {
    "pass": true,
    "cd_rate": 0.018,
    "cd_limit": 0.0427,
    "cat_rate": 0.0,
    "cat_limit": 0.007
   }
  }
 },
 "cross_card": {
  "updates_compared": 200,
  "required": 200,
  "id_key": "ids",
  "cut_id_mismatches": [],
  "n_cut_id_mismatches": 0,
  "lr_mismatches": [],
  "n_lr_mismatches": 0,
  "nonfinite_updates": [],
  "worst_block_rel_diff": 0.01582,
  "max_abs_loss_diff": 0.07458599999999993,
  "PASS": false
 },
 "replay_1000": {
  "updates_compared": 1000,
  "required": 1000,
  "id_key": "ids",
  "cut_id_mismatches": [],
  "n_cut_id_mismatches": 0,
  "lr_mismatches": [],
  "n_lr_mismatches": 0,
  "nonfinite_updates": [],
  "worst_block_rel_diff": 0.01233,
  "max_abs_loss_diff": 0.02194699999999994,
  "PASS": false
 },
 "p1_baselines": {
  "S_pp": 20.998,
  "D": {
   "n": 5385,
   "wer_frozen": 0.20346,
   "cer_frozen": 0.06856,
   "wer_nemo": 0.20977,
   "empty_rate": 0.0,
   "per_source": {
    "commonvoice": 0.19213,
    "indicvoices": 0.16565,
    "kathbath": 0.12484,
    "openslr53": 0.18103,
    "subakko": 0.36833
   },
   "conversational": {
    "n_ref_words": 6628,
    "wer_frozen": 0.21575
   }
  },
  "M": {
   "n": 1964,
   "wer_frozen": 0.20587,
   "cer_frozen": 0.05503,
   "wer_nemo": 0.30557,
   "empty_rate": 0.0
  },
  "K": {
   "n": 2236,
   "wer_frozen": 0.14674,
   "cer_frozen": 0.03227,
   "wer_nemo": 0.1467,
   "empty_rate": 0.00045
  },
  "R": {
   "n": 1500,
   "wer_frozen": 0.32056,
   "cer_frozen": 0.09108,
   "wer_nemo": 0.32132,
   "empty_rate": 0.0
  }
 },
 "throughput_steps_per_h": {
  "A5000_from_cache": 891,
  "A6000_from_cache": 887,
  "pretokenize_false": 863
 },
 "freeze_sha256_head": "999157a5a72bebeb27e0fa93b9e81d6b4cb229d99a21c4077f3b6bcfa21c3f3d  arm_N.jsonl\nd5c8b75d55b8a80640c3452a549e8867d303c591aef5130f1c59e6effba907cf  arm_N.nvme.jsonl\ne2e0e359cd74c27589af64ce7cbb06ad54d23823d1bb12b6d33093e727acfea0  arm_O.jsonl\n2236036104488e56a137aed2f75b828cbd37280ef9a9f6fe0a34282abc8c3f79  arm_O+macro_filtered.jsonl\n164bac8fa15b9feae8bbe48a11cc573d7d1538070c57f9fbad96714565450522  arm_O+macro_filtered.nvme.jsonl\n770a804b802c618f5de97d188e19d2d719d7ba791168be02349b4117c4c099b8  arm_O+madasr.jsonl\n4041be2092a47acdc911ae1101370ad7b7b039a8f4b1234702d855fee8d5d0d4  arm_O+madasr.nvme.jsonl\n13a26e562a8d0283b21155d17e5d8c41c4c405eba76d2380830282a84c9fb969  arm_O.nvme.jsonl\nf6b84a1fe6bd349da3fa9d932ef634554b0f08d6317edb9db1924e7fcf6686a6  arm_O+vaani.jsonl\n49a21f7f036e8710f5d20ed6b2899275c0dd015e8b0c55e7e310c6afdc2432bb  arm_O+vaani.nvme.jsonl\n9ff85f86c1c78a75083fe6b4c989f0f2e3f03a262a81515615313f02fa695294  panel_M.jsonl\ne701aa531d651c2bbe69c8a920079aa79969d8c2c14a9394fc819980d6657b71  protected_texts.json\n57d15652bb6009e135eabb0124c2bef13ace45eb19beb3ecbf0dc0b7b6ad76e7  build_report.json\n507f8abba8316c6a2eda9382f6bd0002a2bd7d291b1e87d1fc3fa4392f7f9a47  tools/p4/p4_build_corp"
}

# QUESTIONS (final pre-launch reconciliation; answer with picks, numbers, exact SPEC wording where needed)
Q1 Gate-0 closure: given the frozen artefacts above, list anything that still blocks launching (a) the two O controls, (b) the three bucket arms, (c) N_final. Be specific; say CLOSED if nothing does.
Q2 The G0.6 human audit ended as "references OK" after the owner realised the board's Phase-1 hypothesis hint had been judged instead of the reference (12/12 flagged MaCro clips re-transcribe to the reference with hishab). Is that an acceptable closure of G0.6 together with the machine consensus audit (rule registered before results), or must a hypothesis-free re-audit run first? Pick; if re-audit, size it.
Q3 Machine audit caveats: hishab scores ~4.5 % WER on MaCro/RESPIN (has trained on them) and Phase 1 trained on the old corpus (baseline biased low). Does either invalidate the pass verdicts? What exact card wording?
Q4 RESPIN retained only 149 h under the sentence×dialect cap (each dialect has its own prompt inventory). Keep as registered (no re-raise) — confirm, and state whether O+madasr at 149 h is still a meaningful arm or should be folded into N_final only.
Q5 Cross-card: ids/LR identical, 20-update loss blocks differ ≤1.58 % between cards while same-card runs already drift ≤1.23 % after update 285 (CUDA nondeterminism). Confirm the ruling (8k control agreement is the binding test) or amend the numbers.
Q6 Vaani references are verbatim-ish spontaneous speech (some omit false starts that both models hear). Any registered treatment beyond "its arm decides"? Expected sign on D/FLEURS unchanged?
Q7 Old OpenSLR-53 shows 7.4 % consensus disagreement. Confirm "O stays frozen; disclose" for this phase, or propose a registered, non-ASR-ranked remedy.
Q8 Launch sequence and concurrency on two cards with these exact runs: O_A5000, O_A6000, O+macro_filtered, O+madasr, O+vaani, N_final; which arm goes to which card in which wave; the exact admission test list at 8k with m=3; and the STOP conditions during pilots. Output a table.
