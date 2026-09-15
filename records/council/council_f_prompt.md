You are the mid-ablation reviewer (gpt-6-astra, xhigh) for Phase 4 of a Bengali ASR programme. Councils A-E shaped the contract; Gate 0 closed; the control and the first bucket arm have finished. Be concrete, numeric, adversarial, and willing to say 'do not change it'. Below: the CONTRACT, then MEASURED RESULTS, then QUESTIONS.

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
  ~10 % in per-update loss from update 1. Cause (known PyTorch behaviour): CUDA dropout/mask kernels derive their Philox offsets
  from the launch grid, which depends on the SM count (A5000 64 SMs, A6000 84), so the two cards draw DIFFERENT dropout and
  SpecAugment realisations — statistically equivalent runs, not numerically identical ones. Consequence: the two controls
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
and the ruling of §8d applies: the binding cross-card evidence is the 8k control agreement, which we no longer have (the A6000
control was abandoned at 4,038), so **every arm is compared with O_A5000 and the card of each run is disclosed**; the two cards'
partial controls agreed within 0.08 pp S at 2k and 4k, which is the only cross-card evidence on record.

## 8j. Arm results
- **Control O (8,000 updates, A5000 then A6000):** S 21.073 (P1 21.030). D 20.32 / M 20.57 / K 14.93 / R 32.76, conversational
  21.47, guards pass. Reading: 8k more updates on the old corpus alone gain nothing (+0.04 pp S) — the old corpus is saturated
  under this recipe, so any arm's gain is attributable to its new data.
- **O+macro_filtered (8,000 updates, A6000): ADMITTED.** S 20.531 vs control 21.073, Δ = −0.517 pp, 95 % CI [−0.62, −0.42],
  simultaneous upper bound (m=3, 99.17th pct) −0.44 < 0; all guards pass (D −0.11, conversational +0.42 ≤ 0.50, worst source
  indicvoices +0.79 ≤ 1.00, empties 0.02 %). Panels: D 20.21 / M 18.03 / K 14.90 / R 33.07. Trajectory S: 21.22 → 21.18 → 20.70
  → 20.53 (the conversational guard failed at 2k/4k/6k and cleared by 8k). Test: `tools/p4/p4_admission_test.py`
  (paired sentence-cluster bootstrap, clusters shared across panels, 10,000 draws, seed 20260905), record
  `experiments/p4/admission_macro_filtered.json`.

## 9. Log
- 2026-09-06: council D pre-pilot reconciliation (`experiments/p4/council/council_d_prepilot.md`): STOP pilots until Gate 0 closes; two-card controls, RESPIN sentence×dialect cap, Vaani arm runs, SPRING excluded (48.3 % Latin rows), plateau anchor rule, clause fixes adopted; launch checklist = council D Q8. Ledger ≈ 298 GPU-h (soft).
- 2026-09-05 (later): council C interim audit (`experiments/p4/council/council_c_interim_audit.md`) → STOP pilots until Gate 0 closes;
  12 required changes adopted above; G0.9(b) reading for MaCro∩CommonVoice duplicates put to the owner.
- 2026-09-05: owner decisions recorded; councils A/B run; MaCro competition bundle (24.7 GB, 963,636 mp3, 32 kHz mono 48 kbps)
  and IISc RESPIN `train_bn_clean` (87.1 GB) downloads started; tugstugi CSV inventory and 0-overlap-with-FLEURS audit done
  (`data4/tugstugi/csv`). tugstugi's `validated` list == competition `valid` split (29,583 of 29,588); his filtered list ⊂ `train`.

# MEASURED RESULTS (2026-09-07)
Frozen corpora: O 541,276 rows / 965.9 h; O+macro_filtered 1,084,696 / 1,692.4 h; O+madasr 636,517 / 1,114.9 h;
O+vaani 627,223 / 1,083.3 h; N(all three) 1,265,884 / 1,958.7 h. New hours: macro 726.5, madasr 149.0, vaani 118.1.
Panels: D 5,385 (locked dev, weight .60), M 1,829 MaCro-validated sentences (weight .20), R 1,500 RESPIN dev (.10),
K 2,236 Kathbath valid (.10). P1 baseline S 21.030 (D 20.35 / M 20.74 / K 14.67 / R 32.06, conversational 21.57).

CONTROL O, 8,000 updates: S 21.073. D 20.32, M 20.57, K 14.93, R 32.76, conversational 21.47, guards pass.
  S trajectory 21.54 (2k) -> 21.32 -> 21.17 -> 21.07 (8k). Reading: 8k more updates on the old corpus alone gained +0.04 pp.
ARM O+macro_filtered, 8,000 updates: S 20.531. D 20.21, M 18.03, K 14.90, R 33.07, conversational 21.89, empties 0.019 %.
  S trajectory 21.22 -> 21.18 -> 20.70 -> 20.53. Conversational guard FAILED at 2k/4k/6k (+1.24/+1.13/+1.21 vs control) and
  CLEARED at 8k (+0.42, limit +0.50). Per-source vs control at 8k: commonvoice -2.54, openslr53 -0.85, subakko -1.07,
  kathbath -0.29, indicvoices +0.79 (limit +1.00). Admission test (tools/p4/p4_admission_test.py): 9,727 sentence clusters,
  10,000 paired draws, seed 20260905, clusters coupled across panels; delta S = -0.542 pp, 95 % CI [-0.736, -0.352],
  simultaneous upper bound at 1-0.025/3 = -0.314 < 0. ADMITTED. (Independent check: the tool's per-panel WERs reproduce the
  trainer's to 3 decimals from val_preds + manifests, so hypothesis ordering is verified.)
RUNNING: O+madasr (update ~700/8000, ~620 updates/h while the co-tenant is active). QUEUED: O+vaani, then N_final.
Schedule continuity VERIFIED: the pilots run the 150k-horizon cosine, logged lr at update 8000 = lr_at(8000) to 15 digits,
so a winning arm's 8k checkpoint is the literal prefix of the main run (no second warmup, no schedule discontinuity).
Throughput 620-830 updates/h (A6000 shared with a co-tenant's notebook; A5000 fully occupied by the same co-tenant).
Main run: 142k further updates = 171-229 GPU-h = 7-9.5 days. Headline: FLEURS test decoded ONCE, paired vs frozen Phase-2
(16.54 %) and Phase-1 (19.67 %); success = WER < 16.54 with the 95 % CI of (P4 - P2) entirely below 0.

# QUESTIONS (answer each with a pick and numbers; "keep as registered" is a valid answer and is preferred where the
# alternative would be a post-hoc change)
Q1 CHECKPOINT SELECTOR FOR THE SHIPPED MODEL. The main run selects its final checkpoint by the lowest eligible S, where S
   weights M (MaCro-validated dev) at 0.20 - and MaCro is now training data (transcript- and audio-disjoint from M, verified,
   but same domain, same collection). The headline is FLEURS (read Wikipedia sentences). Risk: S prefers checkpoints that are
   more MaCro-like rather than better overall. Options: (a) keep S frozen as registered for both admission AND final
   checkpoint selection; (b) keep S for admission but select the final checkpoint on D alone (the locked panel carried since
   Phase 1); (c) select on a pre-registered blend that excludes any panel whose domain is in training. Pick ONE. If you pick
   anything but (a), explain why it is not post-hoc selector shopping, and state the exact wording.
Q2 CONVERSATIONAL REGRESSION OVER 150k UPDATES. MaCro's arm failed the conversational guard at 2k/4k/6k and cleared it at 8k
   with +0.42 pp (limit +0.50). In the main run MaCro is ~43 % of presented rows for 142k more updates. Registered behaviour
   today: a guard failure marks a checkpoint INELIGIBLE for selection but does not stop the run; stop rules are non-finite
   loss, empties >1 %, D > 25.35 %. Is checkpoint-ineligibility sufficient, or must the main run carry an explicit
   conversational stop/abort rule? If a rule, give the exact threshold and the number of consecutive validations.
Q3 N_final AS A SELECTION STATISTIC. Registered: N_final is tested like the others plus S(N_final) <= S(best single arm)
   + 0.20 pp; its interval is a conditional selection statistic. Two sub-questions: (i) what EXACTLY may the model card and
   any write-up claim about the corpus choice, in one sentence; (ii) should the fallback rule (lowest-S passing single arm
   if N_final fails) be tightened, given only macro can independently satisfy the 500-new-hour minimum?
Q4 HORIZON. H=150,000 was registered when the expected corpus was ~2.3 M rows. If only MaCro is admitted the corpus is
   1.08 M rows => 150k = 34.9 nominal epochs (Phase 1 ran 18 on 984 h). Changing H changes the cosine and would invalidate
   the pilot prefix. Keep H=150,000 (with the plateau rule: anchor at 50k, reset on >=0.20 pp cumulative improvement, stop
   after 10 consecutive non-reset validations, earliest stop 70k), or amend? If amend, how, given the prefix problem?
Q5 SKIPPING ARMS. MaCro carries 726.5 of the 993 available new hours and is already admitted. The owner may choose to skip
   O+madasr and O+vaani (~20 GPU-h, ~30 h wall-clock) and start the main run on MaCro alone. What is LOST scientifically if
   those arms are skipped, and is there any circumstance in which skipping is the better decision? Give a recommendation the
   owner can act on.
Q6 WHAT THE MACRO RESULT ACTUALLY SHOWS. delta S = -0.54 pp is dominated by M (-2.54 pp on MaCro's own domain) while D moved
   only -0.11 pp. Is the admission decision therefore mostly "MaCro helps MaCro"? What would the honest one-sentence
   characterisation be, and does it change what we should expect on FLEURS?
Q7 Anything in the contract that is now stale, contradictory, or that these results have falsified. List exact lines.
Q8 The single most likely way the MAIN RUN wastes a week, and the cheapest registered check that prevents it.
Output: Q1-Q8 with picks, then a short list of REQUIRED SPEC amendments (exact wording), then explicit STOP conditions for
the main run.
