# Phase 4 — pre-launch card

Owner-requested before the full run resumes: every hyperparameter, the architecture, and the total audio hours.
Every number here was read from the live artifacts this session, not from the contract prose: the architecture from
`experiments/fastconformer_ctc_bn_1kh/final.nemo`'s own `model_config.yaml`, the optimisation from `configs/p4_ctc.yaml`,
the parameter count and layer geometry from a live checkpoint's tensors, and the hours by summing the frozen manifests.

## 1. Architecture — NeMo `EncDecCTCModelBPE`, FastConformer-CTC large

| | |
|---|---|
| parameters | **115,639,795** (115.64 M), counted from the checkpoint tensors |
| features | 80 log-mel, 16 kHz, 25 ms window / 10 ms hop, Hann, n_fft 512, per-feature norm, dither 1e-5 |
| subsampling | depthwise-striding, factor **8**, 256 conv channels → one frame per **80 ms** |
| encoder | `ConformerEncoder`, **18 layers**, d_model **512**, **8 heads**, ff expansion 4 (hidden 2048) |
| conv module | depthwise kernel **9**, **batch_norm** |
| attention | `rel_pos`, full context `[-1, -1]`, pos_emb_max_len 5000 |
| dropout | 0.1 general / 0.1 pre-encoder / 0.1 attention / 0.0 embedding |
| decoder | `ConvASRDecoder`, feat_in 512, 1024 classes + CTC blank = **1025 outputs** |
| tokenizer | SentencePiece BPE, **1024** tokens, `data/tokenizer_v1024_corpus` (sha 6288cf8e) |
| loss | CTC |

**Initialisation.** Warm start from the Phase-1 model `experiments/fastconformer_ctc_bn_1kh/final.nemo`
(sha256 prefix `61362a36`) with a **fresh AdamW state**. Not a resume of Phase 1: only the weights carry over.

## 2. Optimisation — every registered hyperparameter

| | |
|---|---|
| optimiser | AdamW, betas **(0.9, 0.98)**, eps 1e-8, weight decay **1e-3** (0 on biases and norm scales/shifts) |
| gradient clipping | **1.0** global norm |
| precision | **bf16** autocast (master weights fp32) |
| microbatch | **12** cuts |
| accumulation | **21** |
| effective batch | **252** cuts per update |
| warmup | **1,000** steps, 1e-5 → **2e-4** |
| decay | cosine from 2e-4 to **1e-5** at H = **150,000** |
| horizon H | **150,000** updates = 37,800,000 clip presentations |
| seed / shard_seed | 20260905 / 20260905 |
| SpecAugment | 2 freq masks × 15 bins, 2 time masks × 5 % of length |
| bucketing | Lhotse dynamic, 10 buckets, bins [2.27, 2.9, 3.5, 4.2, 5.18, 6.23, 7.65, 9.69, 12.58] s |
| buffers | bucket 20,000, shuffle 10,000, `concurrent_bucketing: false` |
| dataloader | 8 workers, pin_memory, drop_last, duration 0.1–30.0 s, missing file = hard stop |
| validation | every **2,000** updates on panels D, M, K, R; val batch 32 |
| checkpoints | every 500, keep last 2, keep best 3 by S, milestones 8k / 20k / 40k / 80k / 120k / 150k |
| memory watchdog | 13.0 GiB reserved (measured peak 9.23) |

## 3. Gates the run must survive

- **Progress gate at exactly 20,000** (fires once): S ≤ S(P1) − 1.5 = **19.5296** and D ≤ **19.35** and all guards pass, else Phase 4 TERMINATES.
- **Regression guards at every validation, against the measured Phase-1 baseline**: no D source worse by > **1.0 pp**, conversational slice worse by ≤ **0.5 pp**, empty-output rate ≤ **0.1 %**.
- **Plateau** armed at 50,000: anchor = best eligible S; reset on a ≥ 0.20 pp cumulative improvement; stop after **10** consecutive non-reset validations; earliest stop **70,000**.
- **Selection**: S = 0.60·D + 0.20·M + 0.10·R + 0.10·K on the frozen surface; ties within 0.05 pp go to the EARLIER checkpoint; only guard-passing checkpoints are selectable.

## 4. Total audio hours — by corpus, decided at 8,000

| final corpus | rows | hours | one epoch | epochs at H |
|---|---:|---:|---:|---:|
| O, base only (no bucket passes → Phase 4 STOPS) | 541,276 | 965.9 | 2,148 upd | 69.8 |
| O + macro_filtered | 1,084,696 | **1,692.4** | 4,304 upd | 34.8 |
| O + madasr | 636,517 | 1,114.9 | 2,526 upd | 59.4 |
| O + vaani | 627,223 | 1,083.3 | 2,489 upd | 60.3 |
| O + macro_filtered + madasr (most likely union) | 1,179,937 | 1,841.3 | 4,682 upd | 32.0 |
| **N_final, all three** | **1,265,884** | **1,958.7** | 5,023 upd | 29.9 |

### N_final composition, after dedup, protected-set carving, MAX_PER_TEXT=5 and the RESPIN sentence×dialect cap

| source | rows | hours |
|---|---:|---:|
| macro_filtered | 543,420 | 726.5 |
| indicvoices | 273,001 | 574.4 |
| openslr53 | 162,144 | 159.0 |
| madasr (RESPIN) | 95,241 | 149.0 |
| vaani | 85,947 | 117.4 |
| subakko | 33,800 | 105.8 |
| kathbath | 45,965 | 81.5 |
| commonvoice | 24,712 | 39.5 |
| fleurs_train | 1,654 | 5.8 |
| **total** | **1,265,884** | **1,958.7** |

**Read the madasr row carefully.** RESPIN was ingested at 894.2 h but contributes **149.0 h**, because the registered
sentence×dialect cap (≤5 recordings per sentence per dialect, ≤25 per sentence overall) and O-first precedence remove the
rest. **The corpus is 1,958.7 h at most, which is 41.3 h SHORT of 2,000; no configuration reaches 2,000.** What MaCro
satisfies is the registered ≥500 new-hour minimum (G0.8), and RESPIN's headline size does not change that.

## 5. What is NOT in the corpus, deliberately

Common Voice's unconfirmed `other` tier and `invalidated` — owner ruling 2026-09-07, permanently excluded, audio
deleted. SPRING-INX (failed the Bengali-script rule). Shrutilipi and any forced-alignment, subtitle-derived, YouTube-derived
or synthetic-TTS source. Common Voice release 17 in full (nine releases older than the 26.0 we train on; a cumulative corpus contains its predecessors). Every panel: D, M, K, R,
FLEURS test, the holdout, and Vaani/SPRING official splits.
