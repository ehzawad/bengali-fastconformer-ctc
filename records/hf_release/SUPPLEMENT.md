# stt_bn_fastconformer_ctc — supplement

Credits, licence notices, the full evaluation record and the limitations of
[`ehzawad/stt_bn_fastconformer_ctc`](https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc).
Released 2026-09-15. Weights SHA-256 `bea300ce4f1f1df1fa4707562ddaa7c64db6f01121ef3d64a6333635f3626572`.

## 1. Credits and licence notices

The weights are derived from the following works. Each is used under the licence stated; the
modifications made are: audio decoded to 16 kHz mono PCM, transcripts normalised to a single Bengali
scoring surface (Unicode NFC, punctuation removed, digits kept as written), no text or audio altered
otherwise. The trained weights are released under
[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode), which is the share-alike
obligation carried from OpenSLR-53. No warranty is given for the weights or for any source.

| Work | Creator / copyright | Licence | Hours used |
|---|---|---|---:|
| Bengali.AI Speech competition data (Kaggle `bengaliai-speech`, train split) | Bengali.AI | CC-BY-4.0, commercial use permitted per the competition rules ("DATA ACCESS AND USE: CC-BY-4.0", section B.7; verified by the author on the rules page, 2026-09-15). Rows selected by tugstugi's public WER < 15 id list; transcripts are the competition's own | 726.5 |
| IndicVoices, Bengali | AI4Bharat, IIT Madras | CC-BY-4.0 | 574.4 |
| OpenSLR-53, Large Bengali ASR training data set | © 2016–2018 Google Inc. Please cite: Kjartansson et al., "Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali", SLTU 2018 | CC-BY-SA-4.0 | 159.0 |
| SUBAK.KO | SUST CSE Speech group | CC-BY-4.0 per its dataset card | 105.8 |
| Kathbath, Bengali | AI4Bharat, IIT Madras | CC-BY-4.0 | 81.5 |
| Common Voice, Bengali (validated) | Mozilla Foundation and the contributing speakers | CC0-1.0 | 39.5 |
| FLEURS `bn_in`, train split only | Google Research | CC-BY-4.0 | 5.8 |
| `nvidia/stt_en_fastconformer_ctc_large` (initial encoder weights; the Bengali decoder head is new) | © NVIDIA Corporation | CC-BY-4.0 | — |

Total 1,084,696 clips / 1,692.4 h, counted after decoded-PCM hash de-duplication across sources.
Evaluated but **not** used for training: Vaani (ARTPARK-IISc, CC-BY-4.0) and SPRING-INX R1/R2 (SPRING Lab, IIT
Madras, CC-BY-4.0) — their official validation/test splits are reported below as held-out sets.
Considered and excluded from training: Shrutilipi (alignment-mined text), any pseudo-labelled or synthetic
audio, MaDASR, Vaani and SPRING-INX (the last three rejected on measured admission gates).

## 2. Evaluation record

Scorer: corpus WER = Σ edits / Σ reference words on punctuation-stripped, NFC-normalised text; CER on the same
surface. Greedy CTC decoding, no language model.

### 2.1 FLEURS Bengali test (920 recordings)

**WER 17.1162 %, CER 5.1217 %.** Paired sentence-cluster bootstrap (349 clusters, 20,000 draws) against the
Qwen3-ASR-1.7B Bengali adapter (`ehzawad/stt_bn_qwen3_asr`, 16.5362 % / 4.6781 %): +0.5799 pp, 95 % CI
[−0.2311, +1.3962] — not distinguishable.

### 2.2 Development panels (used for checkpoint selection)

| Panel | Rows | WER | CER |
|---|---:|---:|---:|
| D — unseen-speaker mix of Common Voice, IndicVoices, Kathbath, OpenSLR-53, SUBAK.KO | 5,385 | 16.831 % | 5.593 % |
| M — Bengali.AI Speech validation split (never trained on) | 1,829 | 12.754 % | 3.058 % |
| K — Kathbath validation | 2,236 | 11.757 % | 2.463 % |
| R — MADASR (RESPIN) dev subset | 1,500 | 30.664 % | 8.425 % |

Checkpoint selection used only these panels (lowest weighted score, 0.6 D + 0.2 M + 0.1 R + 0.1 K), never a
test set.

### 2.3 Held-out panels (decoded once, after selection)

| Panel | Rows | WER | CER | Note |
|---|---:|---:|---:|---|
| Holdout — unseen speakers from Common Voice, IndicVoices, Kathbath, OpenSLR-53, SUBAK.KO | 6,767 | 15.008 % | 4.469 % | per source: CV 10.81, IndicVoices 15.98, Kathbath 11.77, OpenSLR-53 14.88, SUBAK.KO 15.55 |
| Vaani official validation | 10,105 | 26.220 % | 8.570 % | 26.193 % on the 10,099 rows after excluding 3 colliding pairs (see below) |
| Vaani official test | 12,695 | 24.767 % | 7.945 % | |
| SPRING-INX R1 validation | 15,055 | 38.916 % | 22.359 % | Latin-free rows (7,755): 33.79 % |
| SPRING-INX R1 test | 1,879 | 39.221 % | 22.831 % | Latin-free rows (938): 34.37 % |
| SPRING-INX R2 validation | 10,936 | 39.477 % | 22.667 % | Latin-free rows (5,913): 34.60 % |
| SPRING-INX R2 test | 1,872 | 38.495 % | 21.712 % | Latin-free rows (1,014): 35.00 % |

Vaani and SPRING-INX were not trained on. SPRING-INX is conversational speech in which about half the rows
contain Latin-script (English) words; this model's Bengali-only vocabulary cannot emit Latin script, so every
such word is an error — the "Latin-free rows" column is a diagnostic re-scoring of the same hypotheses.

**Vaani validation manifest defect.** Three pairs of rows in the Vaani validation split as ingested here share
one audio file name with different durations and transcripts (a file-name collision in the ingest); one row of
each pair therefore has a reference that cannot match its audio. Both numbers are reported; the difference is
0.03 pp.

### 2.4 Public benchmark (2026-09-15)

Published in [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark), one model at a time on
an RTX A5000, batch 1, the benchmark's own scorer.

**FLEURS test (920 utterances):** hishab Conformer Large 14.37 %, Whisper Medium 16.23 %, Qwen3-ASR + Bengali
adapter 16.54 %, **this model 17.12 %**, wav2vec2 20.64 %, hishab FastConformer 36.22 %.

**Official test splits of other corpora** (frozen seeded 1,000-utterance subsets; this model's Vaani/SPRING
rows are its hypotheses from §2.3 scored on those utterances; its Kathbath rows are fresh decodes):

| Model | Vaani test | SPRING R1 eval | SPRING R2 eval | Kathbath test (known spk) | Kathbath test (unknown spk) |
|---|---:|---:|---:|---:|---:|
| **this model** | **24.66 %** | **39.24 %** | **38.41 %** | 13.38 % | 13.43 % |
| Whisper Medium (`SayedShaun/bengali-whisper-medium`) | 26.25 % | 39.96 % | 39.70 % | 9.69 % | 9.79 % |
| Conformer Large (`hishab/titu_stt_bn_conformer_large`) | 33.10 % | 42.46 % | 42.24 % | **9.33 %** | **8.79 %** |
| hishab FastConformer | 28.71 % | 46.47 % | 45.08 % | 25.81 % | 29.87 % |
| wav2vec2 (`SayedShaun/bangla-wave2vec2-unigram`) | 31.26 % | 47.28 % | 47.33 % | 19.08 % | 18.80 % |

Conformer Large trained on Kathbath train and validation; this model on Kathbath train only; Whisper's and
wav2vec2's provenance for these corpora is not documented. Intervals, the SPRING Latin/no-Latin split and
per-set latency are in the benchmark README.

## 3. Limitations

- Spontaneous conversational speech is markedly harder than read speech (SPRING-INX 34–39 % vs holdout
  15 %); regional and dialectal speech has not been evaluated beyond the Vaani districts.
- The vocabulary is Bengali script only: English or other Latin-script words in the audio are transcribed
  as Bengali approximations or dropped.
- Output has no punctuation.
- Rare proper nouns and named entities may be mis-transcribed.
- Empty outputs are rare (0–0.16 % of utterances on the held-out panels).

## 4. Model

FastConformer-CTC large (18 layers, d = 512, 8 heads, 8× depthwise-striding subsampling), encoder
initialised from `nvidia/stt_en_fastconformer_ctc_large`, new 1,024-piece SentencePiece Bengali decoder
trained on the training transcripts only. Greedy CTC decoding, no language model.

## 5. Author

Emrul Zawad (`ehzawad`).
