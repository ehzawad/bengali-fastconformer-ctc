```markdown
---
language: bn
license: cc-by-sa-4.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags: [asr, bengali, fastconformer, ctc]
---

# stt_bn_fastconformer_p4 — Bengali FastConformer-CTC large (Phase 4)

## Dataset

1,084,696 clips / **1,692.4 h** of human-labelled Bengali speech, counted after decoded-PCM hash de-duplication
(MaCro and Common Voice Bengali overlap; shared recordings are counted once). No pseudo-labels, no TTS audio.

| source | hours | licence | attribution |
|---|---:|---|---|
| Bengali.AI Speech competition, `train` split ∩ tugstugi WER<15 ids ("MaCro") | 726.5 new | **`<LICENCE PLACEHOLDER — UNVERIFIED>`** see note | Bengali.AI / Kaggle `bengaliai-speech`; id list by tugstugi |
| OpenSLR-53 Bengali | part of the 984.1 h Phase-1 pool | **CC-BY-SA-4.0** | © 2016–2018 Google Inc. |
| FLEURS `bn_in` (train split only) | ″ | CC-BY-4.0 | Google Research |
| Common Voice Bengali | ″ | `<LICENCE PLACEHOLDER — no archived record>` | Mozilla Foundation |
| Kathbath Bengali | ″ | `<LICENCE PLACEHOLDER — no archived record>` | AI4Bharat |
| IndicVoices Bengali | ″ | `<LICENCE PLACEHOLDER — no archived record>` | AI4Bharat |
| Subakko Bengali | ″ | `<LICENCE PLACEHOLDER — no archived record>` | Subakko |
| base checkpoint `nvidia/stt_en_fastconformer_ctc_large` | — | CC-BY-4.0 | NVIDIA NeMo |

**Licence lineage.** OpenSLR-53 is CC-BY-SA-4.0, so its share-alike terms propagate to these weights: this model is
released **CC-BY-SA-4.0**. **MaCro open item:** the Bengali.AI competition licence (reported as CC-BY-4.0) is *not
independently verified* on our side; the placeholder above stands until the rules text is archived with a checksum.

## Benchmark

FLEURS Bengali **official test**, 920 recordings / 349 sentences, greedy CTC, frozen scorer, corpus WER = Σedits/Σref
words, paired sentence-cluster bootstrap (349 clusters, 20,000 draws, seed 20260905). Phase 4 was decoded on FLEURS test
**twice**, and both events are reported here.

## Output

| event | checkpoint | WER % | CER % | status |
|---|---|---:|---:|---|
| 1 (2026-09-13) | `step_88000`, run incomplete (88k/120k) | 17.5441 | 5.2704 | **interim, unqualified** — not a registered result |
| 2 (final, sealed) | `step_116000`, selected | `<WER>` | `<CER>` | registered headline |

Registered success bar: WER < **16.54 %** (frozen Phase 2) with the 95 % CI of (P4 − P2) entirely below 0.
**Verdict: `<PASS / FAIL>`** — final WER `<WER>` vs 16.54, Δ `<Δ pp>`, 95 % CI `<[lo, hi]>`.
Reference points under the same scorer: Phase 2 16.5362 / 4.6781, Phase 1 19.6723 / 5.7915.

Seven-panel results (D, M, R, K, holdout, Vaani, SPRING one-shots): `<LINK PLACEHOLDER>`
```