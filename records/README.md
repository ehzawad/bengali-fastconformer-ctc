# Run records

The registered contract (SPEC.md), the adversarial review rounds (council/), the audit logs, corpus and
admission reports, and the evaluation bundles of the run that produced `ehzawad/stt_bn_fastconformer_ctc`.
Kept verbatim from the records repository they were written in, internal names included; the code they
refer to as `tools/p4/p4_*.py` is `bnasr/*.py` here (see the top-level README).

---

# bengali-fastconformer-phase4

Phase 4 of a Bengali speech-to-text programme: a FastConformer-CTC large retrained from
the NVIDIA English checkpoint on **1,692 hours of fully human-supervised Bengali**, up
from the 984 hours of Phase 1. Greedy decode, no language model.

**This phase is finished.** The run ended normally at 120,000 updates on 2026-09-15; the
selected checkpoint is published as
[`ehzawad/stt_bn_fastconformer_ctc`](https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc)
(CC-BY-SA-4.0). See [Status](#status) for the registered results.

Phase 1 shipped separately as
[`ehzawad/stt_bn_fastconformer`](https://huggingface.co/ehzawad/stt_bn_fastconformer)
and is the baseline this phase has to beat. Its development word error rate on the
panel used here is 20.348%.

## What is actually new here

Phase 1 trained on six public corpora. Phase 4 adds a seventh, `macro_filtered`, which
is 726 hours and 43% of the training rows, and it expands the protocol around the
model far more than the model itself. The architecture is unchanged.

The substance of this repository is the **protocol**, not the recipe:

- a single registered contract, [`experiments/p4/SPEC.md`](experiments/p4/SPEC.md),
  written before the run and amended only by dated, numbered sections;
- data admission decided by measurement, with three candidate corpora
  (`madasr`, `vaani`, `spring_inx`) investigated in depth and **rejected** on evidence
  rather than on convenience, with the rejections and their reasoning kept in-tree;
- a composite selection score with regression guards that make a checkpoint
  *ineligible* rather than merely worse, so a slice cannot quietly rot;
- an external review record: 23 adversarial review rounds under
  [`experiments/p4/council/`](experiments/p4/council/), each with the exact packet it
  was given and the verdict it returned, including the rounds that overturned a
  decision already taken.

## Status

**Complete.** Normal ending at update 120,000 (2026-09-15 07:52 UTC, 134.9 GPU-hours on one RTX
A5000). Selection by `p4_common.select_best`: lowest eligible S over 60 validations, ties within
0.05 pp to the earlier checkpoint — **`step_116000`**, S = 16.89126, tied with 118,000 and 120,000.
Qualified per SPEC §8w.2 and exported with exact dev-WER reproduction (`experiments/p4/qualified_selection.json`).

Development panels (frozen scoring surface), Phase 1 → Phase 4 at `step_116000`:

| panel | rows | Phase 1 | Phase 4 |
|---|---:|---:|---:|
| D (unseen speakers, five Phase-1 sources) | 5,385 | 20.348 % | **16.831 %** |
| M (Bengali.AI validation) | 1,829 | 20.739 % | **12.754 %** |
| K (Kathbath validation) | 2,236 | 14.674 % | **11.757 %** |
| R (MADASR dev) | 1,500 | 32.056 % | **30.664 %** |
| S = 0.6 D + 0.2 M + 0.1 R + 0.1 K | | 21.0296 | **16.891** |

FLEURS Bengali test (920 rows, sealed; decoded twice — SPEC §8w.14 / §8w.18): event one at update
88,000 **17.5441 %** WER (interim, unqualified); event two on `step_116000` **17.1162 %** WER /
5.1217 % CER. Against Phase 1 (19.6723 %): −2.56 pp, 95 % CI [−3.13, −1.99]. Against the Phase-2
Qwen3-ASR adapter (16.5362 %): +0.58 pp, CI [−0.23, +1.40] — the registered success bar
(WER < 16.54 % with the interval clear) was **not met** (`meets_success` false).

One-shot panels (SPEC §2; audit log `experiments/p4/protected_panel_decodes.jsonl`): holdout
**15.008 %**; Vaani validation **26.220 %** (26.193 % excluding three colliding manifest rows);
Vaani test **24.767 %**; SPRING-INX R1 validation/test **38.916 / 39.221 %**; R2 **39.477 / 38.495 %**.
Vaani and SPRING were never trained on; SPRING is half code-switched and this vocabulary emits no
Latin script. The one-shot decode had an incident — two panels were decoded and their results lost
to a tool crash, then recovered once under a written approval — recorded in full in SPEC §8w.19–8w.20,
with the tool hardening that followed.

A seven-model reproducibility run on the A5000 (this model, Phase 1, the Phase-2 adapter, two hishab
models, Whisper-medium, wav2vec2) is declared in SPEC §8w.20 as a third FLEURS exposure; its
results are published in [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark).

## The corpus

Arm `O+macro_filtered`: **1,084,696 clips / 1,692.4 hours**, all human-transcribed.
No pseudo-labels, no text-to-speech audio, no Shrutilipi.

| source | clips | hours | share |
|---|---:|---:|---:|
| macro_filtered | 543,420 | 726.5 | 42.9% |
| indicvoices | 273,001 | 574.4 | 33.9% |
| openslr53 | 162,144 | 159.0 | 9.4% |
| subakko | 33,800 | 105.8 | 6.3% |
| kathbath | 45,965 | 81.5 | 4.8% |
| commonvoice | 24,712 | 39.5 | 2.3% |
| fleurs_train | 1,654 | 5.8 | 0.3% |

Every row was admitted through the Gate-0 audits in SPEC §3: script purity, protected-panel
sentence and fingerprint overlap, audio duplicate detection by PCM hash, per-transcript
caps, tokenizer feasibility and a duration band. The exclusion ledger is in
[`experiments/p4/corpus/build_report.json`](experiments/p4/corpus/build_report.json),
broken down by source and by reason.

Upstream licences are archived under
[`experiments/p4/licences/`](experiments/p4/licences/). Those terms govern the data and
are not superseded by this repository's licence.

## Layout

```
experiments/p4/SPEC.md           the contract: gates, scoring, selection, amendments
experiments/p4/council/          23 adversarial review rounds, packets and verdicts
experiments/p4/corpus/           build report, freeze checksums, audit outputs
experiments/p4/licences/         upstream dataset terms
experiments/p4/*.sh              supervisor, launcher, watchdogs, monitor liveness
experiments/p4/tests/            identity-binding and run-gate tests
tools/p4/p4_common.py            scoring, guards, eligibility, selection
tools/p4/p4_train.py             the trainer
tools/p4/p4_build_corpus.py      corpus construction and the Gate-0 audits
tools/p4/p4_panel_eval.py        panel decoding on both scoring surfaces
tools/p4/p4_final_test.py        the one-shot test event, with a resumable shard log
tools/p4/p4_paired_bootstrap.py  paired sentence-cluster bootstrap
configs/p4_ctc_initN.yaml        the live configuration
tests/                           normalisation, MaCro WER and stall detection
```

### Two files that are not Phase 4 and cannot be renamed

`tools/p3_common.py` and `tools/p3_train.py` carry a previous phase's name. They are
here because Phase 4 imports them, and `p3_common` in particular is the **frozen
scoring surface**: `frozen_norm`, `wer_counts`, `cer_counts`. Its SHA-256 is written
into every identity manifest this phase produces, and the tests assert on that value.
Renaming the file would invalidate records already written. The name is historical; the
content is load-bearing.

## Reproducing

Requires an NVIDIA GPU with at least 20 GB, NeMo, and the source corpora, which are not
redistributed here. The training data manifests are build products and are gitignored;
`experiments/p4/corpus/FREEZE.sha256` lets a rebuild be verified byte-exact.

```bash
python -m pytest tests experiments/p4/tests -q      # no GPU needed
python tools/p4/p4_build_corpus.py --arm O+macro_filtered
experiments/p4/run_supervised.sh N_O+macro_filtered_H120k
```

The supervisor keeps the run alive across crashes and terminal closure, refuses to
resume over a recorded stop, and re-decides the card at every start.
`experiments/p4/monitors.sh` reports whether each monitor is actually running, measured
from the process table rather than inferred from a log line.

## Model and schedule

FastConformer-CTC large, 18 layers, model dimension 512, 8 attention heads, 8×
depthwise-striding subsampling, about 115.6M parameters, over a 1,024-token Bengali
SentencePiece vocabulary built from the corpus.

AdamW with betas 0.9 and 0.98, epsilon 1e-8, weight decay 1e-3, gradient clipping at
1.0, bf16. Linear warmup over 2,000 updates to a peak learning rate of 5e-4, then cosine
decay to 1e-5 at update 120,000. Effective batch 252, as microbatch 12 with 21
accumulation steps.

## Licence

Code, configuration and records in this repository: [Apache-2.0](LICENSE) (changed from
CC-BY-NC-SA-4.0 on 2026-09-15 by the author; every file here is the author's own work).
The repository is private at the author's discretion.

The model weights are a separate work, released on Hugging Face under
[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/): the share-alike term is
carried from OpenSLR-53. Upstream corpus terms are archived under
`experiments/p4/licences/` and remain in force; they are not relicensed by this repository.

## Citation

See [`CITATION.cff`](CITATION.cff).

Author: Emrul Zawad (ehzawad). Sole author.
