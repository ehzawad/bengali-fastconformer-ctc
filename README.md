# bengali-fastconformer-ctc

Training and evaluation code for
[`ehzawad/stt_bn_fastconformer_ctc`](https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc):
a FastConformer-CTC large (115.6M parameters, 8× subsampling, 1,024-piece Bengali BPE) trained
on 1,692 hours of human-supervised Bengali, greedy decode, no language model. On the FLEURS
Bengali test split it reads 17.12% WER; on the official Vaani and SPRING-INX test splits it leads
every public Bengali model measured in
[bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark).

The repository is the recipe, end to end: how the corpus was built and audited, how the model was
trained and the checkpoint chosen, and how it was evaluated — with the rules encoded in code rather
than in a notebook.

## What is here

```
bnasr/
  build_corpus.py        corpus construction with the admission audits (script purity, protected-
                         panel sentence and audio-fingerprint overlap, PCM-hash de-duplication,
                         per-transcript caps, duration band) and the exclusion ledger
  union_build.py         reproducible union of frozen per-source manifests
  ingest_*.py            per-source ingest: Bengali.AI competition data, Common Voice, Kathbath,
                         RESPIN/MADASR, parquet-hosted corpora (IndicVoices, Vaani, SPRING-INX)
  admission_test.py      the measured gates a candidate source must pass to enter the corpus
  leakage_audit.py       text-key and audio-fingerprint overlap between training data and test sets
  listening_board.py     builds the human listening audit (audio + reference only, no hypothesis)
  train.py, train_core.py   NeMo FastConformer-CTC trainer: warm start, Lhotse dynamic batching,
                         checkpoint resume, validation on the panels, eligibility guards,
                         watchdog, plateau and horizon stops
  common.py              registered constants: panels, selection score, guards, tie rule
  scoring.py             the frozen scoring surface (normaliser, WER/CER counts)
  normalize.py           Bengali text normalisation
  panel_eval.py          decode and score the development and held-out panels; one-shot panels
                         are gated and every decode is written to an append-only audit log
  progress.py            the validation table with eligibility and the selected checkpoint
  qualify.py             final selection + qualification, export to .nemo with exact-WER check
  export_nemo.py         rebuild a checkpoint as a .nemo archive and verify it reproduces
  final_test.py          the sealed one-decode test procedure with a resumable shard log
  paired_bootstrap.py    paired sentence-cluster bootstrap for model comparisons
  recompute_unrounded.py, collapse_check.py, cross_card_check.py, card_agreement.py,
  machine_audit.py, nvme_cache.py, baseline_curve.py, macro_wer.py, manifest.py, hf_fetch.py
configs/ctc_from_nvidia.yaml   the configuration the released model was trained with
configs/ctc.yaml               the same recipe warm-started from an earlier Bengali checkpoint
scripts/                       ingest chains and bundle download helpers
tests/                         unit tests for the normaliser and the competition-WER scorer
data/tokenizer_v1024_corpus/   the 1,024-piece SentencePiece tokenizer
ops/systemd/worker.service     a supervised trainer unit
```

## The recipe

**Corpus.** Seven public sources, all human-read or human-transcribed, 1,084,696 clips / 1,692.4 h
after decoded-PCM hash de-duplication: Bengali.AI Speech competition train (filtered by a public
WER<15 id list; transcripts are the competition's own), IndicVoices, OpenSLR-53, SUBAK.KO, Kathbath,
Common Voice (validated), FLEURS train. Every row entered through `build_corpus.py`'s gates; the
exclusion ledger is written next to the corpus. Three further corpora were evaluated and rejected
on measured evidence (script-purity and duplicate rates), and their official test splits are used
as held-out sets instead. No pseudo-labels, no synthetic audio, no alignment-mined text.

**Training.** Encoder initialised from `nvidia/stt_en_fastconformer_ctc_large`, new Bengali CTC
head, 120,000 updates at an effective batch of 252 utterances, AdamW (0.9, 0.98), linear warm-up
over 2,000 updates to 5e-4 then cosine decay to 1e-5, bf16, one 24 GB GPU. Validation every 2,000
updates on four development panels (unseen-speaker mix, Bengali.AI validation, Kathbath validation,
MADASR dev) with a weighted selection score and regression guards that make a checkpoint
*ineligible* rather than merely worse (per-source regression, conversational-slice regression,
empty-output rate).

**Selection.** Lowest eligible score across all validations; ties within 0.05 pp go to the earlier
checkpoint. The selected checkpoint is exported to `.nemo` and the export must reproduce the
recorded development WER exactly before it is qualified.

**Evaluation.** Held-out panels (an unseen-speaker holdout and the official Vaani and SPRING-INX
splits) are decoded once, after selection, through a gated tool that logs every decode; the FLEURS
test split is decoded by `final_test.py` under a sealed procedure with paired sentence-cluster
bootstraps against the reference models.

## Reproducing

Requires an NVIDIA GPU with ≥20 GB, `nemo_toolkit[asr]==2.7.3` (see `requirements.txt`), and the
source corpora, which are not redistributed here.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q                       # no GPU needed
scripts/run_ingest_chain.sh                                # per-source ingest (see each script)
.venv/bin/python bnasr/build_corpus.py --arm <arm>          # corpus + admission audits
.venv/bin/python bnasr/train.py --config configs/ctc_from_nvidia.yaml --run-name <name>
.venv/bin/python bnasr/progress.py                         # validation table, eligibility, selection
.venv/bin/python bnasr/qualify.py --device cuda            # select, qualify, export .nemo
.venv/bin/python bnasr/panel_eval.py --ckpt <.nemo> --panels D M K R --out <json> --save-hyps <json>
.venv/bin/python bnasr/final_test.py --ckpt <.nemo> --out-dir <dir>
```

Runs, manifests, audit logs and evaluation outputs are written under `experiments/` (gitignored).
Local paths to the FLEURS evaluation set and the NVIDIA base checkpoint are set through
`BNASR_FLEURS_DIR`, `BNASR_FLEURS_MANIFEST` and `init_N` in the config.

## Licence

Code: [Apache-2.0](LICENSE). The model weights are a separate work, released on Hugging Face under
CC-BY-SA-4.0 (the share-alike term is carried from OpenSLR-53). Upstream corpus terms are those of
their publishers and are not relicensed by this repository.

## Citation

See [`CITATION.cff`](CITATION.cff).

Author: Emrul Zawad (ehzawad).
