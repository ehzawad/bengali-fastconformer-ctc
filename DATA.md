# Preparing the data

The data recipe behind [ehzawad/stt_bn_fastconformer_ctc](https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc).

Everything below runs from the repository root with the project virtualenv
(`python -m venv .venv && .venv/bin/pip install -r requirements.txt`). Each step is
resumable and idempotent (already-converted clips are skipped). Nothing here redistributes
any dataset: every source is fetched from its publisher under its own licence.

## 0. Base checkpoint

```bash
.venv/bin/python bnasr/fetch_init.py
```
Downloads `nvidia/stt_en_fastconformer_ctc_large` into `experiments/inits/`, records its
SHA-256 in `experiments/inits/INITS.json` and prints the two config lines. The trainer
refuses to start from a checkpoint whose hash prefix differs from `paths.init_N_sha256_prefix`.

## 1. The six training sources → `data2/<source>/`

| Source | Licence | Access | Command |
|---|---|---|---|
| OpenSLR-53 | CC-BY-SA-4.0 | direct download (16 zip shards, ~20 GB) | `python bnasr/prepare/openslr53.py` |
| FLEURS `bn_in` train (+validation) | CC-BY-4.0 | Hub, open (`datasets`) | `python bnasr/prepare/fleurs.py` |
| Common Voice Bengali | CC0 | manual download from [Mozilla Data Collective](https://datacollective.mozilla.org) (per-user licence acceptance), then | `python bnasr/prepare/commonvoice.py --raw <release>/bn` |
| Kathbath Bengali | CC-BY-4.0 | Hub, gated (`ai4bharat/Kathbath`: accept on the page, `huggingface-cli login`) | `python bnasr/prepare/kathbath.py --fetch` |
| IndicVoices Bengali | CC-BY-4.0 | Hub, gated (`ai4bharat/IndicVoices`) | `python bnasr/prepare/indicvoices.py --fetch` |
| SUBAK.KO | CC-BY-4.0 per its card | Hub, open (`SUST-CSE-Speech/SUBAK.KO`) | `python bnasr/prepare/subakko.py --fetch` |

Each tool writes `data2/<source>/wavs/` (16 kHz mono PCM16; IndicVoices keeps its FLAC
verbatim) and `data2/<source>/manifest.jsonl` with `audio_filepath, text, duration, source,
speaker_id, key`. `key` is a stable per-source identifier (OpenSLR utterance id; record
index in the publisher's release order for the parquet and Common Voice sources), which is
what the released split lists refer to. Every tool takes `--limit N` for a quick check.

Filters applied at this stage, identical to the released corpus: duration 0.3–30 s;
Common Voice validated rows only, excluding the release's own test and dev clips;
IndicVoices rows whose human verification is "excellent", with no annotation tag and in
Bengali script. FLEURS folds train and validation into the pool by default (that is what
the released model saw; pass `--splits train` to keep validation clean).

## 2. Training pool, development panel, holdout → `data2/*.json`

```bash
.venv/bin/python bnasr/prepare/carve.py            # --from-lists when data/panels/ is present
```

`data/panels/{train_pool,val_corpus,holdout_corpus}.jsonl.gz` list every row of the released
training pool (552,932), development panel D (5,385) and unseen-speaker holdout (6,767) as
`(source, key, text_sha)`. With the same publisher releases the carve reproduces those files
row for row and reports any key that is missing or whose transcript moved. If a release has
changed, `--fresh` carves a new speaker-disjoint holdout (~10 h) and panel (~8 h), seeded,
with per-source caps; both modes then drop rows the 1,024-piece tokenizer cannot fit into
the CTC frame budget (12.5 frames/s, ≤ 11 tokens/s).

## 3. The Bengali.AI competition source and the other panels → `data4/`

The released model's corpus adds 726 h of the Bengali.AI Speech competition data
(Kaggle `bengaliai-speech`, CC-BY-4.0), selected by tugstugi's public WER<15 id list with
the competition's own transcripts, and evaluates on three further official splits.

```bash
# Kaggle: needs an account that has joined the competition; KAGGLE_API_TOKEN or ~/.kaggle/access_token
scripts/kg_comp_bundle.sh bengaliai-speech data4/macro/raw/bengaliai-speech.zip
# the id-list CSVs tugstugi published with the competition's 1st-place solution (Kaggle) -> data4/tugstugi/csv/
# (scripts/kg_get.sh <owner/dataset> <file> <out> downloads one file of a Kaggle dataset with a KGAT token)
.venv/bin/python bnasr/ingest_macro.py --bucket macro_filtered        # training bucket
.venv/bin/python bnasr/ingest_macro.py --bucket macro_validated       # panel M (never trained on)
.venv/bin/python bnasr/ingest_kathbath_valid.py                       # panel K: Kathbath official valid
# optional: IISc RESPIN / MADASR (gated by the publisher's data-use form) -> panel R
.venv/bin/python bnasr/ingest_respin.py --set-dir <dev dir> --source madasr_dev --out data4/respin/manifest_madasr_dev.jsonl
# held-out official splits (never trained on): Vaani, SPRING-INX R1/R2 via bnasr/ingest_parquet.py
```

The validation panels the trainer scores are listed in the config (`val.panels`). The
selection score is defined on the registered set — D, M, K and, when MADASR is present, R —
and the trainer refuses to run a validation with a listed panel missing rather than silently
re-weighting. For a run without the Bengali.AI data, edit `val.panels` to the panels you
have; the guards against the shipped baselines (`data/baselines/`) still apply to D.

## 4. Corpus build and training

```bash
.venv/bin/python bnasr/build_corpus.py --arm O+macro_filtered        # admission audits, exclusion ledger, arm manifests
.venv/bin/python bnasr/train.py --config configs/ctc_from_nvidia.yaml --arm O+macro_filtered --run-name <name>
```

`build_corpus.py` runs the Gate-0 audits (script purity, protected-panel text and audio-
fingerprint overlap, PCM-hash de-duplication, per-transcript caps, duration band) and writes
`experiments/corpus/arm_<arm>.jsonl` plus the exclusion ledger; `--skip-missing-buckets`
builds arm `O` from the six sources alone. The trainer validates every 2,000 updates,
applies the eligibility guards, and stops at the horizon or on the registered plateau rule.
`bnasr/progress.py` prints the validation table and the selection; `bnasr/qualify.py`
exports the selected checkpoint and verifies it reproduces its development WER.

## Quick check from a fresh clone

What was actually run to verify this document (2026-09-15): fresh clone → `fetch_init.py` →
`openslr53.py --shards 0 --limit 200` (one real shard: download, extract, join, convert),
`fleurs.py --limit 200`, `subakko.py --fetch --limit 300`, `kathbath.py --fetch --limit 300`,
`indicvoices.py --fetch --limit 300` (one shard each) → `carve.py --fresh` on that pool →
`train.py --config <copy of ctc_from_nvidia.yaml with val.panels: {D: ...}> --manifest
data2/train_corpus_ready.json --run-name smoke --steps 20 --val-at 10 --val-limit 40` on one
24 GB GPU: 20 updates, two validations, clean exit. Separately, `carve.py --from-lists` over
the full prepared sources reproduced the released pool, panel and holdout with 0 missing rows
and 0 transcript mismatches.
