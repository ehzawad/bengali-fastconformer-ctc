# Future work

## Qwen3-ASR Bengali adapter: extend the vocabulary, then fine-tune again

`ehzawad/stt_bn_qwen3_asr` (a LoRA adapter on `Qwen/Qwen3-ASR-1.7B-hf`) reaches 16.54 % WER on
FLEURS Bengali test — within the interval of the FastConformer-CTC — but at ~9.7 s per clip at
batch 1, about 117× the CTC models. Most of that cost is structural: the base tokenizer has no
Bengali merges, so each Bengali character costs ~3 byte-level tokens and the decoder generates
~3× more steps than it needs to. Because of this it was removed from the public comparison
(bengali-asr-benchmark) on 2026-09-15 rather than reported with an unfair latency column.

The fix is a new model, not a runtime tweak:

1. **Extend the tokenizer** with Bengali-script merges trained on the Bengali training transcripts
   (target: ~1 token per Bengali word-piece, i.e. a 3× reduction in generated tokens). Keep the
   base vocabulary intact; append new pieces only.
2. **Initialise the new embedding rows and output rows** (mean of the constituent byte-token
   embeddings is the usual starting point), then **fine-tune** — adapter plus the new embedding /
   output rows, encoder frozen as before — on the same Bengali corpus and the same recipe, with the
   same panels and the same sealed one-decode test procedure so the result is comparable.
3. **Measure the runtime study again** (merged LoRA + static KV cache brought the old adapter from
   9.3 s to 1.5 s per clip at unchanged WER); with the shorter token sequence the two gains stack.
4. Publish as a new checkpoint with its own card and its own benchmark row; never swap it into an
   existing table.

Expected outcome: 2–3× faster decoding at the same accuracy; still an autoregressive decoder, so
still well behind CTC latency. Worth doing if the adapter's accuracy on read speech is needed at
interactive latency; not worth doing for the spontaneous-speech sets, where the FastConformer-CTC
already leads.

## Evaluation

- **Long-form Bengali test set:** none exists publicly with human transcripts, a licence and
  redistributable audio (DL Sprint 4.0 / Bengali-Loop releases transcripts without audio and
  without a licensed test split). Options: ask the Bengali-Loop authors for a licensed evaluation
  subset, or reconstruct multi-minute sessions from a segmented spontaneous corpus and label the
  result a derived condition.
- **RegSpeech12 test** (Bangladeshi dialect, spontaneous, 2,132 clips): the right shape, but the
  Kaggle archive is 12 GB for the whole set, its licence field reads "Unknown", and
  `hishab/titu_stt_bn_conformer_large` lists its sibling corpus (Vasha-Bichita) as training data.
  Revisit if the authors publish a licensed test-only release.
- **Vaani validation manifest:** three file-name collisions in the ingest (10,105 rows, 10,102
  unique paths). Fix the ingest and produce a new manifest version with new hashes; never replace
  the frozen one.
