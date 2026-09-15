# MaCro / Bengali.AI Speech competition — licence record

**Status: CLOSED 2026-09-15 by the owner's verification.** On 2026-09-15 (~10:25 UTC)
the owner, logged in to Kaggle, viewed
`https://www.kaggle.com/competitions/bengaliai-speech/rules` and confirmed
(decision picker, option "Verified: CC-BY-4.0, commercial OK") that the rules
state **DATA ACCESS AND USE: CC-BY-4.0** and that commercial use is permitted
(section B.7). This matches council AA's reading. The verbatim rules text is
NOT archived here (JS-rendered page; no authenticated fetch from this box);
the record of verification is the owner's statement above, and the model card
attributes the source as Bengali.AI Speech (Kaggle `bengaliai-speech`),
CC-BY-4.0. Phase 4 may ship under a commercially usable licence
(CC-BY-SA-4.0, owner's choice 2026-09-15). The history below is kept as written.

---

**Status as of 2026-09-13 (superseded): UNRESOLVED. This blocks publishing Phase 4 under any
commercially-usable licence.**

## Why this file exists

SPEC §3 G0.2 requires: "Licence record for every source archived under
`experiments/p4/licences/`." That directory held records for **RESPIN** and
**SPRING_INX** only — both of which were REJECTED — and nothing for
`macro_filtered`, which is the largest ADMITTED source at 726.5 h / 42.93 % of
the Phase-4 corpus. The gate was satisfied for the sources we did not use and
missed for the one we did. Found 2026-09-13 while checking whether the owner's
licence reversal (NC-SA → a commercially-usable licence) is legally available.

## What the source actually is

Kaggle competition `bengaliai-speech`, `train` split, intersected with
tugstugi's WER<15 id list. Transcripts are the competition's own human
reference (`train.csv` sentence column), never tugstugi's cleaned text — see
`tools/p4/p4_ingest_macro.py`. Bundle on disk:
`data4/macro/bengaliai-speech.zip`, 24,732,338,052 bytes, acquired
2026-09-05T19:26:21Z (`data4/macro/download.log`).

## What is and is not established

**Council AA (2026-09-13) asserts** the competition rules state
"DATA ACCESS AND USE: CC-BY-4.0" and that section B.7 expressly permits
commercial and non-commercial use.

**That assertion is NOT independently verified.** Attempts made 2026-09-13:

* `https://www.kaggle.com/competitions/bengaliai-speech/rules` — the rules text
  is JavaScript-rendered; a plain fetch returns no rules content.
* Kaggle API — returns `401 Unauthenticated`; no `~/.kaggle/kaggle.json` on this
  box (the `.venv-kaggle` CLI exists but has no credentials).
* The acquired bundle itself — `unzip -l` shows only `train.csv` and
  `sample_submission.csv` alongside the audio. **No LICENSE, rules or terms file
  was distributed inside the download**, so the bundle carries no
  acquisition-time licence evidence of its own.
* HF mirrors `bengaliAI/bengaliai-speech` and `arbitropy/bengaliai-speech` —
  both exist, neither carries a licence tag.

So the licence is plausibly CC-BY-4.0 and plausibly permits commercial use, and
we cannot currently prove it from anything on this machine.

## What must be done before Phase 4 ships commercially

1. Fetch the competition rules while authenticated (the owner has a Kaggle
   account; `kaggle competitions list` after placing `kaggle.json`, or simply
   viewing the rules page logged in) and archive the rules text verbatim here,
   with the retrieval date and a sha256 of the captured text.
2. Capture section B.7 and the DATA ACCESS AND USE line specifically.
3. Check whether the terms in force on **2026-09-05**, the acquisition date,
   differ from today's page. If they do, the acquisition-time terms govern.
4. Record tugstugi's id-list provenance and its own terms separately — it
   selects ids, it does not supply transcripts, but it is still a third-party
   artefact in the chain.
5. Only then decide the Phase-4 model licence.

## What this does NOT block

Phase 1 (`ehzawad/stt_bn_fastconformer`) contains **no MaCro** — its corpus is
fleurs_train, openslr53, subakko, kathbath, commonvoice, indicvoices. Its
relicensing is not gated by this item.
