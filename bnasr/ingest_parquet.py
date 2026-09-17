#!/usr/bin/env python3
"""Phase-4 ingestion of Hugging Face parquet speech sets (SPEC §1 buckets ``spring_inx``, ``vaani``).

Generic over the HF audio-dataset convention: one row = {<id col>, <text col>, audio: {bytes, path}} plus
optional metadata columns (speaker, district, gender ...). Every split is written to its own manifest:
  train              -> the training bucket (source = --source)
  validation / test  -> PROTECTED panels (source = --source + "_valid" / "_test"), never trained on;
                        their sentences enter the protected set in build_corpus.py.
Audio bytes are decoded in-process with libsndfile, converted to 16 kHz mono PCM16 (resample_poly), and the
decoded duration is measured (never trusted from metadata). Registered new-row limits (0.5–23 s, 2–35 cps)
apply; every rejection is counted. BLAS threads are pinned to 1 per worker (see the shared-box I/O memory).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import io
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402

MIN_DUR, MAX_DUR = 0.5, 23.0
CPS_MIN, CPS_MAX = 2.0, 35.0
META_KEEP = ("speaker_id", "speaker", "gender", "district", "state", "age", "dialect", "language", "domain", "referenceImage")


import re
PARTIAL_MARKERS = re.compile(r"\[(unintelligible|inaudible|incomprehensible|unclear)\]", re.I)
ANGLE_TAG = re.compile(r"</?[A-Za-z_ ]+>")          # <noise> ... </noise>, <pause>, <talking>: span markers around real speech
SQUARE_EVENT = re.compile(r"\[[^\]]*\]")              # [breathing], [inhaling]: non-speech events
BRACE_GLOSS_SUFFIX = re.compile(r"\s*\{[^}]*\}\s*-(?=\S)")   # "পিস {Piece} -এর": gloss followed by a case suffix -> re-attach the suffix

# Re-attaching a suffix is not string concatenation. A Bengali suffix that is written
# with an INDEPENDENT vowel letter when it stands alone after the hyphen ("-এর", "-এ")
# must become the DEPENDENT vowel SIGN once it is joined to a consonant-final stem:
#     সাইড + -এ  -> সাইডে      (NOT সাইডএ)
#     ডিস্ট্রিক্ট + -এর -> ডিস্ট্রিক্টের  (NOT ডিস্ট্রিক্টএর)
# Defect found 2026-09-16: this rule used .sub("") and produced exactly the second form
# in 6,629 Vaani rows / 11.6 h — text no Bengali reader would write, ~100x enriched
# versus the rest of the corpus, and invisible to the script-purity gate because every
# character is Bengali. ও is deliberately NOT mapped: the emphatic particle after a
# consonant IS written with the independent letter (চলাচলও, বাসও are correct).
VOWEL_LETTER_TO_SIGN = {"আ": "\u09be", "ই": "\u09bf", "ঈ": "\u09c0", "উ": "\u09c1",
                        "ঊ": "\u09c2", "ঋ": "\u09c3", "এ": "\u09c7", "ঐ": "\u09c8",
                        "ঔ": "\u09cc"}
BENGALI_CONSONANT = re.compile(r"[\u0995-\u09b9\u09ce\u09dc-\u09df]$")


def _join_suffix(stem, suffix):
    """Attach a hyphen-marked suffix to the preceding stem with correct orthography."""
    if suffix and stem and BENGALI_CONSONANT.search(stem):
        sign = VOWEL_LETTER_TO_SIGN.get(suffix[0])
        if sign:
            return stem + sign + suffix[1:]
    return stem + suffix
BRACE_GLOSS = re.compile(r"\s*\{[^}]*\}")                   # "লেগ {leg}": {…} is an ENGLISH GLOSS of the preceding Bengali-script loanword, not speech -> delete


def clean_annotations(t, policy):
    """Vaani-style annotation handling (registered in SPEC §1 amendment 2026-09-05).

    Measured on 14,598 Bengali train rows (2026-09-05): 52 % carry <noise>…</noise>, 7 % <talking>…</talking>, 64 % a
    {brace}, 5.5 % [unintelligible]/[inaudible]. Inspection showed the spans wrap the PRIMARY speaker's words while the
    tag names the background condition (noise, other people talking, bird squawk) -- the words inside are what was said,
    so tags are deleted and words kept. A {brace} is an English GLOSS of the preceding Bengali-script loanword
    ("লেগ {leg} পিস {Piece} -এর"), NOT a spoken Latin word: keeping it would duplicate the word in Latin, so it is deleted
    and a following case suffix ("-এর") is re-attached to the loanword. Rows carrying a partial-transcript marker are
    DROPPED (partial labels teach deletion -- Phase-1 lesson with IndicVoices <unintelligible>). Non-speech events in
    [square brackets] are deleted. Returns None when the row must be dropped.
    """
    if policy == "none" or not t:
        return t
    if PARTIAL_MARKERS.search(t):
        return None
    t = ANGLE_TAG.sub(" ", t)
    t = SQUARE_EVENT.sub(" ", t)
    # join the suffix onto the stem, converting its leading independent vowel to a sign
    while True:
        m = BRACE_GLOSS_SUFFIX.search(t)
        if not m:
            break
        stem, suffix = t[:m.start()], t[m.end():]
        cut = len(suffix) - len(suffix.lstrip())
        t = _join_suffix(stem, suffix[cut:]) if not cut else stem + suffix
        if cut:
            break
    t = BRACE_GLOSS.sub(" ", t)
    return t


def surf(t):
    return " ".join(metric_surface(unicodedata.normalize("NFC", t or ""), False).split())


def _split_of(path):
    n = Path(path).name
    for s in ("train", "validation", "valid", "test", "dev"):
        if n.startswith(s + "-") or n.startswith(s + "."):
            return {"valid": "validation", "dev": "validation"}.get(s, s)
    return "train"


def do_shard(args):
    shard, out_dir, source, id_col, text_col, policy = args
    split = _split_of(shard)
    rows, rej = [], Counter()
    pf = pq.ParquetFile(shard)
    for rb in pf.iter_batches(batch_size=64):
        for rec in rb.to_pylist():
            a = rec.get("audio") or {}
            raw = a.get("bytes")
            uid = str((rec.get(id_col) if id_col != "none" else None) or Path(a.get("path") or "").stem or "")
            if not raw or not uid:
                rej["no_audio_or_id"] += 1; continue
            try:
                x, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:  # noqa: BLE001
                rej["undecodable"] += 1; continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = gcd(int(sr), 16000)
                x = resample_poly(x, 16000 // g, int(sr) // g)
            if x.size == 0 or not np.isfinite(x).all():
                rej["empty_or_nonfinite"] += 1; continue
            if np.max(np.abs(x)) < 8 / 32768:
                rej["silent"] += 1; continue
            dur = x.shape[0] / 16000.0
            if not (MIN_DUR <= dur <= MAX_DUR):
                rej["duration_out_of_range"] += 1; continue
            raw_t = clean_annotations(rec.get(text_col), policy)
            if raw_t is None:
                rej["partial_transcript_marker"] += 1; continue
            text = surf(raw_t)
            if not text:
                rej["empty_target"] += 1; continue
            cps = len(text.replace(" ", "")) / dur
            if not (CPS_MIN <= cps <= CPS_MAX):
                rej["cps_out_of_range"] += 1; continue
            # Names longer than the 120-char cut collided and one clip silently
            # OVERWROTE the other: 3 rows of the Vaani validation panel share an audio
            # file today, so three protected rows are guaranteed-wrong audio/text pairs
            # (found 2026-09-16). Keep the readable prefix, but make the name unique.
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in uid)
            if len(safe) > 120:
                safe = safe[:111] + "_" + hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8]
            wav = Path(out_dir) / f"wavs_{split}" / f"{safe}.wav"
            if wav.exists() and wav not in _written:
                raise SystemExit(f"ABORT: {wav} already exists and was not written by this run; "
                                 f"two utterances would share one file. uid={uid!r}")
            _written.add(wav)
            wav.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(wav) + ".tmp"
            sf.write(tmp, x, 16000, format="WAV", subtype="PCM_16")
            os.replace(tmp, wav)
            row = {"audio_filepath": os.path.relpath(wav, REPO), "text": text, "duration": round(dur, 3),
                   "source": source if split == "train" else f"{source}_{split}", "utt_id": uid, "split": split}
            for k in META_KEEP:
                if k in rec and rec[k] is not None and not isinstance(rec[k], (bytes, dict)):
                    row[k] = rec[k] if isinstance(rec[k], (str, int, float)) else str(rec[k])
            rows.append(row)
    return split, rows, dict(rej), str(shard)


_written = set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, help="directory containing the parquet shards (searched recursively)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", required=True, help="bucket tag, e.g. spring_inx_r1 or vaani")
    ap.add_argument("--id-col", default="utterance_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag-policy", choices=["none", "vaani"], default="none",
                    help="vaani: strip <span> tags and [events], DELETE {english gloss} braces, DROP rows with [unintelligible]/[inaudible]")
    a = ap.parse_args()
    shards = sorted(Path(a.parquet_dir).rglob("*.parquet"))
    if not shards:
        sys.exit("no parquet shards found")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    per_split, rej_all = {}, Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for split, rows, rej, shard in ex.map(do_shard, [(str(s), str(out), a.source, a.id_col, a.text_col, a.tag_policy) for s in shards]):
            per_split.setdefault(split, []).extend(rows); rej_all.update(rej)
            print(f"  {Path(shard).name}: {split} +{len(rows)} rows", flush=True)
    rep = {"source": a.source, "shards": len(shards), "tag_policy": a.tag_policy, "rejected": dict(rej_all), "splits": {}}
    for split, rows in per_split.items():
        p = out / f"manifest_{split}.jsonl"
        with open(str(p) + ".tmp", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(str(p) + ".tmp", p)
        rep["splits"][split] = {"rows": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
                                "distinct_texts": len({r["text"] for r in rows}), "manifest": os.path.relpath(p, REPO)}
    (out / "ingest_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
