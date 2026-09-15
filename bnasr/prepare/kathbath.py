#!/usr/bin/env python
"""Kathbath Bengali (AI4Bharat IndicSUPERB, CC-BY-4.0) -> data2/kathbath/.

Hub dataset `ai4bharat/Kathbath` (gated: accept the licence on the Hub page and log in
with `huggingface-cli login` or HF_TOKEN). The Bengali TRAIN shards are converted to
16 kHz mono PCM16 wav; the running record index over the sorted shards is the stable
key (kathbath_<index>.wav). speaker_id and gender are kept for speaker-disjoint carving.
The validation split is the development panel K and is prepared separately by
bnasr/ingest_kathbath_valid.py; the official test split is never trained on.

    python bnasr/prepare/kathbath.py [--fetch] [--raw data2/raw/kathbath] [--limit N]
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, convert_bytes, duration_ok, hf_snapshot, log, rel, summary, write_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data2/raw/kathbath"))
    ap.add_argument("--out", default=str(REPO / "data2/kathbath"))
    ap.add_argument("--fetch", action="store_true", help="download bengali/train-*.parquet from the Hub first")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    if a.fetch:
        hf_snapshot("ai4bharat/Kathbath", ["bengali/train-*.parquet"], raw)
    shards = sorted(raw.glob("**/bengali/train-*.parquet")) or sorted(raw.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no bengali/train-*.parquet under {raw}; use --fetch")
    log(f"kathbath: {len(shards)} shards")
    rows, idx = [], 0
    for sh in shards:
        for rec in pq.read_table(sh).to_pylist():
            idx += 1
            if a.limit and idx > a.limit:
                break
            raw_bytes = ((rec.get("audio_filepath") or rec.get("audio") or {}).get("bytes"))
            text = (rec.get("text") or rec.get("transcription") or "").strip()
            if not raw_bytes or not text:
                continue
            dst = out / "wavs" / f"kathbath_{idx:07d}.wav"
            dur = convert_bytes(raw_bytes, dst)
            if not duration_ok(dur):
                continue
            rows.append({"audio_filepath": rel(dst), "text": text, "duration": dur, "source": "kathbath",
                         "speaker_id": rec.get("speaker_id"), "gender": rec.get("gender"), "key": f"{idx:07d}"})
        log(f"kathbath: {sh.name} -> {len(rows):,} rows")
        if a.limit and idx > a.limit:
            break
    write_rows(out / "manifest.jsonl", rows)
    summary("kathbath", rows)


if __name__ == "__main__":
    main()
