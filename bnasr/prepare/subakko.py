#!/usr/bin/env python
"""SUBAK.KO (SUST CSE Speech group, CC-BY-4.0 per its dataset card) -> data2/subakko/.

Hub dataset `SUST-CSE-Speech/SUBAK.KO` (not gated), parquet columns
{audio: {bytes, path}, transcription, file_path}. The TRAIN split is converted to
16 kHz mono PCM16 wav in stream order; the 1-based stream index is the stable key
(subakko_<index>.wav). The speaker key is recovered from `file_path`: the NAME token
in TN*_NAME_... file names (the TN codes are unreliable — the same speaker appears
as TNM03/TNM04/TNMO3), with SAM and SAMI merged (same code, same directory), and one
key per source directory for the L2_speakers YouTube recordings.

    python bnasr/prepare/subakko.py [--fetch] [--raw data2/raw/subakko] [--limit N]
"""
import argparse
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, convert_bytes, duration_ok, hf_snapshot, log, rel, summary, write_rows

NAME = re.compile(r"^TN[A-Za-z0-9]*?_+([A-Za-z]{2,})")
ALIAS = {"SAM": "SAMI"}


def speaker_from_path(p):
    parts = p.split("/")
    if len(parts) > 4 and parts[2] == "L2_speakers":
        return "L2:" + parts[4]
    m = NAME.match(os.path.basename(p))
    if not m:
        return None
    n = m.group(1).upper()
    return "TN:" + ALIAS.get(n, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data2/raw/subakko"))
    ap.add_argument("--out", default=str(REPO / "data2/subakko"))
    ap.add_argument("--fetch", action="store_true", help="download data/train-*.parquet from the Hub first")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    if a.fetch:
        hf_snapshot("SUST-CSE-Speech/SUBAK.KO", ["data/train-*.parquet"], raw)
    shards = sorted(raw.glob("**/train-*.parquet"))
    if not shards:
        raise SystemExit(f"no train-*.parquet under {raw}; use --fetch")
    log(f"subakko: {len(shards)} shards")
    rows, idx = [], 0
    for sh in shards:
        for rec in pq.read_table(sh).to_pylist():
            idx += 1
            if a.limit and idx > a.limit:
                break
            raw_bytes = (rec.get("audio") or {}).get("bytes")
            text = (rec.get("transcription") or rec.get("text") or "").strip()
            if not raw_bytes or not text:
                continue
            dst = out / "wavs" / f"subakko_{idx:07d}.wav"
            dur = convert_bytes(raw_bytes, dst)
            if not duration_ok(dur):
                continue
            fp = rec.get("file_path") or ""
            rows.append({"audio_filepath": rel(dst), "text": text, "duration": dur, "source": "subakko",
                         "speaker_id": speaker_from_path(fp), "file_path": fp, "key": f"{idx:07d}"})
        log(f"subakko: {sh.name} -> {len(rows):,} rows")
        if a.limit and idx > a.limit:
            break
    write_rows(out / "manifest.jsonl", rows)
    summary("subakko", rows)


if __name__ == "__main__":
    main()
