#!/usr/bin/env python
"""Common Voice Bengali (Mozilla, CC0) -> data2/commonvoice/.

Common Voice must be downloaded by hand from Mozilla Data Collective
(https://datacollective.mozilla.org, the "Common Voice Scripted Speech" release for
`bn`) — the licence acceptance is per user and cannot be scripted here. Unpack it
and point --raw at the directory holding validated.tsv, test.tsv, dev.tsv and clips/.

Keeps ONLY rows of validated.tsv and EXCLUDES the clips in the release's own test.tsv
and dev.tsv: they are the public benchmark split, and training on them would
contaminate every future Common Voice comparison. client_id (16 chars) is the
speaker key. Files are numbered in validated.tsv order (commonvoice_<index>.wav);
the index is the stable key, so keep the same release to reproduce the released lists.

    python bnasr/prepare/commonvoice.py --raw data2/raw/commonvoice/cv-corpus-26.0-2026-06-12/bn [--limit N]
"""
import argparse
import csv
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, convert_file, duration_ok, log, rel, summary, write_rows


def tsv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="the bn/ directory of the Common Voice release")
    ap.add_argument("--out", default=str(REPO / "data2/commonvoice"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.raw), Path(a.out)
    held = {r["path"] for r in tsv(root / "test.tsv")} | {r["path"] for r in tsv(root / "dev.tsv")}
    rows_in = tsv(root / "validated.tsv")
    log(f"commonvoice: validated {len(rows_in):,}, held out (test+dev) {len(held):,}")
    rows, idx, skipped = [], 0, 0
    for r in rows_in:
        if r["path"] in held:
            continue
        idx += 1
        if a.limit and idx > a.limit:
            break
        src = root / "clips" / r["path"]
        text = (r.get("sentence") or "").strip()
        if not src.exists() or not text:
            skipped += 1; continue
        try:
            dur = convert_file(src, out / "wavs" / f"commonvoice_{idx:07d}.wav")
        except Exception:
            skipped += 1; continue
        if not duration_ok(dur):
            continue
        rows.append({"audio_filepath": rel(out / "wavs" / f"commonvoice_{idx:07d}.wav"), "text": text, "duration": dur,
                     "source": "commonvoice", "speaker_id": (r.get("client_id") or "")[:16], "key": f"{idx:07d}",
                     "clip": r["path"]})
        if idx % 5000 == 0:
            log(f"commonvoice: {idx:,}")
    write_rows(out / "manifest.jsonl", rows)
    summary("commonvoice", rows, skipped)


if __name__ == "__main__":
    main()
