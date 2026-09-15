#!/usr/bin/env python
"""OpenSLR-53 (Large Bengali ASR training data set, CC-BY-SA-4.0) -> data2/openslr53/.

Downloads the 16 zip shards (asr_bengali_0.zip ... asr_bengali_f.zip) with resume,
extracts them, joins utt_spk_text.tsv for the transcript and the speaker id, and
converts every FLAC to 16 kHz mono PCM16 wav named openslr_<utt_id>.wav. The utt id
is the stable key.

    python bnasr/prepare/openslr53.py [--raw data2/raw/openslr53] [--shards 0123456789abcdef] [--limit N]
"""
import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import (REPO, convert_file, download_with_resume, duration_ok, log, rel,
                                   summary, write_rows)

BASE_URL = "https://openslr.trmal.net/resources/53"
SHARDS = "0123456789abcdef"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data2/raw/openslr53"))
    ap.add_argument("--out", default=str(REPO / "data2/openslr53"))
    ap.add_argument("--shards", default=SHARDS, help="hex digits of the shards to fetch (all 16 by default)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke)")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    if not a.skip_download:
        for s in a.shards:
            z = download_with_resume(f"{BASE_URL}/asr_bengali_{s}.zip", raw / f"asr_bengali_{s}.zip")
            marker = raw / f".extracted_{s}"
            if not marker.exists():
                log(f"extracting {z.name}")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(raw)
                marker.touch()
    corpus = raw / "asr_bengali"
    tsv = corpus / "utt_spk_text.tsv"
    if not tsv.exists():
        raise SystemExit(f"{tsv} not found; download/extract first")
    meta = {}
    for line in open(tsv, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t", 2)
        if len(parts) == 3:
            meta[parts[0]] = (parts[1], parts[2].strip())
    flacs = sorted(corpus.rglob("*.flac"))
    if a.limit:
        flacs = flacs[:a.limit]
    log(f"openslr53: {len(flacs):,} flac files, {len(meta):,} transcripts")
    jobs = [(f, out / "wavs" / f"openslr_{f.stem}.wav") for f in flacs if f.stem in meta and meta[f.stem][1]]
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for (src, dst), dur in zip(jobs, pool.map(lambda j: convert_file(*j), jobs)):
            if not duration_ok(dur):
                continue
            spk, text = meta[src.stem]
            rows.append({"audio_filepath": rel(dst), "text": text, "duration": dur, "source": "openslr53",
                         "speaker_id": spk, "key": src.stem})
    write_rows(out / "manifest.jsonl", rows)
    summary("openslr53", rows)


if __name__ == "__main__":
    main()
