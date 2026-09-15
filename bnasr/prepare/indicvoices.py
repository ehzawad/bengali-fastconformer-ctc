#!/usr/bin/env python
"""IndicVoices Bengali (AI4Bharat, CC-BY-4.0) -> data2/indicvoices/.

Hub dataset `ai4bharat/IndicVoices` (gated). Keeps only rows whose per-utterance
human verification decision is "excellent", whose text carries no annotation tag
(a tagged row is partially labelled audio — CTC would learn to delete speech) and
whose text is Bengali script. Audio is 16 kHz mono FLAC in the parquet and is
written verbatim, then re-opened and its frame count checked against the shard's
own `samples` column: a truncated write must be a skip, never a keep.

The running record index over the sorted shards is the stable key
(indicvoices_<index>.flac). speaker_id, scenario, gender, state, district are kept.

    python bnasr/prepare/indicvoices.py [--fetch] [--raw data2/raw/indicvoices] [--limit N]
"""
import argparse
import ast
import re
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, TARGET_SR, duration_ok, hf_snapshot, log, rel, summary, write_rows

TAG_RE = re.compile(r"<[^>]{1,40}>")
BENGALI_RE = re.compile(r"[ঀ-৿]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data2/raw/indicvoices"))
    ap.add_argument("--out", default=str(REPO / "data2/indicvoices"))
    ap.add_argument("--fetch", action="store_true", help="download bengali/train-*.parquet from the Hub first")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    if a.fetch:
        hf_snapshot("ai4bharat/IndicVoices", ["bengali/train-*.parquet"], raw)
    shards = sorted(raw.glob("**/bengali/train-*.parquet")) or sorted(raw.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no bengali/train-*.parquet under {raw}; use --fetch")
    log(f"indicvoices: {len(shards)} shards")
    (out / "flac").mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    skipped = {"lang": 0, "not_excellent": 0, "empty": 0, "tag": 0, "script": 0, "no_bytes": 0, "bad_audio": 0}
    for sh in shards:
        for rec in pq.read_table(sh).to_pylist():
            idx += 1
            if a.limit and idx > a.limit:
                break
            if rec.get("lang") not in (None, "bn"):
                skipped["lang"] += 1; continue
            try:
                decision = ast.literal_eval(rec.get("verification_report") or "{}").get("decision")
            except (ValueError, SyntaxError):
                decision = None
            if decision != "excellent":
                skipped["not_excellent"] += 1; continue
            text = (rec.get("text") or "").strip()
            if not text:
                skipped["empty"] += 1; continue
            if TAG_RE.search(text):
                skipped["tag"] += 1; continue
            if not BENGALI_RE.search(text):
                skipped["script"] += 1; continue
            raw_bytes = (rec.get("audio_filepath") or rec.get("audio") or {}).get("bytes")
            if not raw_bytes:
                skipped["no_bytes"] += 1; continue
            dst = out / "flac" / f"indicvoices_{idx:07d}.flac"
            if not dst.exists():
                tmp = dst.with_suffix(".tmp"); tmp.write_bytes(raw_bytes)
                try:
                    info = sf.info(str(tmp))
                    ok = (info.samplerate == TARGET_SR and info.channels == 1 and info.frames > 0
                          and (rec.get("samples") is None or info.frames == rec["samples"]))
                except Exception:
                    ok = False
                if not ok:
                    tmp.unlink(missing_ok=True); skipped["bad_audio"] += 1; continue
                tmp.rename(dst)
            info = sf.info(str(dst)); dur = round(info.frames / TARGET_SR, 3)
            if not duration_ok(dur):
                continue
            rows.append({"audio_filepath": rel(dst), "text": text, "duration": dur, "source": "indicvoices",
                         "speaker_id": rec.get("speaker_id"), "scenario": rec.get("scenario"), "gender": rec.get("gender"),
                         "state": rec.get("state"), "district": rec.get("district"), "key": f"{idx:07d}"})
        log(f"indicvoices: {sh.name} -> {len(rows):,} rows")
        if a.limit and idx > a.limit:
            break
    write_rows(out / "manifest.jsonl", rows)
    summary("indicvoices", rows, skipped)


if __name__ == "__main__":
    main()
