#!/usr/bin/env python3
"""this recipe panel K: the official Kathbath Bengali *valid* split (AI4Bharat, CC-BY-4.0).

Mirrors the baseline converter (baseline converter) so the panel audio is produced by the same
transformation as the Kathbath training rows: 16 kHz mono PCM16 WAV, speaker_id/gender retained.
Text is transformed with the frozen surface (SPEC §1) so panel references and training targets share
one convention. This is an EVALUATION panel: nothing here may enter any training manifest
(build_corpus.py enforces the text-level exclusion; the audio fingerprint audit enforces the rest).
"""
import io
import json
import sys
import unicodedata
from math import gcd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402
from bnasr.manifest import atomic_manifest  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

RAW = REPO / "data4/kathbath/raw/bengali"
OUT = REPO / "data4/kathbath"
TARGET_SR = 16000


def main():
    (OUT / "wavs_valid").mkdir(parents=True, exist_ok=True)
    shards = sorted(RAW.glob("valid-*.parquet"))
    rows, idx, dropped = [], 0, {}
    for sh in shards:
        for rec in pq.read_table(sh).to_pylist():
            idx += 1
            a = rec.get("audio_filepath") or rec.get("audio") or {}
            raw = a.get("bytes")
            if not raw:
                dropped["no_audio"] = dropped.get("no_audio", 0) + 1; continue
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != TARGET_SR:
                g = gcd(sr, TARGET_SR)
                wav = resample_poly(wav, TARGET_SR // g, sr // g)
            text = " ".join(metric_surface(unicodedata.normalize(
                "NFC", (rec.get("text") or rec.get("transcription") or "").strip()), False).split())
            if not text:
                dropped["empty_text"] = dropped.get("empty_text", 0) + 1; continue
            p = OUT / "wavs_valid" / f"kathbath_valid_{idx:05d}.wav"
            tmp = p.with_suffix(".tmp")
            sf.write(str(tmp), wav, TARGET_SR, format="WAV", subtype="PCM_16")
            tmp.rename(p)
            rows.append({"audio_filepath": str(p.relative_to(REPO)), "text": text,
                         "duration": round(len(wav) / TARGET_SR, 3), "source": "kathbath_valid",
                         "speaker_id": rec.get("speaker_id"), "gender": rec.get("gender")})
        print(f"{sh.name}: cumulative {len(rows):,}", flush=True)
    with atomic_manifest(OUT / "manifest_valid.jsonl") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = sum(r["duration"] for r in rows) / 3600
    rep = {"rows": len(rows), "hours": round(h, 2), "speakers": len({r["speaker_id"] for r in rows}),
           "dropped": dropped}
    (OUT / "ingest_report_valid.json").write_text(json.dumps(rep, indent=1))
    print(rep)


if __name__ == "__main__":
    main()
