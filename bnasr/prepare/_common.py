"""Shared plumbing for the per-source preparation tools.

Every source ends up in the same layout:

    data2/<source>/wavs/<source>_<key>.wav     16 kHz mono PCM16 (IndicVoices: FLAC, verbatim)
    data2/<source>/manifest.jsonl              one row per utterance:
        {"audio_filepath": "data2/<source>/wavs/...", "text": ..., "duration": ...,
         "source": "<source>", "speaker_id": ..., "key": "<stable per-source key>"}

`key` is what the shipped panel/holdout/train lists refer to (data/panels/), so a
re-prepared corpus can be checked against the released one row for row.
"""
import hashlib
import json
import os
import sys
import unicodedata
from contextlib import contextmanager
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[2]
TARGET_SR = 16000
MIN_DUR, MAX_DUR = 0.3, 30.0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def rel(p):
    """Repo-relative path string for manifests (what every tool expects)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def to_mono_16k(wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g)
    return wav.astype(np.float32)


def write_wav(dst, wav):
    """Atomic PCM16 write; returns duration in seconds."""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    sf.write(str(tmp), wav, TARGET_SR, format="WAV", subtype="PCM_16")
    os.replace(tmp, dst)
    return round(len(wav) / TARGET_SR, 3)


def convert_file(src, dst):
    """Any libsndfile-readable file (wav/flac/mp3/ogg) -> 16 kHz mono PCM16 wav."""
    if Path(dst).exists():
        info = sf.info(str(dst))
        return round(info.frames / info.samplerate, 3)
    wav, sr = sf.read(str(src), dtype="float32")
    return write_wav(dst, to_mono_16k(wav, sr))


def convert_bytes(raw, dst):
    import io
    if Path(dst).exists():
        info = sf.info(str(dst))
        return round(info.frames / info.samplerate, 3)
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return write_wav(dst, to_mono_16k(wav, sr))


@contextmanager
def atomic_manifest(path):
    """Write the manifest to a sibling temp file and rename on success, so a
    crash never leaves a half-written manifest that looks complete."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yield fh
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_rows(path, rows):
    with atomic_manifest(path) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def duration_ok(d):
    return MIN_DUR <= d <= MAX_DUR


def text_sha(text):
    return hashlib.sha1(unicodedata.normalize("NFC", " ".join(text.split())).encode("utf-8")).hexdigest()[:16]


def summary(source, rows, skipped=None):
    h = sum(r["duration"] for r in rows) / 3600
    spk = len({r.get("speaker_id") for r in rows if r.get("speaker_id")})
    log(f"{source}: {len(rows):,} rows / {h:.1f} h / {spk} speakers" + (f" / skipped {skipped}" if skipped else ""))


def download_with_resume(url, dest, chunk=1 << 20):
    """HTTP download with Range resume (curl -C - semantics), returns dest."""
    import requests
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.with_suffix(dest.suffix + ".done")
    if done.exists():
        return dest
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:          # already complete
            done.touch(); return dest
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(dest, mode) as fh:
            for part in r.iter_content(chunk_size=chunk):
                fh.write(part)
    done.touch()
    return dest


def hf_snapshot(repo_id, patterns, dest):
    """Download matching files of a Hub dataset (gated datasets need HF_TOKEN
    or `huggingface-cli login` and an accepted licence on the Hub page)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns,
                                  local_dir=str(dest), local_dir_use_symlinks=False))
