#!/usr/bin/env python3
"""this recipe ingestion of the Bengali.AI MaCro competition audio (SPEC §1).

What this does and why each step exists
---------------------------------------
* Membership comes from tugstugi's CSVs (ids only): ``macro_filtered`` = competition ``train`` rows he kept
  after WER<15 filtering; ``macro_validated`` = competition ``valid`` rows. Any id not in the competition
  ``train.csv`` is an error, not a skip -- the membership lists must be fully explained (council B #4).
* The TEXT is the competition's own human reference (``train.csv`` sentence), never his cleaned text
  (council A Q1e / B #11), transformed with the frozen surface ``metric_surface(NFC(t), False)``.
* Audio: 32 kHz mono 48 kbps MP3 -> 16 kHz mono PCM16 WAV via ffmpeg (deterministic, ``-ar 16000 -ac 1``).
  The decoded duration is measured from the WAV header (not trusted from any CSV) and compared with
  tugstugi's duration column as a provenance check (|diff| > 0.25 s is flagged and the row quarantined).
* Rejections are counted per reason and written next to the manifest (SPEC §1 publishes them).

Output: ``data4/macro/manifest_<bucket>.jsonl`` with NeMo fields plus ``source``, ``macro_id``, ``split``.
Row-level filters that need the tokenizer/model (feasibility, dedup, carving) live in ``build_corpus.py``.
"""
import argparse
import array
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # BLAS threads x 12 workers drove the shared box to load 1,200; one thread each
import csv
import json
import os
import subprocess
import sys
import unicodedata
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402

RAW = REPO / "data4/macro/raw"
OUT_WAV = REPO / "data4/macro/wavs"
CSV = REPO / "data4/tugstugi/csv"
MIN_DUR, MAX_DUR = 0.5, 23.0          # SPEC §1 registered duration limits for new rows
CPS_MIN, CPS_MAX = 2.0, 35.0          # SPEC §1: NFC non-space code points per second


def target_text(raw):
    return " ".join(metric_surface(unicodedata.normalize("NFC", raw), False).split())


def convert_one(args):
    """mp3 -> 16 kHz mono PCM16 wav; returns (id, wav_path, duration_s, err).

    Decoding is done IN-PROCESS with libsndfile (1.2.x reads MP3) + scipy resample_poly (32 kHz -> 16 kHz is an
    exact /2), not by spawning ffmpeg: the ffmpeg-per-file path drove the shared box to a load average >200
    (each ffmpeg spawns threads) at ~13 files/s; libsndfile decodes ~50 files/s per process single-threaded.
    ffmpeg (-threads 1) remains the fallback for files libsndfile rejects.
    """
    mid, src, dst = args
    if not os.path.exists(dst):
        tmp = dst + ".tmp.wav"
        try:
            x, sr = sf.read(src, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = gcd(sr, 16000)
                x = resample_poly(x, 16000 // g, sr // g)
            sf.write(tmp, x, 16000, format="WAV", subtype="PCM_16")
        except Exception:  # noqa: BLE001 - fall back to ffmpeg for odd encodings
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "1", "-i", src, "-ar", "16000",
                                "-ac", "1", "-sample_fmt", "s16", "-f", "wav", tmp], capture_output=True, text=True)
            if r.returncode != 0:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return mid, dst, 0.0, f"decode:{r.stderr.strip()[:120]}"
        os.replace(tmp, dst)
    try:
        with wave.open(dst) as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                return mid, dst, 0.0, "bad_wav_format"
            n = w.getnframes()
            if n == 0:
                return mid, dst, 0.0, "empty_audio"
            data = w.readframes(n)
        a = array.array("h", data)
        if max(abs(v) for v in a) < 8:            # < 8/32768: effectively all-zero
            return mid, dst, n / 16000.0, "silent_audio"
        return mid, dst, n / 16000.0, None
    except Exception as e:  # noqa: BLE001
        return mid, dst, 0.0, f"wav_read:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", choices=["macro_filtered", "macro_validated"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N ids")
    a = ap.parse_args()

    comp = {}
    with open(RAW / "train.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            comp[r["id"]] = (r["sentence"], r["split"])
    if a.bucket == "macro_filtered":
        want_split = "train"
        members = {}
        with open(CSV / "bengali-macro-madasr-pseudo1257k-wer15.csv", encoding="utf-8", newline="") as fh:
            for r in csv.reader(fh):
                if r[0].startswith("macro/"):
                    members[r[0].split("/")[1][:-4]] = float(r[1])
    else:
        want_split = "valid"
        members = {}
        with open(CSV / "bengali-macro-validated.csv", encoding="utf-8", newline="") as fh:
            for r in csv.reader(fh):
                members[r[0].split("/")[1][:-4]] = float(r[1])
    ids = sorted(members)
    if a.limit:
        ids = ids[: a.limit]
    OUT_WAV.mkdir(parents=True, exist_ok=True)

    rej = {}
    def reject(k):
        rej[k] = rej.get(k, 0) + 1

    jobs = []
    for mid in ids:
        if mid not in comp:
            reject("id_not_in_competition_csv"); continue
        if comp[mid][1] != want_split:
            reject(f"wrong_split:{comp[mid][1]}"); continue
        src = RAW / "train_mp3s" / f"{mid}.mp3"
        if not src.exists():
            reject("mp3_missing"); continue
        jobs.append((mid, str(src), str(OUT_WAV / f"{mid}.wav")))

    out_path = REPO / f"data4/macro/manifest_{a.bucket}.jsonl"
    tmp_path = str(out_path) + ".tmp"
    kept = 0; hours = 0.0
    with open(tmp_path, "w", encoding="utf-8") as out, ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            mid, wav, dur, err = f.result()
            if err:
                reject(err.split(":")[0]); continue
            if abs(dur - members[mid]) > 0.25:
                reject("duration_mismatch_vs_tugstugi_csv"); continue
            if not (MIN_DUR <= dur <= MAX_DUR):
                reject("duration_out_of_range"); continue
            text = target_text(comp[mid][0])
            if not text:
                reject("empty_target"); continue
            cps = len(text.replace(" ", "")) / dur
            if not (CPS_MIN <= cps <= CPS_MAX):
                reject("cps_out_of_range"); continue
            row = {"audio_filepath": os.path.relpath(wav, REPO), "text": text, "duration": round(dur, 3),
                   "source": a.bucket, "macro_id": mid, "split": want_split}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1; hours += dur / 3600
            if i % 20000 == 0:
                print(f"  {i}/{len(jobs)} converted, kept {kept}", flush=True)
    os.replace(tmp_path, out_path)
    report = {"bucket": a.bucket, "member_ids": len(ids), "kept_rows": kept, "kept_hours": round(hours, 2),
              "rejected": rej, "limits": {"dur": [MIN_DUR, MAX_DUR], "cps": [CPS_MIN, CPS_MAX]},
              "text_rule": "metric_surface(NFC(train.csv sentence), score_punctuation=False), whitespace collapsed"}
    (REPO / f"data4/macro/ingest_report_{a.bucket}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
