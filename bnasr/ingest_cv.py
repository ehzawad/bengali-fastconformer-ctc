#!/usr/bin/env python3
"""this recipe ingestion of Common Voice 17.0 Bengali (bucket ``cv17``), owner-authorised 2026-09-07.

Why this bucket: the dev-panel error analysis showed 79 % of the model's errors come from IndicVoices (conversational,
45 %) and SUBAK.KO (formal/literary, 34 % at 36.0 % WER), and that SUBAK.KO's failure is lexical, not length, numerals,
word-joining convention or truncated references (median hypothesis/reference length ratio 1.000). The registered headline
(FLEURS) is read encyclopedic prose. Common Voice's Bengali prompts are largely Wikipedia-derived, i.e. the register we
are weakest in and are measured on — unlike MaCro (everyday prompts), RESPIN (agri/banking prompts) or Vaani (image
descriptions).

Leakage measured on the transcripts BEFORE downloading audio (`data4/cv17/leakage_text.json`):
  FLEURS test 0 sentences, FLEURS validation 0  -> the headline is unaffected.
  D 495 / holdout 463 / M 1,825 sentences appear in CV (Bengali.AI ran the Common Voice Bangla campaign, so MaCro and CV
  share a prompt pool). Those rows are removed by the builder's protected-sentence rule exactly as MaCro's were, which is
  what keeps panel M valid; nothing here changes a panel.

Splits: ``validated`` (73.7 h, 2+ listeners confirmed the recording matches the prompt) and ``other`` (1,181.4 h,
not yet confirmed — the quality filter in `p4_cv_filter.py` is what makes it admissible). ``invalidated`` (community
REJECTED the recording) is never ingested. dev/test/train are subsets of validated and are ingested through it.

Audio: CV mp3 -> 16 kHz mono PCM16 WAV, decoded in-process with libsndfile (see the shared-box I/O memory: never an
ffmpeg swarm). Registered new-row limits apply: duration 0.5-23.0 s, 2-35 NFC code points/second, non-empty frozen-surface
target, decodable and non-silent. Every rejection is counted.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import io
import json
import sys
import tarfile
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402

CV = REPO / "data4/cv17"
OUT_WAV = CV / "wavs"
MIN_DUR, MAX_DUR = 0.5, 23.0
CPS_MIN, CPS_MAX = 2.0, 35.0
csv.field_size_limit(10 ** 9)


def surf(t):
    return " ".join(metric_surface(unicodedata.normalize("NFC", t or ""), False).split())


def do_tar(args):
    """Extract one CV tar, convert every clip whose id is wanted, return (rows, rejections)."""
    tar_path, split, wanted_json, out_dir = args
    wanted = json.loads(Path(wanted_json).read_text())          # {clip filename: [sentence, duration_s]}
    rows, rej = [], Counter()
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            meta = wanted.get(name)
            if meta is None:
                rej["not_in_split_tsv"] += 1
                continue
            text = surf(meta[0])
            if not text:
                rej["empty_target"] += 1
                continue
            try:
                raw = tf.extractfile(m).read()
                x, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:  # noqa: BLE001
                rej["undecodable"] += 1
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = gcd(int(sr), 16000)
                x = resample_poly(x, 16000 // g, int(sr) // g)
            if x.size == 0 or not np.isfinite(x).all():
                rej["empty_or_nonfinite"] += 1
                continue
            if float(np.max(np.abs(x))) < 8 / 32768:
                rej["silent"] += 1
                continue
            dur = x.shape[0] / 16000.0
            if not (MIN_DUR <= dur <= MAX_DUR):
                rej["duration_out_of_range"] += 1
                continue
            cps = len(text.replace(" ", "")) / dur
            if not (CPS_MIN <= cps <= CPS_MAX):
                rej["cps_out_of_range"] += 1
                continue
            dst = Path(out_dir) / f"{Path(name).stem}.wav"
            tmp = str(dst) + ".tmp"
            sf.write(tmp, (np.clip(x, -1, 1) * 32767).astype("<i2"), 16000, format="WAV", subtype="PCM_16")
            os.replace(tmp, dst)
            rows.append({"audio_filepath": os.path.relpath(dst, REPO), "text": text, "duration": round(dur, 3),
                         "source": f"cv17_{split}", "clip": name})
    return rows, dict(rej), os.path.basename(tar_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test", "other"], required=True,
                    help="CV official splits. `train` and filtered `other` are the training bucket; `dev`/`test` are ingested\n                         ONLY as protected one-shot panels and are never trained on (training on a published test\n                         split would make any CV-test comparison meaningless).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-tars", type=int, default=0)
    a = ap.parse_args()

    dur = {}
    with open(CV / "transcript/bn/clip_durations.tsv", encoding="utf-8") as fh:
        for r in csv.reader(fh, delimiter="\t"):
            if len(r) >= 2 and r[1].isdigit():
                dur[r[0]] = int(r[1]) / 1000.0
    wanted = {}
    with open(CV / f"transcript/bn/{a.split}.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            p, s = r.get("path"), r.get("sentence")
            if p and s:
                wanted[p] = [s, dur.get(p, 0.0)]
    wj = CV / f"_wanted_{a.split}.json"
    wj.write_text(json.dumps(wanted))
    out_dir = OUT_WAV / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tars = sorted((CV / "audio/bn" / a.split).glob("*.tar"))
    if a.limit_tars:
        tars = tars[: a.limit_tars]
    print(f"{a.split}: {len(wanted)} clips wanted, {len(tars)} tars", flush=True)

    t0 = time.time(); all_rows = []; rej = Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for rows, r, name in ex.map(do_tar, [(str(t), a.split, str(wj), str(out_dir)) for t in tars]):
            all_rows.extend(rows); rej.update(r)
            print(f"  {name}: +{len(rows)} rows (total {len(all_rows)}, {time.time()-t0:.0f}s)", flush=True)
    man = CV / f"manifest_{a.split}.jsonl"
    with open(str(man) + ".tmp", "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(str(man) + ".tmp", man)
    rep = {"split": a.split, "tars": len(tars), "wanted": len(wanted), "kept_rows": len(all_rows),
           "kept_hours": round(sum(r["duration"] for r in all_rows) / 3600, 2),
           "distinct_texts": len({r["text"] for r in all_rows}), "rejected": dict(rej),
           "limits": {"dur": [MIN_DUR, MAX_DUR], "cps": [CPS_MIN, CPS_MAX]}, "wall_s": round(time.time() - t0)}
    (CV / f"ingest_report_{a.split}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
