#!/usr/bin/env python3
"""this recipe ingestion of an IISc RESPIN Bengali set (SPEC §1: bucket ``madasr``; panel R = official dev).

Layout of a RESPIN set (verified on IISc_RESPIN_dev_bn): <set>/<dialect D1..D5>/<speaker>/<utt>.wav, one
<speaker>.txt per speaker ("utt_id<TAB>text"), and meta_<set>.json keyed by utt id with text, duration,
speaker_id, dialect, domain, gender, slab. Audio is already 16 kHz mono PCM16.

Why the metadata JSON is the source of truth: it carries the human transcript and the speaker/dialect ids
needed for disjointness checks; the per-speaker txt is cross-checked (any disagreement is a rejection).
tugstugi's 2023 MADASR id list does NOT map onto this 2025 release (7/1,183 dev texts match), so no
upstream filter is applied here: the bucket is IISc's own "clean" slab, human-read prompts, whole.
Registered new-row limits (duration 0.5–23 s, cps 2–35) are applied; wav headers are verified.
"""
import argparse
import json
import os
import sys
import unicodedata
import wave
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402

MIN_DUR, MAX_DUR = 0.5, 23.0
CPS_MIN, CPS_MAX = 2.0, 35.0


def surf(t):
    return " ".join(metric_surface(unicodedata.normalize("NFC", t), False).split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True, help="e.g. data4/respin/dev/IISc_RESPIN_dev_bn")
    ap.add_argument("--source", required=True, help="manifest source tag: madasr_dev | madasr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = Path(a.set_dir)
    meta_files = list(sd.glob("meta_*.json"))
    assert len(meta_files) == 1, meta_files
    meta = json.load(open(meta_files[0], encoding="utf-8"))

    # per-speaker txt cross-check
    txt = {}
    for t in sd.rglob("*.txt"):
        for line in open(t, encoding="utf-8"):
            if "\t" in line:
                k, v = line.rstrip("\n").split("\t", 1)
                txt[k] = v
    rej = Counter(); rows = []; hours = 0.0
    for utt, m in meta.items():
        wav = sd / m["dialect"] / str(m["speaker_id"]) / f"{utt}.wav"
        if not wav.exists():
            rej["wav_missing"] += 1; continue
        if utt in txt and " ".join(txt[utt].split()) != " ".join(m["text"].split()):
            rej["txt_meta_disagree"] += 1; continue
        try:
            with wave.open(str(wav)) as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                    rej["bad_wav_format"] += 1; continue
                dur = w.getnframes() / 16000.0
        except Exception:  # noqa: BLE001
            rej["wav_unreadable"] += 1; continue
        if abs(dur - float(m["duration"])) > 0.05:
            rej["duration_mismatch_meta"] += 1; continue
        if not (MIN_DUR <= dur <= MAX_DUR):
            rej["duration_out_of_range"] += 1; continue
        text = surf(m["text"])
        if not text:
            rej["empty_target"] += 1; continue
        cps = len(text.replace(" ", "")) / dur
        if not (CPS_MIN <= cps <= CPS_MAX):
            rej["cps_out_of_range"] += 1; continue
        rows.append({"audio_filepath": os.path.relpath(wav, REPO), "text": text, "duration": round(dur, 3),
                     "source": a.source, "utt_id": utt, "speaker_id": str(m["speaker_id"]),
                     "dialect": m["dialect"], "domain": m.get("domain"), "gender": m.get("gender"),
                     "slab": m.get("slab")})
        hours += dur / 3600
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    rep = {"set_dir": str(sd), "source": a.source, "meta_rows": len(meta), "kept_rows": len(rows),
           "kept_hours": round(hours, 2), "speakers": len({r["speaker_id"] for r in rows}),
           "dialects": dict(Counter(r["dialect"] for r in rows)), "slabs": dict(Counter(r["slab"] for r in rows)),
           "distinct_texts": len({r["text"] for r in rows}), "rejected": dict(rej)}
    Path(str(out).replace(".jsonl", "_report.json")).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
