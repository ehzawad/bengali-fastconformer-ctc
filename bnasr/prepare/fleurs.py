#!/usr/bin/env python
"""FLEURS bn_in (Google, CC-BY-4.0) -> data2/fleurs/.

Takes the Hub dataset `google/fleurs`, config `bn_in`, and writes the requested
splits as fleurs_train_<index>.wav with the transcript. The official TEST split is
never written here: it is the benchmark set and is only ever read by the sealed
test procedure (bnasr/final_test.py). By default train AND validation are folded
into the training pool, which is what the released model was trained on — and
which is why FLEURS validation is not a clean evaluation set for that model.
Pass `--splits train` to keep validation out.

    python bnasr/prepare/fleurs.py [--splits train validation] [--limit N]
"""
import argparse
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, convert_bytes, duration_ok, log, rel, summary, write_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data2/fleurs"))
    ap.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from datasets import Audio, concatenate_datasets, load_dataset
    parts = [load_dataset("google/fleurs", "bn_in", split=s) for s in a.splits]
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    ds = ds.cast_column("audio", Audio(decode=False))
    out = Path(a.out); rows = []
    for i, ex in enumerate(ds):
        if a.limit and i >= a.limit:
            break
        text = (ex.get("transcription") or "").strip()
        if not text:
            continue
        dst = out / "wavs" / f"fleurs_train_{i:06d}.wav"
        dur = convert_bytes(ex["audio"]["bytes"], dst)
        if not duration_ok(dur):
            continue
        rows.append({"audio_filepath": rel(dst), "text": text, "duration": dur, "source": "fleurs_train",
                     "speaker_id": None, "key": f"{i:06d}", "fleurs_id": ex.get("id")})
        if (i + 1) % 500 == 0:
            log(f"fleurs: {i+1:,}")
    write_rows(out / "manifest.jsonl", rows)
    summary("fleurs_train", rows)


if __name__ == "__main__":
    main()
