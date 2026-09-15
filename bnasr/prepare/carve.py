#!/usr/bin/env python
"""Turn the per-source manifests into the three files the trainer and the corpus
builder consume:

    data2/train_corpus_ready.json   the six-source training pool (tokenizer-feasible rows)
    data2/val_corpus.json           development panel D (unseen speakers, five sources)
    data2/holdout_corpus.json       the unseen-speaker holdout (decoded once, after selection)

Two modes.

--from-lists (default when data/panels/ exists): reproduce the RELEASED split exactly.
    data/panels/{val_corpus,holdout_corpus,train_pool}.jsonl.gz list every row of the
    released files as (source, key, text_sha). Rows are matched to the freshly prepared
    manifests by (source, key); a text mismatch is counted and reported (it means the
    upstream release moved), a missing key is reported as well. Nothing is invented.

--fresh: carve a new speaker-disjoint holdout (~10 h, per-source caps) and development
    panel (~8 h across the five groupable sources), seeded, from the prepared manifests.
    Use this when a source release differs from the one the released lists refer to.

Both modes then apply the tokenizer-feasibility repair to the training pool: a row is
dropped if its token count exceeds the CTC frame budget (duration x 12.5 frames/s)
or its tokens-per-second exceed 11 — such rows are unlearnable and poison CTC.

    python bnasr/prepare/carve.py [--from-lists | --fresh] [--tokenizer data/tokenizer_v1024_corpus]
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # run as a script from anywhere
from bnasr.prepare._common import REPO, log, read_rows, text_sha, write_rows

SOURCES = ["openslr53", "fleurs", "commonvoice", "kathbath", "indicvoices", "subakko"]
SRC_NAME = {"fleurs": "fleurs_train"}
SEED = 20260830
FRAME_HZ, MAX_TPS = 12.5, 11.0
HOLDOUT_TARGET_H, HOLDOUT_CAPS = 10.0, {"indicvoices": 4.0, "openslr53": 2.0, "subakko": 2.0, "kathbath": 1.5, "commonvoice": 0.5}
PANEL_TARGET_H, PANEL_CAPS = 8.0, {"indicvoices": 3.2, "openslr53": 1.6, "subakko": 1.6, "kathbath": 1.2, "commonvoice": 0.4}
PANELS_DIR = REPO / "data/panels"


def load_manifests(data2):
    rows = []
    for s in SOURCES:
        m = data2 / s / "manifest.jsonl"
        if not m.exists():
            log(f"WARNING: {m} missing — source {s} absent from this build"); continue
        rs = read_rows(m)
        for r in rs:
            r["source"] = SRC_NAME.get(s, s)
        rows += rs
        log(f"{s}: {len(rs):,} rows")
    return rows


def from_lists(rows):
    by_key = {(r["source"], r["key"]): r for r in rows}
    out = {}
    for name in ("val_corpus", "holdout_corpus", "train_pool"):
        want = [json.loads(l) for l in gzip.open(PANELS_DIR / f"{name}.jsonl.gz", "rt", encoding="utf-8")]
        got, missing, moved = [], 0, 0
        for w in want:
            r = by_key.get((w["source"], w["key"]))
            if r is None:
                missing += 1; continue
            if text_sha(r["text"]) != w["text_sha"]:
                moved += 1
            got.append(r)
        log(f"{name}: released {len(want):,} rows -> matched {len(got):,}, missing {missing:,}, text moved {moved:,}")
        out[name] = got
    return out["train_pool"], out["val_corpus"], out["holdout_corpus"]


def fresh(rows):
    groups = defaultdict(list); ungrouped = 0
    for r in rows:
        if r["source"] == "fleurs_train" or not r.get("speaker_id"):
            ungrouped += 1; continue
        groups[(r["source"], r["speaker_id"])].append(r)
    log(f"fresh carve: {len(groups):,} speaker groups, {ungrouped:,} ungroupable rows stay in training")
    keys = sorted(groups); random.Random(SEED).shuffle(keys)

    def take(target_h, caps, used):
        chosen, per, tot = set(), defaultdict(float), 0.0
        for g in keys:
            if g in used or g[0] not in caps:
                continue
            gh = sum(r["duration"] for r in groups[g]) / 3600
            if per[g[0]] + gh > caps[g[0]] + 0.3 * scale:
                continue
            chosen.add(g); per[g[0]] += gh; tot += gh
            if tot >= target_h:
                break
        return chosen
    # Targets and caps scale down on a small pool (smoke runs, partial sources) so
    # the holdout and panel never swallow more than ~10 % of the hours each.
    total_h = sum(r["duration"] for r in rows) / 3600
    scale = min(1.0, 0.10 * total_h / HOLDOUT_TARGET_H)
    hold = take(HOLDOUT_TARGET_H * scale, {k: v * scale for k, v in HOLDOUT_CAPS.items()}, set())
    panel = take(PANEL_TARGET_H * scale, {k: v * scale for k, v in PANEL_CAPS.items()}, hold)
    holdout = [r for g in sorted(hold) for r in groups[g]]
    val = [r for g in sorted(panel) for r in groups[g]]
    taken = hold | panel
    eval_texts = {text_sha(r["text"]) for r in holdout + val}
    train = [r for r in rows if (r["source"], r.get("speaker_id")) not in taken and text_sha(r["text"]) not in eval_texts]
    log(f"holdout {len(holdout):,} rows / {sum(r['duration'] for r in holdout)/3600:.1f} h; panel {len(val):,} rows / "
        f"{sum(r['duration'] for r in val)/3600:.1f} h; train {len(train):,} rows")
    return train, val, holdout


def feasible(rows, tokenizer_dir):
    import sentencepiece as spm
    from bnasr.normalize import metric_surface
    sp = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_dir) / "tokenizer.model"))
    keep, dropped = [], 0
    for r in rows:
        u = len(sp.encode(metric_surface(r["text"], False)))
        if u > int(r["duration"] * FRAME_HZ) or u / max(r["duration"], 1e-6) > MAX_TPS:
            dropped += 1; continue
        keep.append(r)
    log(f"tokenizer feasibility: dropped {dropped:,} of {len(rows):,}")
    return keep


def strip(r):
    return {"audio_filepath": r["audio_filepath"], "text": r["text"], "duration": r["duration"], "source": r["source"],
            **({"speaker_id": r["speaker_id"]} if r.get("speaker_id") else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data2", default=str(REPO / "data2"))
    ap.add_argument("--tokenizer", default=str(REPO / "data/tokenizer_v1024_corpus"))
    g = ap.add_mutually_exclusive_group(); g.add_argument("--from-lists", action="store_true"); g.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    data2 = Path(a.data2)
    rows = load_manifests(data2)
    use_lists = a.from_lists or (not a.fresh and (PANELS_DIR / "train_pool.jsonl.gz").exists())
    train, val, holdout = from_lists(rows) if use_lists else fresh(rows)
    train = feasible(train, a.tokenizer)
    write_rows(data2 / "train_corpus_ready.json", [strip(r) for r in train])
    write_rows(data2 / "val_corpus.json", [strip(r) for r in val])
    write_rows(data2 / "holdout_corpus.json", [strip(r) for r in holdout])
    log(f"wrote {data2}/train_corpus_ready.json ({len(train):,}), val_corpus.json ({len(val):,}), holdout_corpus.json ({len(holdout):,})")


if __name__ == "__main__":
    main()
