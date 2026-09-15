#!/usr/bin/env python3
"""this recipe machine consensus audit of label quality (owner ruling 2026-09-06, replaces the 230-clip human verdict
export: "you infer and then compare and then decide").

Two INDEPENDENT recognisers judge every sampled clip: our the baseline FastConformer (trained on the old corpus only) and
hishab/titu_stt_bn_conformer_large (a different lab, different corpus). A clip is a CONSENSUS DISAGREEMENT when both
models are far from the reference but close to each other — the signature of a wrong label rather than a hard clip:
    WER(P1, ref) > 0.30  and  WER(H, ref) > 0.30  and  WER(P1, H) < 0.15
A clip is CATASTROPHIC-LIKE when both models are essentially unrelated to the reference (CER > 0.80 for both).
Registered decision rule (before any result was seen): a NEW bucket passes iff its consensus-disagreement rate is
≤ max(2 × old-corpus rate, old-corpus rate + 2 pp) AND its catastrophic-like rate ≤ old-corpus rate + 0.5 pp, where the
old-corpus rate is measured on the same protocol over the six the baseline sources (they are the accepted baseline).
Samples: the 230 listening-board clips (owner-reviewed) + 1,000 uniformly random retained clips per new bucket
(seed 20260906) + 500 per old source. Frozen scoring surface throughout. GPU: a few minutes on one card.
"""
import os
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.scoring import frozen_norm, wer_counts, cer_counts, anonymize_process, pin_gpu  # noqa: E402

anonymize_process()
SEED = 20260906
C = REPO / "experiments/corpus"


def load(p, n=None, seed=SEED):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if n and len(rows) > n:
        random.Random(seed).shuffle(rows); rows = rows[:n]
    return rows


def transcribe(model, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        hyps = model.transcribe(paths[i:i + bs], batch_size=bs, verbose=False)
        for h in hyps:
            out.append(h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else str(h)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-bucket", type=int, default=1000)
    ap.add_argument("--per-old-source", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "experiments/audit/machine_audit.json"))
    a = ap.parse_args()
    pin_gpu()
    import nemo.collections.asr as nemo_asr
    import torch

    groups = {}
    for name in ("macro_filtered", "madasr", "vaani"):
        groups[name] = load(C / f"retained_{name}.jsonl", a.per_bucket)
    old = [json.loads(l) for l in open(C / "arm_O.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for r in old:
        by[r["source"]].append(r)
    for s, rs in by.items():
        groups[f"old:{s}"] = load_list = rs
        random.Random(SEED).shuffle(load_list); groups[f"old:{s}"] = load_list[: a.per_old_source]
    for b in ("board_macro_madasr", "board_vaani"):
        sp = REPO / f"experiments/audit/{b}.sample.json"
        if sp.exists():
            sj = json.load(open(sp, encoding="utf-8"))
            clips = sj.get("clips", sj)
            rows = []
            for c in clips:
                p = c.get("audio_filepath") or c.get("audio_relpath") or c.get("path") or c.get("src")
                t = c.get("text") or c.get("ref") or c.get("reference")
                if p and t:
                    rows.append({"audio_filepath": p, "text": t, "duration": c.get("duration", 0)})
            groups[f"board:{b}"] = rows
    items = [(g, r) for g, rs in groups.items() for r in rs]
    paths = [str(REPO / r["audio_filepath"]) if not r["audio_filepath"].startswith("/") else r["audio_filepath"] for _, r in items]
    print(f"{len(items)} clips across {len(groups)} groups", flush=True)

    results = {}
    for tag, loader in (("P1", lambda: nemo_asr.models.ASRModel.restore_from(str(REPO / "experiments/fastconformer_ctc_bn_1kh/final.nemo"), map_location="cuda")),
                        ("H", lambda: nemo_asr.models.ASRModel.from_pretrained("hishab/titu_stt_bn_conformer_large", map_location="cuda"))):
        t0 = time.time(); m = loader(); m.eval()
        with torch.inference_mode():
            hyps = transcribe(m, paths)
        results[tag] = [frozen_norm(h or "") for h in hyps]
        del m; torch.cuda.empty_cache()
        print(f"{tag}: decoded {len(hyps)} in {time.time()-t0:.0f}s", flush=True)

    per = defaultdict(lambda: {"n": 0, "consensus_disagreement": 0, "catastrophic_like": 0,
                               "wer_p1_sum": [0, 0], "wer_h_sum": [0, 0]})
    detail = []
    for i, (g, r) in enumerate(items):
        ref = frozen_norm(r["text"]); p1 = results["P1"][i]; h = results["H"][i]
        e1, n = wer_counts(ref, p1); eh, _ = wer_counts(ref, h); ex, nx = wer_counts(p1 or "*", h or "*")
        c1, cn = cer_counts(ref, p1); ch, _ = cer_counts(ref, h)
        w1, wh = e1 / max(n, 1), eh / max(n, 1); w12 = ex / max(nx, 1)
        cd = w1 > 0.30 and wh > 0.30 and w12 < 0.15
        cat = (c1 / max(cn, 1)) > 0.80 and (ch / max(cn, 1)) > 0.80
        d = per[g]; d["n"] += 1; d["consensus_disagreement"] += cd; d["catastrophic_like"] += cat
        d["wer_p1_sum"][0] += e1; d["wer_p1_sum"][1] += n; d["wer_h_sum"][0] += eh; d["wer_h_sum"][1] += n
        if cd or cat:
            detail.append({"group": g, "path": r["audio_filepath"], "ref": ref, "p1": p1, "h": h,
                           "wer_p1": round(w1, 3), "wer_h": round(wh, 3), "wer_p1_h": round(w12, 3), "cat": cat})
    summary = {}
    for g, d in per.items():
        summary[g] = {"n": d["n"], "consensus_disagreement_rate": round(d["consensus_disagreement"] / d["n"], 4),
                      "catastrophic_like_rate": round(d["catastrophic_like"] / d["n"], 4),
                      "wer_p1": round(d["wer_p1_sum"][0] / max(d["wer_p1_sum"][1], 1), 4),
                      "wer_hishab": round(d["wer_h_sum"][0] / max(d["wer_h_sum"][1], 1), 4)}
    olds = [v for k, v in summary.items() if k.startswith("old:")]
    old_n = sum(per[k]["n"] for k in per if k.startswith("old:"))
    old_cd = sum(per[k]["consensus_disagreement"] for k in per if k.startswith("old:")) / max(old_n, 1)
    old_cat = sum(per[k]["catastrophic_like"] for k in per if k.startswith("old:")) / max(old_n, 1)
    verdicts = {}
    for b in ("macro_filtered", "madasr", "vaani"):
        v = summary[b]
        ok = (v["consensus_disagreement_rate"] <= max(2 * old_cd, old_cd + 0.02)
              and v["catastrophic_like_rate"] <= old_cat + 0.005)
        verdicts[b] = {"pass": bool(ok), "cd_rate": v["consensus_disagreement_rate"], "cd_limit": round(max(2 * old_cd, old_cd + 0.02), 4),
                       "cat_rate": v["catastrophic_like_rate"], "cat_limit": round(old_cat + 0.005, 4)}
    out = {"rule": __doc__.split("Registered decision rule")[1].split("Samples")[0].strip(), "seed": SEED,
           "old_corpus_baseline": {"n": old_n, "consensus_disagreement_rate": round(old_cd, 4), "catastrophic_like_rate": round(old_cat, 4)},
           "summary": summary, "verdicts": verdicts, "flagged_examples": detail[:400], "n_flagged": len(detail)}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"baseline": out["old_corpus_baseline"], "summary": summary, "verdicts": verdicts}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
