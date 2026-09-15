#!/usr/bin/env python3
"""Council D Q1 cross-card check: the first N optimizer updates of two runs that consumed the IDENTICAL input stream
(same manifest, seed, batch config) on different cards must agree exactly on cut order and LR and closely on loss.

Registered rule (SPEC §5 O_A6000 row): identical cut ids and LR sequence at every update; ≤1 % relative difference
between corresponding 20-update mean losses; finite losses and gradients on both. This is an engineering tripwire for
"same code path on both cards", not a proof of numerical equivalence at the 0.3 pp admission margin.
Usage: cross_card_check.py <run_a steps.jsonl> <run_b steps.jsonl> [--updates 200] [--out report.json]
"""
import argparse
import json
import math
from pathlib import Path


def load(p, n):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("update") is not None][:n]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = load(args.a, args.updates), load(args.b, args.updates)
    n = min(len(A), len(B))
    rep = {"updates_compared": n, "required": args.updates}
    id_key = next((k for k in ("micro_ids", "ids", "cut_ids", "first_ids") if A and k in A[0]), None)
    rep["id_key"] = id_key
    id_mism = [i + 1 for i in range(n) if id_key and A[i].get(id_key) != B[i].get(id_key)]
    lr_mism = [i + 1 for i in range(n) if not math.isclose(float(A[i]["lr"]), float(B[i]["lr"]), rel_tol=1e-9)]
    nonfinite = [i + 1 for i in range(n) for r in (A[i], B[i])
                 if not (math.isfinite(float(r["loss"])) and math.isfinite(float(r.get("grad_norm", 0.0))))]
    blocks = []
    for s in range(0, n - n % 20, 20):
        ma = sum(float(r["loss"]) for r in A[s:s + 20]) / 20
        mb = sum(float(r["loss"]) for r in B[s:s + 20]) / 20
        blocks.append({"updates": f"{s + 1}-{s + 20}", "mean_a": round(ma, 5), "mean_b": round(mb, 5),
                       "rel_diff": round(abs(ma - mb) / max(abs(ma), 1e-9), 5)})
    worst = max((b["rel_diff"] for b in blocks), default=None)
    rep.update({"cut_id_mismatches": id_mism[:20], "n_cut_id_mismatches": len(id_mism),
                "lr_mismatches": lr_mism[:20], "n_lr_mismatches": len(lr_mism),
                "nonfinite_updates": nonfinite[:20], "blocks": blocks, "worst_block_rel_diff": worst,
                "max_abs_loss_diff": max((abs(float(A[i]["loss"]) - float(B[i]["loss"])) for i in range(n)), default=None)})
    rep["PASS"] = bool(n >= args.updates and not id_mism and not lr_mism and not nonfinite
                       and worst is not None and worst <= 0.01 and (id_key is not None))
    if id_key is None:
        rep["note"] = "steps.jsonl carries no per-update cut-id field; cut order could not be compared"
    s = json.dumps(rep, indent=1)
    if args.out:
        Path(args.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
