#!/usr/bin/env python3
"""SPEC §8t / council L Q10: CTC blank-collapse diagnostics for a validation step, from the saved predictions.

Collapse = near-all-blank output: near-100 % empties, hypothesis/reference token-length ratio near zero, deletion-dominated
errors - even while the loss falls. Healthy CTC can still have many blank frames. This reads runs/<run>/val_preds/<panel>_step_<u>.json
and the panel manifest; it decodes nothing. Exit 0 always (diagnostic); prints one JSON line per panel and writes
runs/<run>/collapse_<step>.json.
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.scoring import frozen_norm   # noqa: E402  - the frozen scoring surface, same as the admission test
from bnasr.common import PANELS, load_manifest   # noqa: E402
def toks(t): return t.split()
def lev_ops(ref, hyp):
    """Return (sub, ins, del) word-level edit counts via DP with backtrace."""
    n, m = len(ref), len(hyp)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(ref[i-1]!=hyp[j-1]))
    i, j = n, m; S = I = Dl = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (ref[i-1]!=hyp[j-1]):
            S += (ref[i-1]!=hyp[j-1]); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i-1][j]+1: Dl += 1; i -= 1
        else: I += 1; j -= 1
    return S, I, Dl
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--panels", default="D"); a = ap.parse_args()
    rd = REPO / "experiments/runs" / a.run; out = {"run": a.run, "step": a.step, "panels": {}}
    for panel in a.panels.split(","):
        pf = rd / "val_preds" / f"{panel}_step_{a.step}.json"
        if not pf.exists(): out["panels"][panel] = {"missing": str(pf)}; continue
        hyps = json.load(open(pf, encoding="utf-8"))          # positional list, panel-manifest row order
        rows = load_manifest(PANELS[panel])
        if not isinstance(hyps, list) or len(hyps) != len(rows):
            out["panels"][panel] = {"refused": f"{len(hyps) if isinstance(hyps, list) else '?'} hyps vs {len(rows)} rows"}; continue
        pairs = [(frozen_norm(r["text"]), frozen_norm(h or "")) for r, h in zip(rows, hyps)]
        n = len(pairs); empties = sum(1 for r, h in pairs if not h.strip())
        rw = sum(len(toks(r)) for r, _ in pairs); hw = sum(len(toks(h)) for _, h in pairs)
        S = I = Dl = 0
        sample = pairs if n <= 1500 else pairs[::max(1, n//1500)]
        for r, h in sample:
            s_, i_, d_ = lev_ops(toks(r), toks(h)); S += s_; I += i_; Dl += d_
        tot = max(1, S+I+Dl)
        res = {"n": n, "empty_rate": round(empties/n, 4), "hyp_over_ref_words": round(hw/max(1, rw), 4),
               "sub_share": round(S/tot, 3), "ins_share": round(I/tot, 3), "del_share": round(Dl/tot, 3),
               "ops_sample_n": len(sample)}
        res["verdict"] = ("COLLAPSE-LIKE" if (res["empty_rate"] > 0.5 or res["hyp_over_ref_words"] < 0.2 or (res["del_share"] > 0.8 and res["hyp_over_ref_words"] < 0.5))
                          else "healthy-shape")
        out["panels"][panel] = res; print(json.dumps({panel: res}))
    (rd / f"collapse_{a.step}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
if __name__ == "__main__": main()
