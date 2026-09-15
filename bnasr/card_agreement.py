#!/usr/bin/env python3
"""SPEC §8q.5 / §8s.4 hardware qualification: the O_A6000 control vs O_ref (O_A5000) at update 8,000.

Required: |ΔS| <= 0.10 pp, |ΔD| <= 0.20 pp, every D source |Δ| <= 0.30 pp. A missing record is NOT a pass.
Exit 0 = PASS, 2 = FAIL, 3 = not evaluable. Writes experiments/card_agreement_<step>.json.
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
def rec(run, step):
    p = REPO / "experiments/runs" / run / "val.jsonl"
    if not p.exists(): return None
    hits = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("update") == step]
    return hits[0] if len(hits) == 1 else None
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a6000", default="O_A6000"); ap.add_argument("--ref", default="O_A5000")
    ap.add_argument("--step", type=int, default=8000); a = ap.parse_args()
    x, r = rec(a.a6000, a.step), rec(a.ref, a.step)
    out = {"step": a.step, "a6000_run": a.a6000, "ref_run": a.ref, "bounds_pp": {"S": 0.10, "D": 0.20, "source": 0.30}}
    if x is None or r is None:
        out["result"] = "not_evaluable"; out["reason"] = f"missing exactly-one {a.step} record: a6000={x is not None} ref={r is not None}"
        print(json.dumps(out, indent=1)); (REPO / f"experiments/card_agreement_{a.step}.json").write_text(json.dumps(out, indent=1)); return 3
    dS = x["S_pp"] - r["S_pp"]; dD = (x["panels"]["D"]["wer_frozen"] - r["panels"]["D"]["wer_frozen"]) * 100
    src = {k: (x["panels"]["D"]["per_source"][k] - r["panels"]["D"]["per_source"][k]) * 100 for k in r["panels"]["D"]["per_source"]}
    checks = {"S": abs(dS) <= 0.10 + 1e-9, "D": abs(dD) <= 0.20 + 1e-9, **{f"source/{k}": abs(v) <= 0.30 + 1e-9 for k, v in src.items()}}
    out.update({"delta_S_pp": round(dS, 4), "delta_D_pp": round(dD, 4), "delta_source_pp": {k: round(v, 4) for k, v in src.items()},
                "S": [x["S_pp"], r["S_pp"]], "checks": checks, "result": "pass" if all(checks.values()) else "fail"})
    print(json.dumps(out, indent=1)); (REPO / f"experiments/card_agreement_{a.step}.json").write_text(json.dumps(out, indent=1))
    return 0 if out["result"] == "pass" else 2
if __name__ == "__main__": sys.exit(main())
