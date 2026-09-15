#!/usr/bin/env python
"""Reconstruct the frozen panel WERs and S EXACTLY, from preserved predictions.

SPEC §8w.7 defect 7 (council Q, Q5 #7): ``ctc_panel_eval`` rounds every WER
fraction to five decimal places BEFORE the number is used for selection, the
eligibility guards and the empty-rate guard.  Five decimals on a fraction is a
0.001 pp grid with up to 0.0005 pp of error per panel, and S is rounded again
on top of that, so a 1e-9 pp comparison tolerance cannot restore what was
discarded.

This script does the arithmetic the forward fix will do: it re-scores each
already-written validation from the hypotheses preserved under
``<run>/val_preds/{PANEL}_step_{UPDATE}.json`` using the repo's OWN frozen
surface (``frozen_norm`` + ``wer_counts`` out of ``bnasr/scoring.py``, both
re-exported by ``bnasr/common.py``), keeps the INTEGER edit and reference
word counts, and derives every decision value from the exact rational
``edits / ref_words`` with no display rounding anywhere.  S is recomputed with
the registered weights via the same ``P4.selection_score`` weight vector
(MADASR admitted, §8m: 0.60 D + 0.20 M + 0.10 R + 0.10 K on percentage points).

It is READ-ONLY with respect to the run: nothing under ``experiments/runs/``
is opened for writing, no model is loaded and no GPU is touched.  It is pure
CPU text scoring over the four panel manifests per update.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 ionice -c 3 \
        .venv/bin/python bnasr/recompute_unrounded.py \
            [--run experiments/runs/N_O+macro_filtered_H120k] \
            [--config configs/ctc_from_nvidia.yaml] \
            [--out experiments/unrounded_recon.json]

The output JSON carries, per update and per panel: the recorded (5-dp) value,
the exact value as an integer ratio and as a float, the difference in pp, and
the empty counts; plus recorded vs exact S, and a verdict block that answers
the only question that matters - whether any registered comparison (the
per-source and conversational eligibility guards, the empty-rate guard, the
0.05 pp selection tie band) would have decided differently on the exact
numbers.  A panel whose exact value does not agree with the recorded one to
within the expected 5-dp rounding error (5e-6 on the fraction) is flagged as a
MISMATCH: that would mean the recorded number is not a rounded version of what
the preserved predictions say, which is a far more serious finding than the
rounding grid itself.
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from bnasr import common as P4                                    # noqa: E402
from bnasr.common import (frozen_norm, wer_counts, cer_counts,   # noqa: E402
                       load_manifest, is_conversational)

# The tolerance a 5-dp round can introduce on a WER *fraction*.  Anything
# larger means the recorded value is not merely rounded - it is wrong.
ROUND_TOL = 5e-6 + 1e-12


def exact_panel(manifest_path, hyps):
    """Re-score one panel exactly: integer counts, no rounding.

    Mirrors ``ctc_panel_eval``'s accumulation byte for byte (same
    ``frozen_norm``, same ``wer_counts`` empty-hypothesis convention, same
    per-source and conversational slicing) but keeps the numerator and
    denominator as integers instead of collapsing to a rounded float.
    """
    rows = load_manifest(manifest_path)
    if len(hyps) != len(rows):
        raise SystemExit(
            f"{manifest_path}: {len(hyps)} hypotheses for {len(rows)} rows - "
            "the preserved predictions do not correspond to this manifest")
    we = wn = ce = cn = 0
    empties = 0
    src = {}
    conv = [0, 0]
    for row, hyp in zip(rows, hyps):
        nr, nh = frozen_norm(row["text"]), frozen_norm(hyp)
        if not nh.strip():
            empties += 1
        e, n = wer_counts(nr, nh)
        c, m = cer_counts(nr, nh)
        we, wn, ce, cn = we + e, wn + n, ce + c, cn + m
        acc = src.setdefault(row.get("source", "?"), [0, 0])
        acc[0] += e
        acc[1] += n
        if is_conversational(row):
            conv[0] += e
            conv[1] += n
    return {
        "n": len(rows),
        "wer_edits": we, "wer_ref_words": wn,
        "cer_edits": ce, "cer_ref_chars": cn,
        "empties": empties,
        "per_source": {s: {"edits": a[0], "ref_words": a[1]}
                       for s, a in sorted(src.items())},
        "conversational": {"edits": conv[0], "ref_words": conv[1]},
    }


def ratio(num, den):
    """Exact fraction and its nearest float, with den==0 handled as 0/1."""
    f = Fraction(num, den) if den else Fraction(0, 1)
    return f, float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/N_O+macro_filtered_H120k")
    ap.add_argument("--config", default="configs/ctc_from_nvidia.yaml")
    ap.add_argument("--out", default="experiments/unrounded_recon.json")
    args = ap.parse_args()

    run = Path(args.run)
    run = run if run.is_absolute() else REPO / run
    cfg_path = Path(args.config)
    cfg_path = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else REPO / out_path
    if run in out_path.parents or out_path.is_relative_to(REPO / "experiments/runs"):
        raise SystemExit("refusing to write inside experiments/runs/")

    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    panels = {}
    for name, rel in cfg["val"]["panels"].items():
        p = Path(rel)
        panels[name] = p if p.is_absolute() else REPO / p
        if not panels[name].exists():
            raise SystemExit(f"registered panel {name} missing: {rel}")

    admitted = P4.madasr_admitted(cfg)
    weights = (P4.S_WEIGHTS_WITH_MADASR if admitted
               else P4.S_WEIGHTS_NO_MADASR)

    records = [json.loads(l) for l in
               (run / "val.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]

    doc = {
        "run": str(run),
        "config": str(cfg_path),
        "defect": "SPEC 8w.7 #7 - 5-dp rounding of WER before selection and guards",
        "method": ("re-scored from preserved val_preds/ with frozen_norm + "
                   "wer_counts; WER kept as the exact integer ratio "
                   "edits/ref_words; S = sum(w * wer * 100) with the registered "
                   "weights, unrounded"),
        "madasr_admitted": admitted,
        "s_weights": weights,
        "tie_band_pp": P4.TIE_BAND_PP,
        "guard_source_pp": P4.GUARD_SOURCE_PP,
        "guard_conversational_pp": P4.GUARD_CONVERSATIONAL_PP,
        "guard_empty_rate": P4.GUARD_EMPTY_RATE,
        "round_tolerance_on_fraction": ROUND_TOL,
        "updates": [],
    }
    baseline = P4.load_baseline() or {}
    max_panel_dpp = 0.0
    max_s_dpp = 0.0
    mismatches = []

    for rec in records:
        update = int(rec["update"])
        entry = {"update": update, "panels": {}}
        exact_wers = {}
        for name in sorted(panels):
            pred = run / "val_preds" / f"{name}_step_{update}.json"
            if not pred.exists():
                entry["panels"][name] = {"error": f"no preserved predictions: {pred}"}
                continue
            hyps = json.loads(pred.read_text(encoding="utf-8"))
            ex = exact_panel(panels[name], hyps)
            frac, val = ratio(ex["wer_edits"], ex["wer_ref_words"])
            cfrac, cval = ratio(ex["cer_edits"], ex["cer_ref_chars"])
            erate, erval = ratio(ex["empties"], ex["n"])
            exact_wers[name] = val
            recorded = rec["panels"].get(name, {})
            rec_w = recorded.get("wer_frozen")
            d_pp = (val - float(rec_w)) * 100.0 if rec_w is not None else None
            ok = rec_w is not None and abs(val - float(rec_w)) <= ROUND_TOL
            if rec_w is not None and not ok:
                mismatches.append({"update": update, "panel": name,
                                   "recorded": rec_w, "exact": val,
                                   "delta_pp": d_pp})
            if d_pp is not None:
                max_panel_dpp = max(max_panel_dpp, abs(d_pp))
            p = {
                "n": ex["n"],
                "wer_edits": ex["wer_edits"],
                "wer_ref_words": ex["wer_ref_words"],
                "wer_exact_ratio": f'{ex["wer_edits"]}/{ex["wer_ref_words"]}',
                "wer_recorded": rec_w,
                "wer_exact": val,
                "wer_delta_pp": d_pp,
                "within_rounding_error": ok,
                "cer_exact_ratio": f'{ex["cer_edits"]}/{ex["cer_ref_chars"]}',
                "cer_recorded": recorded.get("cer_frozen"),
                "cer_exact": cval,
                "empties": ex["empties"],
                "empties_recorded": recorded.get("empties"),
                "empty_rate_exact_ratio": f'{ex["empties"]}/{ex["n"]}',
                "empty_rate_recorded": recorded.get("empty_rate"),
                "empty_rate_exact": erval,
                "empty_rate_guard_limit": P4.GUARD_EMPTY_RATE,
                "empty_rate_guard_pass_recorded": (
                    None if recorded.get("empty_rate") is None
                    else float(recorded["empty_rate"]) <= P4.GUARD_EMPTY_RATE),
                "empty_rate_guard_pass_exact": erval <= P4.GUARD_EMPTY_RATE,
            }
            if name == "D":
                base_src = ((baseline.get("D") or {}).get("per_source") or {})
                per = {}
                for s, a in ex["per_source"].items():
                    _, sval = ratio(a["edits"], a["ref_words"])
                    srec = (recorded.get("per_source") or {}).get(s)
                    b = base_src.get(s)
                    row = {
                        "exact_ratio": f'{a["edits"]}/{a["ref_words"]}',
                        "recorded": srec, "exact": sval,
                        "delta_vs_recorded_pp": (None if srec is None
                                                 else (sval - float(srec)) * 100.0),
                    }
                    if b is not None:
                        row["guard_delta_pp_recorded"] = (
                            None if srec is None else (float(srec) - float(b)) * 100.0)
                        row["guard_delta_pp_exact"] = (sval - float(b)) * 100.0
                        row["guard_limit_pp"] = P4.GUARD_SOURCE_PP
                        row["guard_pass_recorded"] = (
                            None if srec is None
                            else (float(srec) - float(b)) * 100.0 <= P4.GUARD_SOURCE_PP)
                        row["guard_pass_exact"] = (
                            (sval - float(b)) * 100.0 <= P4.GUARD_SOURCE_PP)
                    per[s] = row
                p["per_source"] = per
                c = ex["conversational"]
                _, cw = ratio(c["edits"], c["ref_words"])
                crec = (recorded.get("conversational") or {}).get("wer_frozen")
                cb = ((baseline.get("D") or {}).get("conversational") or {}).get("wer_frozen")
                conv = {"exact_ratio": f'{c["edits"]}/{c["ref_words"]}',
                        "n_ref_words": c["ref_words"],
                        "recorded": crec, "exact": cw,
                        "delta_vs_recorded_pp": (None if crec is None
                                                 else (cw - float(crec)) * 100.0)}
                if cb is not None:
                    conv["guard_delta_pp_recorded"] = (
                        None if crec is None else (float(crec) - float(cb)) * 100.0)
                    conv["guard_delta_pp_exact"] = (cw - float(cb)) * 100.0
                    conv["guard_limit_pp"] = P4.GUARD_CONVERSATIONAL_PP
                    conv["guard_pass_recorded"] = (
                        None if crec is None
                        else (float(crec) - float(cb)) * 100.0 <= P4.GUARD_CONVERSATIONAL_PP)
                    conv["guard_pass_exact"] = (
                        (cw - float(cb)) * 100.0 <= P4.GUARD_CONVERSATIONAL_PP)
                p["conversational"] = conv
            entry["panels"][name] = p

        if all(k in exact_wers for k in weights):
            s_exact = sum(weights[k] * exact_wers[k] * 100.0 for k in weights)
        else:
            s_exact = None
        s_rec = rec.get("S_pp")
        entry["S_recorded_pp"] = s_rec
        entry["S_exact_pp"] = s_exact
        entry["S_delta_pp"] = (None if (s_exact is None or s_rec is None)
                               else s_exact - float(s_rec))
        if entry["S_delta_pp"] is not None:
            max_s_dpp = max(max_s_dpp, abs(entry["S_delta_pp"]))
        entry["eligible_recorded"] = P4.is_eligible(rec)
        # Eligibility on the exact numbers: every registered guard re-evaluated.
        checks = []
        d = entry["panels"].get("D", {})
        for s, row in (d.get("per_source") or {}).items():
            if "guard_pass_exact" in row:
                checks.append(row["guard_pass_exact"])
        if "guard_pass_exact" in (d.get("conversational") or {}):
            checks.append(d["conversational"]["guard_pass_exact"])
        if "empty_rate_guard_pass_exact" in d:
            checks.append(d["empty_rate_guard_pass_exact"])
        entry["eligible_exact"] = bool(checks) and all(checks)
        entry["eligibility_changed"] = (
            entry["eligible_exact"] != entry["eligible_recorded"])
        doc["updates"].append(entry)

    # ------------------------------------------------------------------ tie band
    hist_rec = [(u["S_recorded_pp"], u["update"], u["eligible_recorded"])
                for u in doc["updates"]]
    hist_exa = [(u["S_exact_pp"], u["update"], u["eligible_exact"])
                for u in doc["updates"]]
    sel_rec = P4.select_best(hist_rec)
    sel_exa = P4.select_best(hist_exa)
    ordered_rec = [u["update"] for u in
                   sorted(doc["updates"], key=lambda x: (x["S_recorded_pp"], x["update"]))]
    ordered_exa = [u["update"] for u in
                   sorted(doc["updates"], key=lambda x: (x["S_exact_pp"], x["update"]))]
    s_vals = sorted(u["S_exact_pp"] for u in doc["updates"])
    closest = min((b - a for a, b in zip(s_vals, s_vals[1:])), default=None)

    doc["verdict"] = {
        "n_updates": len(doc["updates"]),
        "max_abs_panel_wer_delta_pp": max_panel_dpp,
        "max_abs_S_delta_pp": max_s_dpp,
        "panels_outside_expected_rounding_error": mismatches,
        "recomputation_matches_recorded": not mismatches,
        "eligibility_changed_at": [u["update"] for u in doc["updates"]
                                   if u["eligibility_changed"]],
        "empty_rate_guard_changed_at": [
            u["update"] for u in doc["updates"]
            for p in u["panels"].values()
            if p.get("empty_rate_guard_pass_recorded") is not None
            and p.get("empty_rate_guard_pass_exact")
            != p.get("empty_rate_guard_pass_recorded")],
        "S_ranking_recorded": ordered_rec,
        "S_ranking_exact": ordered_exa,
        "S_ranking_changed": ordered_rec != ordered_exa,
        "closest_S_gap_pp": closest,
        "select_best_recorded": sel_rec,
        "select_best_exact": sel_exa,
        "select_best_changed": sel_rec != sel_exa,
        "note": ("select_best is reported for completeness; no checkpoint is "
                 "eligible yet (the P1 per-source guards are still far from "
                 "passing this early in the run), so the tie band has nothing "
                 "to act on. The relevant test is whether any pair of S values "
                 "is close enough that the rounding grid could reorder them: "
                 "compare closest_S_gap_pp with max_abs_S_delta_pp."),
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(doc["verdict"], ensure_ascii=False, indent=2))
    for u in doc["updates"]:
        print(f'\nupdate {u["update"]}:  S recorded {u["S_recorded_pp"]}  '
              f'exact {u["S_exact_pp"]!r}  delta {u["S_delta_pp"]:+.3e} pp')
        for name, p in sorted(u["panels"].items()):
            print(f'  {name}: recorded {p["wer_recorded"]}  exact '
                  f'{p["wer_exact_ratio"]} = {p["wer_exact"]!r}  '
                  f'delta {p["wer_delta_pp"]:+.3e} pp  '
                  f'{"OK" if p["within_rounding_error"] else "MISMATCH"}')
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
