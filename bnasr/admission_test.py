#!/usr/bin/env python3
"""this recipe 8,000-update ADMISSION test for one arm against the control O_ref (SPEC §5, §8k Q1/Q3, §8m).

What this decides, and against what
-----------------------------------
This tool answers exactly one question: does a candidate arm's 8,000-update checkpoint clear the registered
*admission* rule against the control `O_ref`?  It is NOT the continuation/final eligibility test.  SPEC §8k Q2 is
explicit that those two comparisons have different baselines: **admission compares against O_ref**, while
continuation/checkpoint eligibility compares against the frozen baseline (every D source ≤ P1 +1.0 pp,
conversational ≤ P1 +0.5 pp = 22.075 %).  "Admission against O_ref does NOT certify continuation eligibility."
The P1 comparison is therefore computed and reported here as clearly-labelled INFORMATION ONLY and never enters
`ADMITTED`; conflating the two is how an arm gets promoted on the wrong yardstick.

The registered rule (SPEC §5 "Bucket pass rule at 8,000", tightened by §8l/§8m to m = 4)
--------------------------------------------------------------------------------------
All of the following, jointly:
  * point:        Δ = S(arm) − S(control) ≤ −0.30 pp
  * marginal:     upper end of the paired 95 % bootstrap CI of Δ < 0
  * simultaneous: the EMPIRICAL bootstrap quantile of Δ at 1 − 0.025/m < 0.  With m = 4 predeclared single-bucket
                  arms (macro_filtered, madasr, vaani, cv17 — §8m keeps m = 4 even though cv17's arm is deferred,
                  "deferring or skipping an arm never reduces m") the quantile is 0.99375.  This is a quantile of
                  the 10,000 realised draws, NOT a normal approximation: nothing here forms mean ± z·sd, because
                  Δ is a ratio-of-sums statistic whose bootstrap distribution is visibly skewed and whose tail is
                  the whole point of a simultaneous bound.
  * guards vs the control: D ≤ +0.30 pp, each D source ≤ +1.00 pp, conversational ≤ +0.50 pp, empty rate ≤ 0.1 %.

The bootstrap (SPEC §5 "paired sentence-clustered 95 % CI", 10,000 draws, seed 20260905)
----------------------------------------------------------------------------------------
A cluster is a FROZEN-SURFACE REFERENCE STRING (`scoring.frozen_norm` of the manifest `text`), and the cluster
table is built ACROSS ALL FOUR PANELS at once, so a sentence that appears in two panels is one cluster and
resamples TOGETHER in both — panel D and panel M genuinely share sentences (MaCro and the Common Voice Bangla
campaign draw on the same prompt pool), and treating those as independent units would understate the variance of
Δ exactly where the shared rows drive it.  One draw:

    draw n_clusters cluster ids with replacement  ->  integer multiplicity vector `counts`
    for every panel:  WER = Σ counts[cluster(row)]·edits(row) / Σ counts[cluster(row)]·ref_words(row)
    Δ = Σ_p weight[p] · (WER_arm[p] − WER_control[p]) · 100

i.e. corpus WER is summed edits over summed reference words WITHIN each draw (never a mean of per-utterance
rates), the SAME multiplicity vector is applied to the arm and to the control (paired), and S uses the registered
0.60 D / 0.20 M / 0.10 R / 0.10 K weighting, kept frozen by §8k Q1 for admission and for checkpoint selection.
`--selftest` proves the cross-panel coupling on a fixture in which one sentence sits in two panels.

Hypothesis ORDERING is the silent failure mode this tool refuses to have
-----------------------------------------------------------------------
`val_preds/<panel>_step_<step>.json` is a positional list written by `train.validate()` in the panel
manifest's row order (`common.ctc_panel_eval` asserts `len(hyps) == len(rows)` and zips them positionally).
If that alignment were ever off — a re-ordered manifest, a partially rewritten prediction file, a run resumed
against a different panel — every number below would still look plausible while scoring the wrong pairs.  So the
tool refuses to run unless BOTH of these hold: the hypothesis list length equals the manifest row count, and the
per-panel corpus WER recomputed here from (manifest, val_preds) reproduces the trainer's own `val.jsonl` entry
for that step to within 0.005 pp, for the arm AND for the control.  The trainer computed its WER from the
in-memory decode; reproducing it from disk in manifest order is what proves the stored list is still in that
order.  Everything is read from disk; nothing is decoded here.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.scoring import frozen_norm, wer_counts  # noqa: E402
from bnasr.common import CONVERSATIONAL, PANELS, S_WEIGHTS_WITH_MADASR, load_manifest  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Registered constants.  Every one is asserted against the SPEC section that registers it: a constant that drifts
# from SPEC silently re-decides an already-recorded verdict, and the assert is the only thing that makes editing
# this file and editing SPEC.md the same act.
# ---------------------------------------------------------------------------------------------------------------
WEIGHTS = dict(S_WEIGHTS_WITH_MADASR)          # single source of truth: the trainer scores S with this same dict
assert WEIGHTS == {"D": 0.60, "M": 0.20, "R": 0.10, "K": 0.10}, "SPEC §5/§8k Q1: S = 0.60 D + 0.20 M + 0.10 R + 0.10 K, FROZEN"
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12

SEED = 20260905                                # SPEC §5: "10,000 draws, seed 20260905"
DRAWS = 10000                                  # SPEC §5
ALPHA = 0.025                                  # SPEC §5: one-sided 2.5 % per arm, Bonferroni-split across m arms
M_DEFAULT = 4                                  # SPEC §8l/§8m: "m stays 4 ... deferring or skipping an arm never reduces m"
assert abs((1 - ALPHA / M_DEFAULT) - 0.99375) < 1e-12, "SPEC §8m: the simultaneous quantile at m=4 is 1-0.025/4 = 0.99375"

DELTA_MAX_PP = -0.30                           # SPEC §5: "S <= S(O) - 0.3 pp"
GUARD_D_PP = 0.30                              # SPEC §5: "D <= D(O) + 0.3 pp"
GUARD_SOURCE_PP = 1.00                         # SPEC §5: "no D source worse than O by >1.0 pp"
GUARD_CONV_PP = 0.50                           # SPEC §5: "conversational slice worse by <=0.5 pp"
GUARD_EMPTY = 0.001                            # SPEC §5: "empties <=0.1 %"
ORDER_TOL_PP = 0.005                           # this tool: recomputed vs trainer per-panel WER, the ordering proof

# The continuation/final guards (SPEC §8k Q2) — reported, never used for ADMITTED.
P1_CONV_LIMIT_PP = 22.075                      # SPEC §8k Q2: "conversational <= P1 +0.5 pp = 22.075 %" (P1 exactly 21.575)
P1_SOURCE_PP = 1.00                            # SPEC §8k Q2: "every D source <= P1 +1.0 pp"
BASELINE_P1 = REPO / "data/baselines/baseline.json"

PANEL_MANIFEST = dict(PANELS)                  # SPEC §2, via common.PANELS — the trainer decodes exactly these
assert set(PANEL_MANIFEST) == set(WEIGHTS), "SPEC §2/§5: S is defined on exactly the panels D/M/K/R"
assert PANEL_MANIFEST["D"] == REPO / "data2/val_corpus.json"
assert CONVERSATIONAL == {"source": "indicvoices", "scenario": "Conversation"}   # SPEC §2 conversational slice


def panel_counts(run_dir, panel, step, manifests):
    """-> (cluster_key[], edits[], ref_words[], empties) aligned to the panel manifest's row order.

    `load_manifest` is scoring's, i.e. literally the function `ctc_panel_eval` used to build the row list the
    hypotheses were zipped onto — so "manifest order" here is the same object, not a re-derivation of it.
    """
    hp = Path(run_dir) / "val_preds" / f"{panel}_step_{step}.json"
    if not hp.exists():
        raise SystemExit(f"REFUSING: {hp} missing")
    hyps = json.load(open(hp, encoding="utf-8"))
    rows = load_manifest(manifests[panel])
    if not isinstance(hyps, list):
        raise SystemExit(f"REFUSING: {hp} is not a positional list (got {type(hyps).__name__})")
    if len(hyps) != len(rows):
        raise SystemExit(f"REFUSING: {panel}: {len(hyps)} hypotheses vs {len(rows)} manifest rows in {manifests[panel]}")
    keys, errs, refs, empties = [], [], [], 0
    for r, h in zip(rows, hyps):
        ref = frozen_norm(r["text"])
        hyp = frozen_norm(h or "")
        if not hyp.strip():
            empties += 1
        e, n = wer_counts(ref, hyp)
        keys.append(ref); errs.append(e); refs.append(n)
    return keys, np.array(errs, dtype=np.int64), np.array(refs, dtype=np.int64), empties


def val_record(run_dir, step):
    """The trainer's own val.jsonl entry FOR THIS STEP (not simply the last line).

    Taking the last line was wrong for any run that validated past `step`: a resumed or continued run would have
    been compared against a later checkpoint's numbers while the hypotheses came from `step`.
    """
    p = Path(run_dir) / "val.jsonl"
    if not p.exists():
        raise SystemExit(f"REFUSING: {p} missing (no trainer record to check hypothesis ordering against)")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    hit = [r for r in recs if int(r.get("update", -1)) == int(step)]
    if len(hit) != 1:
        raise SystemExit(f"REFUSING: {p} has {len(hit)} records for update={step} (expected exactly 1)")
    return hit[0]


def check_ordering(name, run_dir, step, panel_wer_recomputed, empties_recomputed, strict=True):
    """The ordering proof: recomputed per-panel corpus WER must equal the trainer's to ORDER_TOL_PP.

    Returns the report dict; raises SystemExit on any mismatch.  `val.jsonl` stores wer_frozen rounded to 5 dp
    (= 0.0005 pp), so a genuine positional match lands two orders of magnitude inside the 0.005 pp tolerance and
    ANY real misalignment (a single swapped pair on a 1,500-row panel moves the WER by ~0.1 pp) blows through it.
    """
    if not strict:
        return {"checked": False}
    v = val_record(run_dir, step)
    out = {"checked": True, "tolerance_pp": ORDER_TOL_PP, "panels": {}}
    bad = []
    for p, w in panel_wer_recomputed.items():
        trainer = v["panels"][p]["wer_frozen"] * 100.0
        d = abs(w - trainer)
        out["panels"][p] = {"recomputed_pp": round(w, 4), "trainer_val_jsonl_pp": round(trainer, 4),
                            "abs_diff_pp": round(d, 5), "ok": bool(d <= ORDER_TOL_PP)}
        if d > ORDER_TOL_PP:
            bad.append(f"{p}: recomputed {w:.4f} pp vs trainer {trainer:.4f} pp (diff {d:.4f} pp)")
    te = v["panels"]["D"].get("empties")
    if te is not None:
        out["panels"]["D"]["empties_recomputed"] = empties_recomputed["D"]
        out["panels"]["D"]["empties_trainer"] = te
        if empties_recomputed["D"] != te:
            bad.append(f"D empties: recomputed {empties_recomputed['D']} vs trainer {te}")
    if bad:
        raise SystemExit(f"REFUSING to run: hypothesis/manifest ordering unverified for {name} ({run_dir}):\n  "
                         + "\n  ".join(bad))
    return out


def build(arm, control, step, manifests, strict=True):
    """Load both runs, verify ordering, and return the coupled cluster structure.

    Cluster ids are assigned from ONE dict shared by every panel, keyed on the frozen-surface reference string, so
    the same sentence in D and in M receives the same id and therefore the same bootstrap multiplicity.
    """
    cluster_id, data, wer_a, wer_c, emp_a, emp_c = {}, {}, {}, {}, {}, {}
    for p in WEIGHTS:
        keys, ea, ra, na = panel_counts(arm, p, step, manifests)
        keys_c, ec, rc, nc = panel_counts(control, p, step, manifests)
        if keys != keys_c:
            raise SystemExit(f"REFUSING: {p}: the two runs disagree on the panel's reference strings/row order")
        if not np.array_equal(ra, rc):
            raise SystemExit(f"REFUSING: {p}: reference word counts differ between runs (same manifest expected)")
        ids = np.array([cluster_id.setdefault(k, len(cluster_id)) for k in keys], dtype=np.int64)
        data[p] = (ids, ea, ec, ra)
        wer_a[p] = 100.0 * ea.sum() / max(ra.sum(), 1)
        wer_c[p] = 100.0 * ec.sum() / max(ra.sum(), 1)
        emp_a[p], emp_c[p] = na, nc
    proof = {"arm": check_ordering("arm", arm, step, wer_a, emp_a, strict),
             "control": check_ordering("control", control, step, wer_c, emp_c, strict)}
    return cluster_id, data, wer_a, wer_c, proof


def scorer(data, panels):
    """-> f(counts|None) = (Δ in pp, {panel: (WER_arm, WER_control)}) with corpus WER = Σ edits / Σ ref words."""
    def score(sel_counts=None):
        s, per = 0.0, {}
        for p in panels:
            ids, ea, ec, ra = data[p]
            w = 1.0 if sel_counts is None else sel_counts[ids]
            den = max(float((ra * w).sum()), 1e-9)
            wa = float((ea * w).sum()) / den
            wc = float((ec * w).sum()) / den
            per[p] = (wa * 100, wc * 100)
            s += WEIGHTS[p] * (wa - wc) * 100
        return s, per
    return score


def bootstrap(data, panels, n_clusters, draws, seed=SEED):
    score = scorer(data, panels)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for i in range(draws):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(np.float64)
        deltas[i], _ = score(counts)
    return deltas


# ---------------------------------------------------------------------------------------------------------------
# Self-test: a fixture in which ONE sentence appears in TWO panels, proving the coupling is real and load-bearing.
# ---------------------------------------------------------------------------------------------------------------
SHARED = "শেয়ার করা বাক্য এক"


def _selftest(tmpdir=None):
    """Prove (1) a sentence in two panels is ONE cluster, (2) it gets the SAME multiplicity in both panels on every
    draw, and (3) coupling is not cosmetic — the coupled Δ distribution is materially wider than the uncoupled one.

    (3) matters because (1) and (2) can both hold in an implementation that never actually shares variance (e.g. if
    the shared sentence carried no errors).  The fixture puts the shared sentence in D and M with correlated,
    non-trivial error counts, which is precisely the D∩M situation §8l describes for the MaCro/Common Voice prompt
    pool, so the two distributions must separate.
    """
    tmp = Path(tempfile.mkdtemp(prefix="admission_selftest_", dir=tmpdir))
    rng = np.random.default_rng(7)
    manifests, arm, ctl = {}, tmp / "arm", tmp / "ctl"
    (arm / "val_preds").mkdir(parents=True); (ctl / "val_preds").mkdir(parents=True)
    n_by_panel = {"D": 60, "M": 40, "K": 30, "R": 30}
    for p, n in n_by_panel.items():
        rows, hyp_a, hyp_c = [], [], []
        for i in range(n):
            # the shared sentence occupies row 0 of D and row 0 of M -> the coupling case; everything else unique
            if i == 0 and p in ("D", "M"):
                ref = SHARED
                ha, hc = "শেয়ার করা ভুল এক", "শেয়ার করা ভুল দুই"      # arm 1 error, control 2 errors, both panels
            else:
                ref = f"{p} বাক্য {i} ক খ গ ঘ"
                k = int(rng.integers(0, 3))
                ha = " ".join(ref.split()[:5 - k] + ["ভুল"] * k)
                hc = " ".join(ref.split()[:5 - min(k + 1, 4)] + ["ভুল"] * min(k + 1, 4))
            rows.append({"audio_filepath": f"{p}/{i}.wav", "text": ref, "duration": 3.0, "source": "fixture"})
            hyp_a.append(ha); hyp_c.append(hc)
        mp = tmp / f"panel_{p}.jsonl"
        mp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        manifests[p] = mp
        (arm / "val_preds" / f"{p}_step_1.json").write_text(json.dumps(hyp_a, ensure_ascii=False), encoding="utf-8")
        (ctl / "val_preds" / f"{p}_step_1.json").write_text(json.dumps(hyp_c, ensure_ascii=False), encoding="utf-8")

    cluster_id, data, wer_a, wer_c, _ = build(arm, ctl, 1, manifests, strict=False)
    total_rows = sum(n_by_panel.values())
    assert len(cluster_id) == total_rows - 1, f"expected one merged cluster, got {len(cluster_id)} of {total_rows}"
    sid = cluster_id[frozen_norm(SHARED)]
    assert data["D"][0][0] == sid and data["M"][0][0] == sid, "shared sentence did not get one id across D and M"
    assert data["D"][1][0] == 1 and data["M"][1][0] == 1, "fixture: shared row must carry a real arm error"
    assert data["D"][2][0] == 2 and data["M"][2][0] == 2, "fixture: shared row must carry a real control error"

    # (2) identical multiplicity in both panels, on every draw
    rng2 = np.random.default_rng(SEED)
    n = len(cluster_id)
    for _ in range(200):
        counts = np.bincount(rng2.integers(0, n, n), minlength=n).astype(np.float64)
        wD = counts[data["D"][0]]; wM = counts[data["M"][0]]
        assert wD[0] == wM[0] == counts[sid], "shared sentence resampled with different weights in D and M"

    # (3) coupled vs uncoupled: give every (panel,row) its own cluster id and compare the spread of Delta
    coupled = bootstrap(data, list(WEIGHTS), n, 4000)
    off = 0
    ind = {}
    for p in WEIGHTS:
        ids, ea, ec, ra = data[p]
        ind[p] = (np.arange(off, off + len(ids)), ea, ec, ra)
        off += len(ids)
    uncoupled = bootstrap(ind, list(WEIGHTS), off, 4000)
    sd_c, sd_u = float(coupled.std()), float(uncoupled.std())
    assert sd_c > sd_u, f"coupling did not widen the interval (coupled sd {sd_c:.4f} vs uncoupled {sd_u:.4f})"

    res = {"selftest": "PASS", "rows_total": total_rows, "clusters": len(cluster_id),
           "shared_sentence_in_panels": ["D", "M"], "shared_cluster_id": int(sid),
           "same_multiplicity_in_both_panels_over_draws": 200,
           "coupled_delta_sd_pp": round(sd_c, 5), "uncoupled_delta_sd_pp": round(sd_u, 5),
           "coupled_ci95": [round(float(np.percentile(coupled, 2.5)), 4), round(float(np.percentile(coupled, 97.5)), 4)],
           "uncoupled_ci95": [round(float(np.percentile(uncoupled, 2.5)), 4), round(float(np.percentile(uncoupled, 97.5)), 4)],
           "fixture_dir": str(tmp)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="run dir of the candidate arm")
    ap.add_argument("--control", default=str(REPO / "experiments/runs/O_A5000"),
                    help="SPEC §8j: O_ref = runs/O_A5000 is the admission comparator")
    ap.add_argument("--step", type=int, default=8000)
    ap.add_argument("--m", type=int, default=M_DEFAULT,
                    help="predeclared single-bucket arms for the simultaneous bound (SPEC §8m: 4)")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true", help="run the cross-panel coupling fixture and exit")
    ap.add_argument("--tmpdir", default=None, help="where --selftest writes its fixture (default: system temp)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest(a.tmpdir)
    if not a.arm:
        raise SystemExit("--arm is required")
    if a.m < 1:
        raise SystemExit("--m must be >= 1")
    if a.m != M_DEFAULT:
        print(f"WARNING: m={a.m} is not the registered m={M_DEFAULT} (SPEC §8m); this is not the registered verdict",
              file=sys.stderr)

    panels = list(WEIGHTS)
    cluster_id, data, wer_a, wer_c, order_proof = build(a.arm, a.control, a.step, PANEL_MANIFEST, strict=True)
    n_clusters = len(cluster_id)

    point_delta, per_panel = scorer(data, panels)()
    deltas = bootstrap(data, panels, n_clusters, a.draws)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    q = 1 - ALPHA / a.m
    sim_hi = np.percentile(deltas, 100 * q)            # EMPIRICAL quantile of the realised draws, not mean+z*sd

    va, vc = val_record(a.arm, a.step), val_record(a.control, a.step)
    Pa, Pc = va["panels"], vc["panels"]
    # ADMISSION guards: every one against the CONTROL (SPEC §5; §8k Q2 "admission against O_ref").
    guards = {
        "D_vs_control_pp": round((Pa["D"]["wer_frozen"] - Pc["D"]["wer_frozen"]) * 100, 3),
        "conversational_vs_control_pp": round((Pa["D"]["conversational"]["wer_frozen"]
                                               - Pc["D"]["conversational"]["wer_frozen"]) * 100, 3),
        "per_source_vs_control_pp": {k: round((Pa["D"]["per_source"][k] - Pc["D"]["per_source"][k]) * 100, 3)
                                     for k in Pa["D"]["per_source"]},
        "empty_rate": Pa["D"]["empty_rate"],
    }
    if set(Pa["D"]["per_source"]) != set(Pc["D"]["per_source"]):
        raise SystemExit("REFUSING: the two runs report different D sources; the per-source guard is undefined")
    guard_pass = bool(guards["D_vs_control_pp"] <= GUARD_D_PP
                      and guards["conversational_vs_control_pp"] <= GUARD_CONV_PP
                      and all(v <= GUARD_SOURCE_PP for v in guards["per_source_vs_control_pp"].values())
                      and guards["empty_rate"] <= GUARD_EMPTY)
    # The registered empty guard is D's (SPEC §2/§5, and the value recorded in §8j); the other panels are reported
    # so that a collapse confined to M/K/R cannot hide behind a clean D.
    empty_all = {p: Pa[p]["empty_rate"] for p in panels}

    decision = bool(point_delta <= DELTA_MAX_PP and hi < 0 and sim_hi < 0 and guard_pass)

    # INFORMATION ONLY — SPEC §8k Q2 continuation/final eligibility, which uses the FROZEN P1 baseline, not O_ref.
    cont = {"note": "SPEC §8k Q2: continuation/checkpoint eligibility uses the frozen P1 baseline, NOT O_ref; "
                    "this block is informational and does not enter ADMITTED"}
    if BASELINE_P1.exists():
        b = json.load(open(BASELINE_P1, encoding="utf-8"))["D"]
        cont["conversational_pp"] = round(Pa["D"]["conversational"]["wer_frozen"] * 100, 3)
        cont["conversational_limit_pp"] = P1_CONV_LIMIT_PP
        cont["conversational_ok"] = bool(cont["conversational_pp"] <= P1_CONV_LIMIT_PP)
        cont["per_source_vs_P1_pp"] = {k: round((Pa["D"]["per_source"][k] - b["per_source"][k]) * 100, 3)
                                       for k in Pa["D"]["per_source"] if k in b["per_source"]}
        cont["per_source_ok"] = bool(all(v <= P1_SOURCE_PP for v in cont["per_source_vs_P1_pp"].values()))
        cont["ELIGIBLE_FOR_CONTINUATION"] = bool(cont["conversational_ok"] and cont["per_source_ok"])

    out = {"arm": str(a.arm), "control": str(a.control), "step": a.step, "draws": a.draws, "seed": SEED,
           "n_sentence_clusters": n_clusters, "m": a.m,
           "S_arm_pp": round(va["S_pp"], 3), "S_control_pp": round(vc["S_pp"], 3),
           "delta_S_pp": round(point_delta, 3), "ci95": [round(lo, 3), round(hi, 3)],
           "simultaneous_quantile": round(q, 6), "simultaneous_upper_pp": round(sim_hi, 3),
           "per_panel_wer_arm_vs_control": {p: [round(x, 3), round(y, 3)] for p, (x, y) in per_panel.items()},
           "guards": guards, "guards_pass": guard_pass,
           "empty_rate_all_panels": empty_all,
           "hypothesis_ordering_proof": {"tolerance_pp": ORDER_TOL_PP,
                                         "arm": order_proof["arm"]["panels"],
                                         "control": order_proof["control"]["panels"]},
           "continuation_eligibility_vs_P1_INFO_ONLY": cont,
           "tests": {"delta<=-0.30": bool(point_delta <= DELTA_MAX_PP), "ci_upper<0": bool(hi < 0),
                     "simultaneous_upper<0": bool(sim_hi < 0), "guards": guard_pass},
           "ADMITTED": decision}
    s = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(s)
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
