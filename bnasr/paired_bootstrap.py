#!/usr/bin/env python3
"""Paired sentence-clustered bootstrap between two hypothesis files (SPEC §5, §6).

Why this shape of test, and no other
------------------------------------
Every this recipe comparison that can stop or start a run is a PAIRED, CLUSTERED
interval on a DIFFERENCE of corpus WERs. Three design choices are load-bearing
and each one exists because the obvious alternative is wrong here:

1. **Paired, not two independent intervals.** The pilot gate (SPEC §5) and the
   headline (SPEC §6) both ask whether system A is better than system B *on the
   same utterances*. Two independent CIs that overlap are perfectly compatible
   with A beating B on every single row; the question "do the CIs overlap" is
   not the question SPEC asks. So each replicate draws ONE index set and scores
   BOTH systems on it — `resample_indices()` is called once per draw and the
   same array is used twice.

2. **Clustered on the reference sentence, not on the recording.** FLEURS test
   is 920 recordings of 349 distinct sentences (SPEC §2/§6), panel R is 1,500
   recordings of 494 sentences (SPEC §2). Several speakers read the same
   sentence, and a model that fails a sentence tends to fail every recording of
   it, so recordings are not independent draws. Resampling recordings would
   pretend there are 920 independent units where there are 349, and the
   interval comes out too narrow — which biases every decision in this SPEC
   towards "significant". Whole clusters are resampled with replacement and
   EVERY recording of a sampled cluster is included (council B #140/#143,
   council A "retaining all recordings in each selected cluster").
   The cluster key is the FROZEN-SURFACE reference string: it is the surface
   selection is defined on (SPEC §2) and two rows that normalise to the same
   string are the same sentence for scoring purposes, which is exactly the
   dependence being modelled.

3. **Corpus WER inside each replicate, i.e. Σedits / Σref-words.** Not the mean
   of per-utterance rates. WER is a ratio of totals; averaging rates reweights a
   3-word Common Voice row like a 30-word IndicVoices extempore row. SPEC §6
   spells this out ("corpus WER = Σedits/Σref words").

Alignment is BY KEY, never by position
--------------------------------------
The two hypothesis files come from two different runs of
``bnasr/panel_eval.py --save-hyps`` (or, for the headline, from the frozen
benchmark predictions), and nothing guarantees they iterate a panel in the same
order — a resumed decode, a rebuilt manifest or a plain dict round-trip is
enough to permute them. So rows are joined on the utterance key
(``audio_filepath`` / ``file``) and a key present in the panel but missing from
either system ABORTS. Silently dropping it would compare the two systems on
different denominators and call the difference a model effect; filling it with
an empty hypothesis would charge one system for a bookkeeping failure.

Registered numbers (SPEC §5, §6)
--------------------------------
10,000 draws for the pilot decisions, 20,000 for the headline, seed 20260905,
two-sided 95 % percentile interval, decision = "CI upper bound < 0".

Usage
-----
    bnasr/paired_bootstrap.py \
        --panel-manifest data2/val_corpus.json --panel D \
        --hyps-a experiments/eval/armN.hyps.json \
        --hyps-b experiments/eval/armO.hyps.json \
        --out experiments/eval/N_vs_O_D.json

    # self-test on the fixtures (no GPU, no model, seconds)
    bnasr/paired_bootstrap.py --selftest
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import (  # noqa: E402
    anonymize_process, cer_counts, frozen_norm, load_manifest, sha256_file,
    wer_counts,
)

anonymize_process()

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants — transcriptions of SPEC.md, asserted next to the cite.
# ---------------------------------------------------------------------------

# SPEC §5: "paired sentence-clustered 95 % CI upper bound < 0 (10,000 draws,
# seed 20260905)". SPEC §6: "20,000 draws, seed 20260905".
DRAWS_PILOT = 10000
DRAWS_FINAL = 20000
SEED = 20260905
assert DRAWS_PILOT == 10000, "SPEC §5: 10,000 draws for pilot decisions"
assert DRAWS_FINAL == 20000, "SPEC §6: 20,000 draws for the headline"
assert SEED == 20260905, "SPEC §5/§6: seed 20260905"

# Two-sided 95 % percentile interval. SPEC §5/§6 state "95 % CI"; the decision
# uses only the UPPER endpoint, but the lower one is reported because an
# interval quoted by one endpoint is not an interval.
CI_ALPHA = 0.05
assert abs(CI_ALPHA - 0.05) < 1e-12, "SPEC §5/§6: 95 % CI"

# SPEC §2 registers the cluster counts these panels must produce. Checked, not
# assumed: a panel that has been rebuilt with different text normalisation
# would silently change the number of independent units the interval is built
# from, which changes its width without changing a single hypothesis.
EXPECTED_CLUSTERS = {"D": None, "M": None, "R": 494, "K": None, "FLEURS_test": 349}
assert EXPECTED_CLUSTERS["FLEURS_test"] == 349, "SPEC §6: 349 sentence clusters"
assert EXPECTED_CLUSTERS["R"] == 494, "SPEC §2: panel R = 494 distinct sentences"

# The sealed test audio. This tool reads TEXT only (hypothesis JSON), never
# audio, but the guard is kept so a future caller cannot point --panel-manifest
# at the sealed directory's manifest and have it silently work.
SEALED_TEST_DIR = Path(os.environ.get("BNASR_FLEURS_DIR", "/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn")).resolve()


def load_hyps(path, panel=None):
    """Read a hypothesis file into {key: hypothesis}.

    Three shapes are accepted, because three different producers exist and
    normalising them here is cheaper than a converter script per producer:

    * ``panel_eval.py --save-hyps``: ``{"panels": {P: {path: hyp}}}``
    * a bare mapping ``{key: hypothesis}``
    * the benchmark's ``predictions.json``: a LIST of rows with ``file`` and
      ``transcript`` (used by the headline against frozen the baseline/2 hypotheses)

    A list is de-duplicated strictly: two rows for one key are an error, not a
    last-one-wins, because "which duplicate won" is invisible in the result.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = {"path": str(path), "sha256": sha256_file(path)}
    if isinstance(obj, dict) and "panels" in obj:
        panels = obj["panels"]
        if panel is None:
            if len(panels) != 1:
                raise SystemExit(
                    f"{path} holds panels {sorted(panels)}; pass --panel to say "
                    f"which one to compare")
            panel = next(iter(panels))
        if panel not in panels:
            raise SystemExit(f"{path} has no panel {panel!r} (has {sorted(panels)})")
        meta["ckpt_sha256"] = obj.get("ckpt_sha256")
        meta["panel"] = panel
        return dict(panels[panel]), meta
    if isinstance(obj, list):
        out = {}
        for r in obj:
            key = r.get("file") or r.get("audio_filepath")
            if key is None:
                raise SystemExit(f"{path}: row without 'file'/'audio_filepath'")
            if key in out:
                raise SystemExit(f"{path}: duplicate key {key!r}")
            out[key] = r.get("transcript", r.get("text", "")) or ""
        meta["rows"] = len(out)
        return out, meta
    if isinstance(obj, dict):
        return {k: (v or "") for k, v in obj.items()}, meta
    raise SystemExit(f"{path}: unrecognised hypothesis file shape")


def build_rows(panel_rows, hyps_a, hyps_b, key_field="audio_filepath",
               ref_field="text"):
    """Join the panel against both systems BY KEY. A missing key aborts.

    Returns (keys, refs, hyp_a, hyp_b). The panel manifest — not either
    hypothesis file — defines the row set and the references, so a system
    cannot change what it is scored on by shipping a shorter file.
    """
    keys, refs, ha, hb = [], [], [], []
    missing_a, missing_b = [], []
    seen = set()
    for r in panel_rows:
        k = r[key_field]
        if k in seen:
            raise SystemExit(
                f"panel has duplicate key {k!r}; the join is by key and a "
                f"duplicate would score one row twice")
        seen.add(k)
        if k not in hyps_a:
            missing_a.append(k)
        if k not in hyps_b:
            missing_b.append(k)
        if missing_a or missing_b:
            continue
        keys.append(k)
        refs.append(r[ref_field])
        ha.append(hyps_a[k])
        hb.append(hyps_b[k])
    if missing_a or missing_b:
        raise SystemExit(
            f"ABORT: {len(missing_a)} prediction(s) missing from A "
            f"(e.g. {missing_a[:3]}) and {len(missing_b)} from B "
            f"(e.g. {missing_b[:3]}). A paired test over a subset of the panel "
            f"is a different test; fix the hypothesis file, do not drop rows.")
    return keys, refs, ha, hb


def per_row_counts(refs, hyps):
    """Frozen-surface (word errors, ref words) per row, plus char counts.

    The frozen surface is applied to BOTH sides here and nowhere else, so the
    two systems can never be scored through different normalisers.
    """
    we, wn, ce, cn = [], [], [], []
    for ref, hyp in zip(refs, hyps):
        r, h = frozen_norm(ref), frozen_norm(hyp)
        e, n = wer_counts(r, h)
        c, m = cer_counts(r, h)
        we.append(e)
        wn.append(n)
        ce.append(c)
        cn.append(m)
    return (np.asarray(we, dtype=np.int64), np.asarray(wn, dtype=np.int64),
            np.asarray(ce, dtype=np.int64), np.asarray(cn, dtype=np.int64))


def cluster_blocks(refs):
    """Row indices grouped by frozen-surface reference string.

    Sorted by key so the block ORDER — and therefore the sequence of blocks the
    RNG's integers index into — is a function of the data alone. Without the
    sort the result would depend on dict insertion order, i.e. on manifest row
    order, and a "seeded, reproducible" interval that moves when a manifest is
    re-sorted is not reproducible.
    """
    buckets = defaultdict(list)
    for i, r in enumerate(refs):
        buckets[frozen_norm(r)].append(i)
    return [np.asarray(buckets[k], dtype=np.int64) for k in sorted(buckets)]


def paired_bootstrap(err_a, err_b, n_ref, refs, draws, seed=SEED):
    """Paired sentence-clustered bootstrap of (WER_a − WER_b) in percentage points.

    One resample per draw, used for BOTH systems — that is the pairing. k
    clusters are drawn with replacement from k clusters, every recording of a
    drawn cluster is included, and each system's WER in that replicate is
    Σedits/Σref-words over the drawn rows.
    """
    blocks = cluster_blocks(refs)
    k = len(blocks)
    if k < 2:
        raise SystemExit(f"only {k} distinct reference cluster(s); a clustered "
                         f"bootstrap over one cluster has no variance")
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        sel = np.concatenate([blocks[j] for j in rng.integers(0, k, k)])
        denom = max(int(n_ref[sel].sum()), 1)
        deltas[d] = (int(err_a[sel].sum()) - int(err_b[sel].sum())) / denom
    deltas.sort()
    # Percentile interval by index, the same convention as tools/p3_final_test.py
    # (which produced the Phase-3 numbers) so this recipe intervals are comparable.
    lo = float(deltas[int((CI_ALPHA / 2) * draws)])
    hi = float(deltas[int((1 - CI_ALPHA / 2) * draws)])
    wer_a = float(err_a.sum() / max(n_ref.sum(), 1))
    wer_b = float(err_b.sum() / max(n_ref.sum(), 1))
    return {
        "n_rows": int(len(err_a)),
        "n_clusters": k,
        "wer_a_pp": round(wer_a * 100, 4),
        "wer_b_pp": round(wer_b * 100, 4),
        "delta_pp_a_minus_b": round((wer_a - wer_b) * 100, 4),
        "ci95_pp": [round(lo * 100, 4), round(hi * 100, 4)],
        "p_a_better": float((deltas < 0).mean()),
        # THE SPEC DECISION. SPEC §5 ("paired sentence-clustered 95 % CI upper
        # bound < 0") and SPEC §6 ("the 95 % CI of (P4 − P2) lies entirely below
        # 0") are the same rule; A is the candidate, B the incumbent.
        "ci_upper_below_zero": bool(hi < 0.0),
        "draws": draws, "seed": seed,
        "cluster_key": "frozen_norm(reference)",
        "wer_definition": "corpus WER = sum(edits)/sum(ref words) per draw",
    }


def compare(panel_manifest, hyps_a_path, hyps_b_path, draws, panel=None,
            seed=SEED, label_a="A", label_b="B", key_field="audio_filepath",
            ref_field="text"):
    """Full comparison record for one panel: point deltas, CI, CER, provenance."""
    mpath = Path(panel_manifest)
    if SEALED_TEST_DIR in mpath.resolve().parents:
        raise SystemExit(f"refusing to read from the sealed test directory: {mpath}")
    panel_rows = load_manifest(mpath)
    ha, meta_a = load_hyps(hyps_a_path, panel)
    hb, meta_b = load_hyps(hyps_b_path, panel)
    keys, refs, sa, sb = build_rows(panel_rows, ha, hb, key_field, ref_field)

    ea, na, ca, ma = per_row_counts(refs, sa)
    eb, nb, cb, mb = per_row_counts(refs, sb)
    assert (na == nb).all(), (
        "reference word counts differ between the systems — the two sides were "
        "not scored against the same references")

    res = paired_bootstrap(ea, eb, na, refs, draws, seed)
    expect = EXPECTED_CLUSTERS.get(panel or "")
    res["clusters_registered"] = expect
    res["clusters_match_spec"] = None if expect is None else (res["n_clusters"] == expect)
    res.update({
        "cer_a_pp": round(float(ca.sum() / max(ma.sum(), 1)) * 100, 4),
        "cer_b_pp": round(float(cb.sum() / max(mb.sum(), 1)) * 100, 4),
        "empty_a": int(sum(1 for h in sa if not h.strip())),
        "empty_b": int(sum(1 for h in sb if not h.strip())),
    })
    return {
        "tool": "bnasr/paired_bootstrap.py",
        "spec": "experiments/SPEC.md §5 (pilot gate) / §6 (headline)",
        "panel": panel,
        "panel_manifest": {"path": str(mpath), "sha256": sha256_file(mpath),
                           "rows": len(panel_rows)},
        "system_a": dict(label=label_a, **meta_a),
        "system_b": dict(label=label_b, **meta_b),
        "scorer": {p.name: sha256_file(p) for p in
                   [REPO / "bnasr/scoring.py", Path(__file__).resolve()]},
        "result": res,
    }


# ---------------------------------------------------------------------------
# Self-test. Four properties, each one a way this tool could be silently wrong.
# ---------------------------------------------------------------------------

FIXTURES = REPO / "experiments/fixtures"


def _write_fixture_panel(path, n=120):
    """A synthetic panel with real cluster structure: 40 sentences × 3 speakers.

    Deterministic and text-only — no audio is referenced, and nothing here goes
    near a real manifest, so the self-test runs anywhere in seconds.
    """
    rng = np.random.default_rng(7)
    vocab = ["আমি", "তুমি", "সে", "বই", "পড়ি", "খাই", "ভালো", "আজ", "কাল", "ঘর"]
    rows = []
    for s in range(40):
        words = [vocab[int(i)] for i in rng.integers(0, len(vocab), 8)]
        text = " ".join(words)
        for spk in range(3):
            rows.append({"audio_filepath": f"synthetic/s{s:03d}_spk{spk}.wav",
                         "text": text, "duration": 4.0, "source": "fixture"})
    assert len(rows) == n, f"{len(rows)} != {n}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return rows


def selftest():
    """Run the four fixture properties. Returns a report dict; raises on failure."""
    panel = FIXTURES / "bootstrap_panel.jsonl"
    rows = _write_fixture_panel(panel)

    # System A: a fixed, mediocre transcriber — drop the 3rd word of every row.
    hyp_a = {r["audio_filepath"]: " ".join(r["text"].split()[:2] + r["text"].split()[3:])
             for r in rows}
    # System B: identical to A (property 1) …
    hyp_b_same = dict(hyp_a)
    # … and system C: A plus a KNOWN injected error on every recording of the
    # first 20 sentences — a large, systematic, cluster-aligned degradation that
    # a correct paired test must find significant.
    hyp_c = {}
    bad_sentences = {r["text"] for r in rows[:60]}
    for r in rows:
        w = hyp_a[r["audio_filepath"]].split()
        if r["text"] in bad_sentences:
            w = ["ভুল"] * len(w)          # every word wrong
        hyp_c[r["audio_filepath"]] = " ".join(w)

    (FIXTURES / "bootstrap_hyps_a.json").write_text(
        json.dumps({"panels": {"FIX": hyp_a}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (FIXTURES / "bootstrap_hyps_same.json").write_text(
        json.dumps({"panels": {"FIX": hyp_b_same}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (FIXTURES / "bootstrap_hyps_worse.json").write_text(
        json.dumps({"panels": {"FIX": hyp_c}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # A key-order permutation of A. Position-based alignment would silently
    # compare row i of one file with row j of the other; a by-key join must be
    # bit-identical to the unshuffled case.
    shuffled = dict(sorted(hyp_a.items(), key=lambda kv: kv[0][::-1]))
    (FIXTURES / "bootstrap_hyps_a_shuffled.json").write_text(
        json.dumps({"panels": {"FIX": shuffled}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # A truncated file: one prediction missing. Must abort.
    trunc = {k: v for k, v in list(hyp_a.items())[:-1]}
    (FIXTURES / "bootstrap_hyps_a_missing.json").write_text(
        json.dumps({"panels": {"FIX": trunc}}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    checks = {}

    # (1) identical systems -> Δ exactly 0, CI containing 0, decision False.
    r1 = compare(panel, FIXTURES / "bootstrap_hyps_a.json",
                 FIXTURES / "bootstrap_hyps_same.json", DRAWS_PILOT, panel="FIX")
    res1 = r1["result"]
    assert res1["delta_pp_a_minus_b"] == 0.0, res1
    assert res1["ci95_pp"] == [0.0, 0.0], res1
    assert res1["ci_upper_below_zero"] is False, res1
    assert res1["n_clusters"] == 40, res1
    checks["identical_systems"] = res1

    # (2) known injected error set -> CI excludes 0 and lies entirely BELOW 0
    #     when the better system is A (the candidate).
    r2 = compare(panel, FIXTURES / "bootstrap_hyps_a.json",
                 FIXTURES / "bootstrap_hyps_worse.json", DRAWS_PILOT, panel="FIX")
    res2 = r2["result"]
    assert res2["delta_pp_a_minus_b"] < 0, res2
    assert res2["ci95_pp"][1] < 0, res2
    assert res2["ci_upper_below_zero"] is True, res2
    # …and the sign convention is not accidental: swapping the arguments must
    # flip the decision, otherwise the tool would declare any difference a win.
    r2b = compare(panel, FIXTURES / "bootstrap_hyps_worse.json",
                  FIXTURES / "bootstrap_hyps_a.json", DRAWS_PILOT, panel="FIX")
    assert r2b["result"]["ci_upper_below_zero"] is False, r2b["result"]
    assert r2b["result"]["ci95_pp"][0] > 0, r2b["result"]
    checks["injected_error_set"] = res2
    checks["sign_convention_swapped"] = r2b["result"]

    # (3) shuffled hypothesis order must give a BIT-IDENTICAL result.
    r3 = compare(panel, FIXTURES / "bootstrap_hyps_a_shuffled.json",
                 FIXTURES / "bootstrap_hyps_worse.json", DRAWS_PILOT, panel="FIX")
    assert r3["result"] == res2, "by-key alignment failed under a key permutation"
    checks["shuffled_order_identical"] = True

    # (4) a missing prediction must ABORT, not be dropped.
    try:
        compare(panel, FIXTURES / "bootstrap_hyps_a_missing.json",
                FIXTURES / "bootstrap_hyps_worse.json", DRAWS_PILOT, panel="FIX")
    except SystemExit as exc:
        checks["missing_prediction_aborts"] = str(exc)[:160]
    else:
        raise AssertionError("a missing prediction did not abort")

    # (5) determinism: the same inputs and seed twice give the same interval.
    r5 = compare(panel, FIXTURES / "bootstrap_hyps_a.json",
                 FIXTURES / "bootstrap_hyps_worse.json", DRAWS_PILOT, panel="FIX")
    assert r5["result"]["ci95_pp"] == res2["ci95_pp"], "not deterministic"
    checks["deterministic"] = True

    return {"tool": "bnasr/paired_bootstrap.py --selftest",
            "fixtures_dir": str(FIXTURES), "checks": checks}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-manifest")
    ap.add_argument("--panel", default=None,
                    help="panel name inside a --save-hyps file (D/M/R/K/holdout)")
    ap.add_argument("--hyps-a", help="CANDIDATE system (the one that must win)")
    ap.add_argument("--hyps-b", help="INCUMBENT system")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--draws", type=int, default=DRAWS_PILOT,
                    help=f"SPEC §5 pilots {DRAWS_PILOT}; SPEC §6 headline {DRAWS_FINAL}")
    ap.add_argument("--final", action="store_true",
                    help=f"use the SPEC §6 headline draw count ({DRAWS_FINAL})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--key-field", default="audio_filepath")
    ap.add_argument("--ref-field", default="text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print(json.dumps(selftest(), indent=1, ensure_ascii=False))
        return
    if not (a.panel_manifest and a.hyps_a and a.hyps_b):
        raise SystemExit("--panel-manifest, --hyps-a and --hyps-b are required")
    draws = DRAWS_FINAL if a.final else a.draws
    rep = compare(a.panel_manifest, a.hyps_a, a.hyps_b, draws, panel=a.panel,
                  seed=a.seed, label_a=a.label_a, label_b=a.label_b,
                  key_field=a.key_field, ref_field=a.ref_field)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
