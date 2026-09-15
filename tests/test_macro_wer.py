"""Tests that freeze the behaviour of bnasr/macro_wer.py, the headline scorer.

Every expected number in here is computed BY HAND in a comment next to the
assertion.  If an assertion fails, the scorer's contract moved -- re-derive the
published numbers, do not update the expectation to match the code.

ASCII words are used in most fixtures on purpose: metric_surface passes ASCII
letters through unchanged (verified against the frozen bn_normalize_v1), and a
reviewer can hand-count "a b" vs "a x" without a Bengali shaping engine.
Bengali fixtures appear exactly where Bengali behaviour is the point
(danda-only references normalizing to empty, punctuation stripping).

Run:  .venv/bin/python -m pytest tests/test_macro_wer.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "bnasr") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bnasr import macro_wer as mw  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared two-source fixture: macro and micro provably differ
# --------------------------------------------------------------------------- #

TWO_SOURCES = ("alpha", "beta")
TWO_PREFIXES = {"alpha": "spk:alpha:", "beta": "blk:beta:"}
TWO_UNITS = {"alpha": "speaker", "beta": "block"}


def two_source_fixture():
    """Three rows, two sources, one cluster each.

    alpha u1: ref "a b c d" hyp "a b c d"  -> E=0, R=4, wer 0
    alpha u2: ref "a b"     hyp "a x"      -> 1 substitution, E=1, R=2, wer 1/2
    beta  u3: ref 10 words, hyp misses the last word -> 1 deletion, E=1, R=10,
              wer 1/10

    Hand computation:
      w_alpha = (0 + 1/2) / 2          = 0.25
      w_beta  = (1/10) / 1             = 0.10
      W       = (0.25 + 0.10) / 2      = 0.175      (utterance-macro headline)
      micro   = (0+1+1) / (4+2+10)     = 2/16 = 0.125  (word-weighted)
    0.175 != 0.125: volume-weighting and source-balancing genuinely disagree
    on this fixture, so a scorer that silently computes micro cannot pass.
    """
    rows = [
        {"audio_filepath": "u1.wav", "text": "a b c d", "source": "alpha",
         "group": "spk:alpha:A"},
        {"audio_filepath": "u2.wav", "text": "a b", "source": "alpha",
         "group": "spk:alpha:A"},
        {"audio_filepath": "u3.wav", "text": "p q r s t u v w x y",
         "source": "beta", "group": "blk:beta:0"},
    ]
    hyps = ["a b c d", "a x", "p q r s t u v w x"]
    return rows, hyps


def score_two_source(**kw):
    rows, hyps = two_source_fixture()
    return mw.source_balanced_wer(
        rows, hyps,
        source_order=TWO_SOURCES,
        cluster_prefixes=TWO_PREFIXES,
        cluster_units=TWO_UNITS,
        bootstrap_resamples=kw.pop("bootstrap_resamples", 100),
        **kw,
    )


def test_macro_headline_differs_from_micro():
    doc = score_two_source()
    # Hand: W = 0.175 (see fixture docstring).
    assert doc["headline"]["source_balanced_utterance_macro_wer"] == pytest.approx(0.175, abs=1e-12)
    # Hand: corpus micro = 2/16 = 0.125, and it must be labeled word-weighted.
    assert doc["corpus_micro_word_weighted"]["wer"] == pytest.approx(0.125, abs=1e-12)
    assert doc["corpus_micro_word_weighted"]["errors"] == 2
    assert doc["corpus_micro_word_weighted"]["ref_words"] == 16
    # They must differ on this fixture -- that is the point of the fixture.
    assert doc["headline"]["source_balanced_utterance_macro_wer"] != \
        doc["corpus_micro_word_weighted"]["wer"]


def test_per_source_numbers():
    doc = score_two_source()
    by = {s["source"]: s for s in doc["sources"]}
    a, b = by["alpha"], by["beta"]
    # Hand: w_alpha = (0 + 0.5)/2 = 0.25; alpha micro = 1/6.
    assert a["utterance_macro_wer"] == pytest.approx(0.25, abs=1e-12)
    assert a["micro_wer"] == pytest.approx(1 / 6, abs=1e-12)
    assert (a["substitutions"], a["deletions"], a["insertions"]) == (1, 0, 0)
    assert a["ref_words"] == 6 and a["n_clusters"] == 1
    # Hand: w_beta = 0.1; beta micro = 1/10; the single error is a deletion.
    assert b["utterance_macro_wer"] == pytest.approx(0.1, abs=1e-12)
    assert b["micro_wer"] == pytest.approx(0.1, abs=1e-12)
    assert (b["substitutions"], b["deletions"], b["insertions"]) == (0, 1, 0)
    # Source order in the output is exactly the requested order.
    assert [s["source"] for s in doc["sources"]] == list(TWO_SOURCES)


def test_single_cluster_bootstrap_ci_is_degenerate():
    # Each source has exactly ONE cluster, so every bootstrap draw selects it
    # C_s=1 time and reproduces the point estimate exactly: CI == [W, W].
    doc = score_two_source(bootstrap_resamples=50)
    W = doc["headline"]["source_balanced_utterance_macro_wer"]
    assert doc["headline"]["ci95"] == [pytest.approx(W, abs=1e-15)] * 2
    by = {s["source"]: s for s in doc["sources"]}
    assert by["alpha"]["utterance_macro_wer_ci95"] == \
        [pytest.approx(0.25, abs=1e-15)] * 2


def test_bootstrap_is_deterministic():
    d1 = score_two_source(bootstrap_resamples=200)
    d2 = score_two_source(bootstrap_resamples=200)
    assert d1["headline"]["ci95"] == d2["headline"]["ci95"]
    assert [s["utterance_macro_wer_ci95"] for s in d1["sources"]] == \
        [s["utterance_macro_wer_ci95"] for s in d2["sources"]]


# --------------------------------------------------------------------------- #
# Bootstrap weighting: utterance occurrences, not mean-of-cluster-means
# --------------------------------------------------------------------------- #

def test_bootstrap_weights_by_utterance_occurrences():
    """One source, two clusters of unequal size.

    Cluster X: 1 utterance,  ref "a" hyp "b"  -> E=1, R=1, wer 1.0
    Cluster Y: 3 utterances, ref==hyp         -> wer 0.0 each
    Point estimate (hand): w = (1.0 + 0 + 0 + 0) / 4 = 0.25.

    A draw picks C_s=2 cluster ids with replacement.  Hand-enumerating the
    resampled mean over utterance OCCURRENCES:
      (X,X): sum E/R = 2.0 over 2 occurrences  -> 1.0
      (X,Y): sum E/R = 1.0 over 1+3 = 4        -> 0.25
      (Y,Y): 0 over 6                          -> 0.0
    A mean-of-cluster-means bootstrap would give 0.5 for the mixed draw.
    Therefore: every draw must be one of {0.0, 0.25, 1.0}, and 0.5 must never
    appear.  With 400 draws, P(no mixed draw) = 2^-400, so requiring 0.25 to
    appear cannot flake in practice.
    """
    utts = mw.build_utterances(
        [
            {"audio_filepath": "x1.wav", "text": "a", "source": "alpha",
             "group": "spk:alpha:X"},
            {"audio_filepath": "y1.wav", "text": "c c", "source": "alpha",
             "group": "spk:alpha:Y"},
            {"audio_filepath": "y2.wav", "text": "d d", "source": "alpha",
             "group": "spk:alpha:Y"},
            {"audio_filepath": "y3.wav", "text": "e e", "source": "alpha",
             "group": "spk:alpha:Y"},
        ],
        ["b", "c c", "d d", "e e"],
        source_order=("alpha",),
        cluster_prefixes={"alpha": "spk:alpha:"},
    )
    draws = mw.stratified_cluster_bootstrap(
        utts, source_order=("alpha",), resamples=400
    )
    vals = set(np.round(draws["alpha"], 12))
    assert vals <= {0.0, 0.25, 1.0}, f"unexpected draw values {vals}"
    assert 0.25 in vals, "mixed draw never appeared; astronomically unlikely"
    assert 0.5 not in vals, "0.5 means cluster-mean-of-means weighting"
    # overall == the single source's draws
    assert np.array_equal(draws["overall"], draws["alpha"])


# --------------------------------------------------------------------------- #
# score_utterance_wer
# --------------------------------------------------------------------------- #

def test_score_utterance_wer_counts():
    # "a b c" -> "a x c y": 1 substitution (b->x) + 1 insertion (y).
    ref_n, hyp_n, r, h, s, d, i = mw.score_utterance_wer("a b c", "a x c y")
    assert (ref_n, hyp_n) == ("a b c", "a x c y")
    assert (r, h, s, d, i) == (3, 4, 1, 0, 1)


def test_utterance_wer_may_exceed_one():
    # ref "a" hyp "a x y z": 3 insertions on a 1-word reference -> wer 3.0.
    # Insertions are real errors; clamping to 1.0 would hide hallucination.
    utts = mw.build_utterances(
        [{"audio_filepath": "u.wav", "text": "a", "source": "alpha",
          "group": "spk:alpha:A"}],
        ["a x y z"],
        source_order=("alpha",),
        cluster_prefixes={"alpha": "spk:alpha:"},
    )
    assert utts[0].insertions == 3
    assert utts[0].utterance_wer == pytest.approx(3.0)


def test_punctuation_is_stripped_before_alignment():
    # Frozen surface: metric_surface(text, False).  The danda and '!' vanish,
    # so these two differ by nothing.
    ref_n, hyp_n, r, h, s, d, i = mw.score_utterance_wer("কলা!", "কলা।")
    assert ref_n == hyp_n == "কলা"
    assert (s, d, i) == (0, 0, 0)


def test_score_utterance_wer_refuses_empty_reference():
    # A danda-only reference normalizes to zero words.  The function must
    # refuse -- report-and-drop is the CALLER's job, and returning counts here
    # would let them be averaged in with a fabricated denominator.
    with pytest.raises(mw.MacroWerAbort, match="empty_reference"):
        mw.score_utterance_wer("।।।", "কলা")


# --------------------------------------------------------------------------- #
# Empty-reference report-and-drop
# --------------------------------------------------------------------------- #

def test_empty_reference_reported_and_dropped():
    rows = [
        {"audio_filepath": "u1.wav", "text": "a b", "source": "alpha",
         "group": "spk:alpha:A"},
        # danda-only reference -> normalizes to "" -> must be dropped, and its
        # 2 hypothesis words recorded as would-be insertions, NOT scored with
        # a fabricated R=1.
        {"audio_filepath": "u2.wav", "text": "।।।", "source": "alpha",
         "group": "spk:alpha:A"},
    ]
    hyps = ["a b", "x y"]
    utts = mw.build_utterances(
        rows, hyps, source_order=("alpha",),
        cluster_prefixes={"alpha": "spk:alpha:"},
    )
    dropped = utts[1]
    assert dropped.included is False
    assert dropped.utterance_wer is None
    assert dropped.exclusion_reason == "empty_reference_after_metric_surface"
    assert dropped.hyp_words == 2          # the would-be insertions
    assert dropped.errors == 0             # nothing fabricated

    doc = mw.source_balanced_wer(
        rows, hyps, source_order=("alpha",),
        cluster_prefixes={"alpha": "spk:alpha:"},
        cluster_units={"alpha": "speaker"},
        bootstrap_resamples=10,
    )
    # Hand: only u1 is scored, perfectly -> W = 0; the drop is visible in the
    # counts, and micro denominators exclude the dropped row entirely.
    assert doc["headline"]["source_balanced_utterance_macro_wer"] == 0.0
    assert doc["headline"]["n_utterances_manifest"] == 2
    assert doc["headline"]["n_utterances_scored"] == 1
    assert doc["headline"]["empty_references_dropped"] == 1
    assert doc["sources"][0]["empty_references_dropped"] == 1
    assert doc["corpus_micro_word_weighted"]["ref_words"] == 2


def test_predictions_jsonl_records_would_be_insertions(tmp_path):
    rows = [
        {"audio_filepath": "u1.wav", "text": "a b", "source": "alpha",
         "group": "spk:alpha:A"},
        {"audio_filepath": "u2.wav", "text": "।", "source": "alpha",
         "group": "spk:alpha:A"},
    ]
    utts = mw.build_utterances(
        rows, ["a b", "x y z"], source_order=("alpha",),
        cluster_prefixes={"alpha": "spk:alpha:"},
    )
    out = tmp_path / "pred.jsonl"
    sha = mw.write_predictions_jsonl(out, utts)
    recs = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(recs) == 2 and len(sha) == 64
    assert recs[1]["included"] is False
    assert recs[1]["would_be_insertions"] == 3
    assert recs[1]["utterance_wer"] is None


def test_abort_when_drop_empties_a_source():
    # Every beta reference normalizes to empty -> n_beta would be 0.  Averaging
    # W over a phantom source is meaningless; the scorer must abort.
    rows = [
        {"audio_filepath": "u1.wav", "text": "a", "source": "alpha",
         "group": "spk:alpha:A"},
        {"audio_filepath": "u2.wav", "text": "।", "source": "beta",
         "group": "blk:beta:0"},
    ]
    with pytest.raises(mw.MacroWerAbort, match="zero scoreable"):
        mw.source_balanced_wer(
            rows, ["a", ""],
            source_order=TWO_SOURCES,
            cluster_prefixes=TWO_PREFIXES,
            bootstrap_resamples=10,
        )


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #

def test_abort_on_count_mismatch():
    rows, hyps = two_source_fixture()
    with pytest.raises(mw.MacroWerAbort, match="hypothesis count"):
        mw.build_utterances(rows, hyps[:-1], source_order=TWO_SOURCES,
                            cluster_prefixes=TWO_PREFIXES)


def test_abort_on_unknown_source():
    rows, hyps = two_source_fixture()
    rows[1] = dict(rows[1], source="gamma")
    with pytest.raises(mw.MacroWerAbort, match="source 'gamma'"):
        mw.build_utterances(rows, hyps, source_order=TWO_SOURCES,
                            cluster_prefixes=TWO_PREFIXES)


def test_abort_on_wrong_cluster_prefix():
    rows, hyps = two_source_fixture()
    # A beta row wearing an alpha-style speaker key: the bootstrap would use
    # the wrong independence unit, so this must die, not warn.
    rows[2] = dict(rows[2], group="spk:alpha:Z")
    with pytest.raises(mw.MacroWerAbort, match="cluster key"):
        mw.build_utterances(rows, hyps, source_order=TWO_SOURCES,
                            cluster_prefixes=TWO_PREFIXES)


def test_abort_on_missing_cluster_key():
    rows, hyps = two_source_fixture()
    del rows[0]["group"]
    with pytest.raises(mw.MacroWerAbort, match="cluster key"):
        mw.build_utterances(rows, hyps, source_order=TWO_SOURCES,
                            cluster_prefixes=TWO_PREFIXES)


def test_abort_on_source_without_declared_prefix():
    rows, hyps = two_source_fixture()
    with pytest.raises(mw.MacroWerAbort, match="prefix declared"):
        mw.build_utterances(rows, hyps, source_order=TWO_SOURCES,
                            cluster_prefixes={"alpha": "spk:alpha:"})


# --------------------------------------------------------------------------- #
# Output document shape
# --------------------------------------------------------------------------- #

def test_schema_top_level_and_serializability():
    doc = score_two_source()
    assert doc["schema"] == "bengali-asr-source-balanced-wer/1"
    assert set(doc) == {
        "schema", "generated_utc", "instrument", "model", "decoding",
        "metric", "bootstrap", "headline", "sources",
        "corpus_micro_word_weighted", "artifacts", "integrity",
    }
    assert doc["metric"]["normalizer"] == "bn_normalize_v1"
    assert doc["metric"]["normalizer_sha256"] == (
        "238364e725be1143132e8e3b32c964ba3baceb229cf3f9bca6f368f8bad1b436"
    )
    assert doc["metric"]["empty_reference_rule"] == "report_and_drop"
    assert doc["bootstrap"]["seed"] == 20260822
    assert all(doc["integrity"].values())
    # The contract forbids NaN/Infinity anywhere in the document.
    json.dumps(doc, ensure_ascii=False, allow_nan=False)


def test_frozen_source_order_constant():
    # The release headline is defined over exactly these five, in this order.
    assert mw.SOURCE_ORDER == (
        "shrutilipi", "subakko", "indicvoicesr", "openslr53", "fleurs_train",
    )
    assert mw.BOOTSTRAP_RESAMPLES == 10_000
    assert mw.BOOTSTRAP_SEED == 20260822


# --------------------------------------------------------------------------- #
# Real source panel (skips cleanly when the manifest is absent)
# --------------------------------------------------------------------------- #

PANEL = REPO_ROOT / "data2" / "source_panel.json"


@pytest.mark.skipif(not PANEL.is_file(), reason="source_panel.json not present")
def test_real_panel_perfect_hypotheses_score_zero():
    """Feed the panel its own references: W must be exactly 0 and every one of
    the 4,491 rows must survive cluster validation (the panel's group keys are
    the production bootstrap units -- this is the integration check that the
    scorer accepts them all)."""
    rows = [json.loads(l) for l in open(PANEL, encoding="utf-8")]
    doc = mw.source_balanced_wer(
        rows, [r["text"] for r in rows], bootstrap_resamples=25,
    )
    assert doc["headline"]["source_balanced_utterance_macro_wer"] == 0.0
    assert doc["headline"]["ci95"] == [0.0, 0.0]
    assert doc["headline"]["n_utterances_manifest"] == 4491
    assert doc["headline"]["empty_references_dropped"] == 0
    by = {s["source"]: s for s in doc["sources"]}
    # Cluster counts pinned by the panel build: 40/29/100/50/491.
    assert by["shrutilipi"]["n_clusters"] == 40
    assert by["subakko"]["n_clusters"] == 29
    assert by["indicvoicesr"]["n_clusters"] == 100
    assert by["openslr53"]["n_clusters"] == 50
    assert by["fleurs_train"]["n_clusters"] == 491
