"""Source-balanced utterance-macro WER: the headline scorer for the final model.

Why this exists next to src/evaluation.py instead of inside it: evaluation.py
aggregates corpus-micro (total errors / total reference words), which lets the
5,471-row shrutilipi slice of a manifest outvote the 96-row fleurs slice ~57:1.
The council froze a different headline for the release: weight every SOURCE
equally, and inside a source weight every UTTERANCE equally,

    W = (1/S) * sum_s (1/n_s) * sum_u E_u / R_u

so a model that only learned newsreader shrutilipi cannot hide behind volume.
Corpus-micro is still reported, but always labeled word-weighted, never as the
headline.  evaluation.py is deliberately left untouched: its numbers are in
already-published tables and must keep reproducing them.

Contract highlights (all council-frozen, see the pillar-2 report):
  * Text surface is metric_surface(text, False) from the frozen bn_normalize_v1
    (sha256 238364e725be1143132e8e3b32c964ba3baceb229cf3f9bca6f368f8bad1b436),
    applied identically to reference and hypothesis; words are str.split().
  * E_u = S_u + D_u + I_u from a jiwer word alignment.  Per-utterance WER may
    exceed 1.0 because insertions are real errors.
  * A reference that normalizes to zero words is REPORTED AND DROPPED:
    utterance_wer null, included false, would_be_insertions recorded.  R_u is
    never replaced with a fabricated 1 -- that would invent a denominator and
    silently reward deleting the hypothesis.  If dropping empties a source the
    scorer aborts rather than averaging over a phantom.
  * Uncertainty comes from a 10,000-draw source-stratified CLUSTER bootstrap
    (seed 20260822, numpy default_rng/PCG64).  Utterance-level resampling would
    pretend the 40 shrutilipi stream blocks are 1,000 independent samples;
    clusters are the panel's `group` keys (speakers / stream blocks / sentence
    ids), validated per source, never inferred here.

Every alignment is cross-checked against an independent O(R*H) Levenshtein
before anything is reported; a single disagreement aborts the run.  The two
silent-data-loss incidents in this project's history (partial manifest read as
"split done", truncated wav read as "converted") are why nothing in this file
trusts a single code path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bnasr.normalize import (
    NORMALIZER_VERSION,
    UNICODE_VERSION,
    metric_surface,
    source_sha256,
)

SCHEMA = "bengali-asr-source-balanced-wer/1"

# Frozen five-source order for the release headline.  Reordering would change
# nothing numerically (the mean is symmetric) but would break byte-comparison
# of results.json files, so it is pinned.
SOURCE_ORDER = (
    "shrutilipi",
    "subakko",
    "indicvoicesr",
    "openslr53",
    "fleurs_train",
)

# Cluster-key prefixes as emitted by tools/build_source_panel/groups.py.  The
# scorer validates rather than constructs: a wrong or missing key means the
# manifest was built by something else and the bootstrap would silently use the
# wrong independence units.
CLUSTER_PREFIXES: Mapping[str, str] = {
    "shrutilipi": "blk:shrutilipi:",
    "subakko": "spk:subakko:",
    "indicvoicesr": "spk:indicvoicesr:",
    "openslr53": "spk:openslr53:",
    "fleurs_train": "text:",
}

CLUSTER_UNITS: Mapping[str, str] = {
    "shrutilipi": "stream_index_block_500",
    "subakko": "speaker",
    "indicvoicesr": "speaker_id",
    "openslr53": "speaker",
    "fleurs_train": "sentence_id",
}

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260822
EMPTY_REF_REASON = "empty_reference_after_metric_surface"


class MacroWerAbort(RuntimeError):
    """Raised for any contract violation.  Never caught internally: a scoring
    run that cannot satisfy the frozen contract must die loudly, not emit a
    plausible-looking number."""


@dataclass(frozen=True)
class UtteranceWer:
    manifest_row: int
    utterance_id: str
    source: str
    cluster_id: str
    reference: str
    hypothesis: str
    ref_words: int
    hyp_words: int
    substitutions: int
    deletions: int
    insertions: int
    errors: int
    utterance_wer: float | None
    included: bool
    exclusion_reason: str | None


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def _levenshtein_distance(ref: list[str], hyp: list[str]) -> int:
    """Independent minimum word edit distance -- the cross-check path.

    Deliberately NOT jiwer.  E_u = S+D+I from any minimum-cost alignment
    equals the plain Levenshtein distance, so checking the total is a complete
    check of E_u without trusting jiwer's own backtrace twice.
    """
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            sub = prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            cur[j] = min(sub, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[len(hyp)]


def _jiwer_sdi_batch(refs: list[str], hyps: list[str]) -> list[tuple[int, int, int]]:
    """Primary alignment path via jiwer, batched (one process_words call).

    Inputs must be normalized, non-empty-reference strings.  Returns (S, D, I)
    per pair, derived from the alignment chunks exactly the way
    src/evaluation.py::_align does -- the two scorers must agree on E_u.
    """
    import jiwer

    tf = jiwer.Compose(
        [
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ]
    )
    result = jiwer.process_words(
        refs, hyps, reference_transform=tf, hypothesis_transform=tf
    )
    out: list[tuple[int, int, int]] = []
    for local, alignment in enumerate(result.alignments):
        s = d = i = 0
        for chunk in alignment:
            span_ref = chunk.ref_end_idx - chunk.ref_start_idx
            span_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx
            if chunk.type == "substitute":
                s += span_ref
            elif chunk.type == "delete":
                d += span_ref
            elif chunk.type == "insert":
                i += span_hyp
            elif chunk.type != "equal":  # pragma: no cover
                raise MacroWerAbort(
                    f"unknown jiwer alignment chunk type {chunk.type!r}"
                )
        out.append((s, d, i))
    return out


def score_utterance_wer(
    reference_raw: str,
    hypothesis_raw: str,
    *,
    score_punctuation: bool = False,
) -> tuple[str, str, int, int, int, int, int]:
    """Normalize one ref/hyp pair and align it.

    Returns (ref_norm, hyp_norm, R, H, S, D, I).  Raises MacroWerAbort when the
    reference normalizes to zero words: the CALLER owns the report-and-drop
    bookkeeping, and returning fabricated counts here would let a lazy caller
    average them in.
    """
    ref_norm = " ".join(metric_surface(reference_raw, score_punctuation).split())
    hyp_norm = " ".join(metric_surface(hypothesis_raw, score_punctuation).split())
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if not ref_words:
        raise MacroWerAbort(
            f"reference normalized to zero words ({EMPTY_REF_REASON}); "
            f"the caller must report-and-drop this row, not score it"
        )
    (s, d, i), = _jiwer_sdi_batch([ref_norm], [hyp_norm])
    dist = _levenshtein_distance(ref_words, hyp_words)
    if s + d + i != dist:
        raise MacroWerAbort(
            f"alignment cross-check failed: jiwer S+D+I={s + d + i} vs "
            f"independent Levenshtein {dist} for ref={ref_norm!r} "
            f"hyp={hyp_norm!r}"
        )
    return ref_norm, hyp_norm, len(ref_words), len(hyp_words), s, d, i


# --------------------------------------------------------------------------- #
# Utterance table construction
# --------------------------------------------------------------------------- #

def build_utterances(
    rows: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[str],
    *,
    source_order: Sequence[str] = SOURCE_ORDER,
    cluster_prefixes: Mapping[str, str] | None = None,
    score_punctuation: bool = False,
) -> list[UtteranceWer]:
    """Score every manifest row against its hypothesis, in manifest order."""
    if len(rows) != len(hypotheses):
        raise MacroWerAbort(
            f"hypothesis count {len(hypotheses)} != manifest row count "
            f"{len(rows)}; refusing to guess an alignment"
        )
    prefixes = dict(CLUSTER_PREFIXES if cluster_prefixes is None else cluster_prefixes)
    for s in source_order:
        if s not in prefixes:
            raise MacroWerAbort(
                f"no cluster-key prefix declared for source {s!r}; the "
                f"bootstrap unit must be explicit, never inferred"
            )

    # Pass 1: normalize, classify empties, collect non-empty pairs for one
    # batched jiwer call (per-pair calls are ~40x slower on the 9k-row dev set).
    norm: list[tuple[str, str]] = []
    include_idx: list[int] = []
    meta: list[dict] = []
    for row_i, (row, hyp_raw) in enumerate(zip(rows, hypotheses)):
        src = row.get("source")
        if src not in source_order:
            raise MacroWerAbort(
                f"manifest row {row_i} has source {src!r}, not one of "
                f"{tuple(source_order)}"
            )
        cluster = row.get("group")
        if not isinstance(cluster, str) or not cluster.startswith(prefixes[src]):
            raise MacroWerAbort(
                f"manifest row {row_i} (source {src}) has cluster key "
                f"{cluster!r}, which does not start with the required prefix "
                f"{prefixes[src]!r}"
            )
        ref_norm = " ".join(metric_surface(row["text"], score_punctuation).split())
        hyp_norm = " ".join(metric_surface(hyp_raw, score_punctuation).split())
        meta.append(
            {
                "row": row_i,
                "utt": os.path.basename(str(row.get("audio_filepath", row_i))),
                "source": src,
                "cluster": cluster,
                "ref": ref_norm,
                "hyp": hyp_norm,
            }
        )
        if ref_norm.split():
            include_idx.append(row_i)
            norm.append((ref_norm, hyp_norm))

    sdi_by_row: dict[int, tuple[int, int, int]] = {}
    if norm:
        refs = [r for r, _ in norm]
        hyps = [h for _, h in norm]
        for row_i, sdi in zip(include_idx, _jiwer_sdi_batch(refs, hyps)):
            sdi_by_row[row_i] = sdi

    utterances: list[UtteranceWer] = []
    for m in meta:
        row_i = m["row"]
        ref_words = m["ref"].split()
        hyp_words = m["hyp"].split()
        if row_i not in sdi_by_row:
            # Report-and-drop: the hypothesis words are what WOULD have been
            # insertions, recorded so the drop is auditable, never scored.
            utterances.append(
                UtteranceWer(
                    manifest_row=row_i,
                    utterance_id=m["utt"],
                    source=m["source"],
                    cluster_id=m["cluster"],
                    reference=m["ref"],
                    hypothesis=m["hyp"],
                    ref_words=0,
                    hyp_words=len(hyp_words),
                    substitutions=0,
                    deletions=0,
                    insertions=0,
                    errors=0,
                    utterance_wer=None,
                    included=False,
                    exclusion_reason=EMPTY_REF_REASON,
                )
            )
            continue
        s, d, i = sdi_by_row[row_i]
        dist = _levenshtein_distance(ref_words, hyp_words)
        if s + d + i != dist:
            raise MacroWerAbort(
                f"alignment cross-check failed at manifest row {row_i} "
                f"({m['utt']}): jiwer E={s + d + i}, independent Levenshtein "
                f"E={dist}"
            )
        r_n = len(ref_words)
        utterances.append(
            UtteranceWer(
                manifest_row=row_i,
                utterance_id=m["utt"],
                source=m["source"],
                cluster_id=m["cluster"],
                reference=m["ref"],
                hypothesis=m["hyp"],
                ref_words=r_n,
                hyp_words=len(hyp_words),
                substitutions=s,
                deletions=d,
                insertions=i,
                errors=s + d + i,
                utterance_wer=(s + d + i) / r_n,
                included=True,
                exclusion_reason=None,
            )
        )
    return utterances


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def cluster_draw_indices(
    cluster_counts: Mapping[str, int],
    *,
    source_order: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Deterministic per-source cluster index draws, shape (resamples, C_s).

    Seeding: one SeedSequence per source, spawned from the master seed in
    source order.  Spawn (rather than one shared stream) means source A's
    cluster count cannot perturb source B's draws -- required so that PAIRED
    system comparisons on the same manifest reuse the exact same draws.
    """
    children = np.random.SeedSequence(seed).spawn(len(source_order))
    draws: dict[str, np.ndarray] = {}
    for child, src in zip(children, source_order):
        c = cluster_counts[src]
        if c <= 0:
            raise MacroWerAbort(f"source {src!r} has no clusters to resample")
        rng = np.random.default_rng(child)  # PCG64
        draws[src] = rng.integers(0, c, size=(resamples, c), dtype=np.int64)
    return draws


def _per_source_cluster_stats(
    utterances: Sequence[UtteranceWer],
    source_order: Sequence[str],
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """For each source: (sorted cluster ids, per-cluster sum(E/R), per-cluster n)."""
    per: dict[str, dict[str, list[float]]] = {s: {} for s in source_order}
    for u in utterances:
        if not u.included:
            continue
        per[u.source].setdefault(u.cluster_id, []).append(u.errors / u.ref_words)
    out = {}
    for s in source_order:
        if not per[s]:
            raise MacroWerAbort(
                f"source {s!r} has zero scoreable utterances; a five-source "
                f"balanced mean over a phantom source is meaningless"
            )
        ids = sorted(per[s])  # sorted: draw index i must always mean the same cluster
        r_sum = np.array([sum(per[s][c]) for c in ids], dtype=np.float64)
        n = np.array([len(per[s][c]) for c in ids], dtype=np.float64)
        out[s] = (ids, r_sum, n)
    return out


def stratified_cluster_bootstrap(
    utterances: Sequence[UtteranceWer],
    *,
    source_order: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Bootstrap draws of w_s per source plus the overall balanced W.

    Per draw and source: sample C_s cluster ids with replacement; a cluster
    drawn k times contributes its rows k times.  The resampled source mean is
    utterance-occurrence weighted,

        w_s^(b) = sum_selected sum_u E/R  /  #selected utterance occurrences,

    NOT a mean of cluster means -- a 3-row cluster must outweigh a 1-row
    cluster inside a draw exactly as it does in the point estimate.
    Returns {"overall": (resamples,), <source>: (resamples,)}.
    """
    stats = _per_source_cluster_stats(utterances, source_order)
    draws = cluster_draw_indices(
        {s: len(stats[s][0]) for s in source_order},
        source_order=source_order,
        resamples=resamples,
        seed=seed,
    )
    out: dict[str, np.ndarray] = {}
    acc = np.zeros(resamples, dtype=np.float64)
    for s in source_order:
        _, r_sum, n = stats[s]
        idx = draws[s]  # (resamples, C_s)
        w = r_sum[idx].sum(axis=1) / n[idx].sum(axis=1)
        out[s] = w
        acc += w
    out["overall"] = acc / len(source_order)
    return out


def percentile_ci(draws: np.ndarray, quantiles=(0.025, 0.975)) -> list[float]:
    lo, hi = np.quantile(draws, quantiles)  # linear interpolation, the default
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def source_balanced_wer(
    rows: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[str],
    *,
    source_order: Sequence[str] = SOURCE_ORDER,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    score_punctuation: bool = False,
    cluster_prefixes: Mapping[str, str] | None = None,
    cluster_units: Mapping[str, str] | None = None,
    instrument: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    decoding: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a manifest and return the full results document.

    instrument/model/decoding/artifacts blocks are supplied by the caller
    (the tail driver knows the hashes; the scorer must not guess them) and are
    embedded verbatim so results.json is self-describing.
    """
    units = dict(CLUSTER_UNITS if cluster_units is None else cluster_units)
    utterances = build_utterances(
        rows,
        hypotheses,
        source_order=source_order,
        cluster_prefixes=cluster_prefixes,
        score_punctuation=score_punctuation,
    )

    boot = stratified_cluster_bootstrap(
        utterances,
        source_order=source_order,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    sources_out = []
    w_point = []
    tot_err = tot_ref = 0
    for s in source_order:
        included = [u for u in utterances if u.source == s and u.included]
        dropped = [u for u in utterances if u.source == s and not u.included]
        n_s = len(included)
        w_s = sum(u.errors / u.ref_words for u in included) / n_s
        w_point.append(w_s)
        ref_w = sum(u.ref_words for u in included)
        errs = sum(u.errors for u in included)
        empty_hyp = sum(1 for u in included if u.hyp_words == 0)
        tot_err += errs
        tot_ref += ref_w
        sources_out.append(
            {
                "source": s,
                "cluster_unit": units.get(s, "unspecified"),
                "n_clusters": len({u.cluster_id for u in included}),
                "n_utterances_manifest": n_s + len(dropped),
                "n_utterances_scored": n_s,
                "empty_references_dropped": len(dropped),
                "ref_words": ref_w,
                "hyp_words": sum(u.hyp_words for u in included),
                "substitutions": sum(u.substitutions for u in included),
                "deletions": sum(u.deletions for u in included),
                "insertions": sum(u.insertions for u in included),
                "errors": errs,
                "empty_hypotheses": empty_hyp,
                "empty_hypothesis_rate": empty_hyp / n_s,
                "utterance_macro_wer": w_s,
                "utterance_macro_wer_ci95": percentile_ci(boot[s]),
                "micro_wer": errs / ref_w,
            }
        )

    headline_w = sum(w_point) / len(source_order)
    # Cross-check the vectorized bootstrap's point path: recompute W from the
    # per-cluster stats it used.  Guards against a stats/draws indexing slip.
    stats = _per_source_cluster_stats(utterances, source_order)
    w_check = sum(
        float(stats[s][1].sum() / stats[s][2].sum()) for s in source_order
    ) / len(source_order)
    if abs(w_check - headline_w) > 1e-12:
        raise MacroWerAbort(
            f"metric cross-check failed: direct W={headline_w!r} vs "
            f"cluster-table W={w_check!r}"
        )

    n_dropped = sum(1 for u in utterances if not u.included)
    doc = {
        "schema": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "instrument": dict(instrument) if instrument else None,
        "model": dict(model) if model else None,
        "decoding": dict(decoding) if decoding else None,
        "metric": {
            "name": "source_balanced_utterance_macro_wer",
            "normalizer": NORMALIZER_VERSION,
            "normalizer_sha256": source_sha256(),
            "unicode_version": UNICODE_VERSION,
            "score_punctuation": score_punctuation,
            "word_tokenization": "str.split",
            "error_definition": "S+D+I",
            "empty_reference_rule": "report_and_drop",
            "formula": "(1/S)*sum_s((1/n_s)*sum_u(E_u/R_u))",
        },
        "bootstrap": {
            "method": "source-stratified cluster percentile bootstrap",
            "rng": "numpy.random.default_rng/PCG64",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "quantiles": [0.025, 0.975],
            "cluster_units": {s: units.get(s, "unspecified") for s in source_order},
        },
        "headline": {
            "source_balanced_utterance_macro_wer": headline_w,
            "ci95": percentile_ci(boot["overall"]),
            "n_sources": len(source_order),
            "n_utterances_manifest": len(rows),
            "n_utterances_scored": len(rows) - n_dropped,
            "empty_references_dropped": n_dropped,
        },
        "sources": sources_out,
        "corpus_micro_word_weighted": {
            "ref_words": tot_ref,
            "errors": tot_err,
            "wer": tot_err / tot_ref,
        },
        "artifacts": dict(artifacts) if artifacts else None,
        "integrity": {
            "hypothesis_count_matches_manifest": True,
            "all_sources_present": True,
            "cluster_keys_valid": True,
            "metric_crosscheck_ok": True,
        },
    }
    return doc


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #

def write_predictions_jsonl(path: str | os.PathLike[str],
                            utterances: Sequence[UtteranceWer]) -> str:
    """Write the per-utterance record and return its sha256.

    Excluded rows are written too (that IS the report half of
    report-and-drop), with would_be_insertions instead of scored counts.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for u in utterances:
            rec = asdict(u)
            if not u.included:
                rec["would_be_insertions"] = u.hyp_words
            json.dump(rec, f, ensure_ascii=False, allow_nan=False)
            f.write("\n")
    os.replace(tmp, p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_results(path: str | os.PathLike[str], doc: Mapping[str, Any]) -> None:
    """allow_nan=False on purpose: a NaN reaching here means a division the
    contract forbids already happened; refuse to serialize it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Score a manifest with the source-balanced utterance-macro WER."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hypotheses", required=True,
                    help="jsonl with one {'hypothesis': ...} per manifest row, "
                         "in manifest order; audio_filepath, when present, is "
                         "cross-checked against the manifest row")
    ap.add_argument("--output", required=True)
    ap.add_argument("--predictions-out", default=None)
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    hyp_rows = [json.loads(l) for l in open(args.hypotheses, encoding="utf-8")]
    if len(hyp_rows) != len(rows):
        raise MacroWerAbort(
            f"{args.hypotheses}: {len(hyp_rows)} rows vs manifest {len(rows)}"
        )
    for i, (r, h) in enumerate(zip(rows, hyp_rows)):
        if "audio_filepath" in h and h["audio_filepath"] != r["audio_filepath"]:
            raise MacroWerAbort(
                f"hypothesis row {i} is for {h['audio_filepath']!r} but the "
                f"manifest row is {r['audio_filepath']!r}; order is broken"
            )
    hyps = [h["hypothesis"] for h in hyp_rows]

    utterances = build_utterances(rows, hyps)
    artifacts = None
    if args.predictions_out:
        sha = write_predictions_jsonl(args.predictions_out, utterances)
        artifacts = {"predictions_jsonl": args.predictions_out,
                     "predictions_sha256": sha}
    doc = source_balanced_wer(
        rows, hyps,
        bootstrap_resamples=args.resamples,
        bootstrap_seed=args.seed,
        artifacts=artifacts,
    )
    dump_results(args.output, doc)
    hl = doc["headline"]
    print(f"source_balanced_utterance_macro_wer={hl['source_balanced_utterance_macro_wer']:.6f} "
          f"ci95=[{hl['ci95'][0]:.6f},{hl['ci95'][1]:.6f}] "
          f"scored={hl['n_utterances_scored']}/{hl['n_utterances_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
