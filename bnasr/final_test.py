#!/usr/bin/env python3
"""SPEC §6 headline: the ONE FLEURS-test decode of the this recipe winner (CTC).

This is the only this recipe tool permitted to read
``/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn``. Every other tool in
``bnasr/`` refuses paths under it; this one guards the door from the inside:

* ``--dry-run`` rehearses on the LOCKED DEV panel and asserts that no path it
  touches resolves under the sealed directory, so a rehearsal can never burn the
  real event;
* ``--selftest`` exercises the bundle, resume and marker machinery on synthetic
  audio with no model at all;
* a real run refuses to start if a completion marker already exists (SPEC §6:
  "No second decode."), and ``--force`` is the only way past that;
* the held decode is ONE PHASE-LEVEL EVENT (council Q defect 9). Before the
  first prediction the tool freezes an identity — selected checkpoint + sha256,
  decoder settings, scorer/surface, panel manifests and their row counts, and
  the event id — into ``IDENTITY.json`` in the bundle, records the event in
  ``experiments/final_test_event.json`` (OUTSIDE any bundle) under an
  exclusive lock, and stamps every prediction row with the identity digest. A
  resume must match the recorded identity exactly; predictions from another
  checkpoint are refused, never reused; and pointing ``--out-dir`` at a fresh
  directory does not mint a second authorisation — only
  ``P4_FINAL_TEST_SECOND_EVENT=1`` does.

Adapted from ``tools/p3_final_test.py`` — the Phase-3 tool that produced the
last headline — with three changes that this recipe forces:

1. **CTC, not transducer.** ``scoring.dev_eval``/``load_asr_model`` reach into
   ``model.joint`` to disable fused loss/WER; ``EncDecCTCModelBPE`` has no joint.
   The model loader here is ``panel_eval.load_ctc_model``, which asserts the
   restored class and forces greedy decoding (SPEC §6: "greedy CTC decoding …
   no decoder search").
2. **Two paired comparisons, not one.** SPEC §6 requires the interval against
   frozen Phase-2 (the success criterion) *and* against the baseline (the lineage
   claim). Both use ``bnasr/paired_bootstrap.py`` — one implementation of
   the interval for the pilots and the headline, so the number that stops the
   project and the number that starts it cannot come from two different tests.
3. **Resumable inference and an immutable bundle.** 920 clips at batch 1 through
   ``asr_core`` is ~15 minutes on a shared card; a killed process must not cost
   the sealed set a second opening. Hypotheses are appended to a JSONL shard as
   they are produced, a restart skips the keys already in it, and the completed
   bundle is written once and then made read-only with a manifest of SHA-256s.

Why the frozen baselines are re-scored BEFORE anything else runs
----------------------------------------------------------------
The headline is a comparison against two files on disk that were produced months
ago by another repository. If either file is not the one the registered numbers
were computed from, the comparison is against an unknown system. So the first
thing this tool does — before the model is loaded, before a single test file is
opened — is re-score both prediction files through THIS repository's frozen
scorer and check them against the values SPEC §6 registers. Phase 2 must
re-score to 16.54 % and the baseline to 19.67 %. See ``P1_SPEC_WER`` for the stale
constant that made this check report a discrepancy that did not exist, and how
it was reconciled on 2026-09-13.

Usage
-----
    bnasr/final_test.py --selftest                       # no GPU, no model
    CUDA_VISIBLE_DEVICES=GPU-3ce8... bnasr/final_test.py \
        --ckpt experiments/fastconformer_ctc_bn_1kh/final.nemo --dry-run --dry-n 8
    CUDA_VISIBLE_DEVICES=GPU-3ce8... bnasr/final_test.py \
        --ckpt experiments/full/candidates/step_XXXXXX.nemo   # ONCE, ever
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import (  # noqa: E402
    anonymize_process, cer_counts, frozen_norm, load_manifest, sha256_file,
    pin_gpu, wer_counts,
)

anonymize_process()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import stat  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402

from bnasr.paired_bootstrap import (  # noqa: E402
    DRAWS_FINAL, SEED, cluster_blocks, paired_bootstrap,
)

BENCH = Path("/mnt/sdb/arafat/ehz/llm/bench_assets")
SEALED_TEST_DIR = (BENCH / "eval_fleurs_bn").resolve()
EVAL_MANIFEST = BENCH / "eval_manifest.json"

# SPEC §6: "frozen Phase-2 (16.54 %) and the baseline (19.67 %) hypotheses".
P2_PREDICTIONS = Path("/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/"
                      "outputs_a5000/qwen3_adapter/predictions.json")
P1_PREDICTIONS = Path("/mnt/sdb/arafat/ehz/llm/bengali-asr-benchmark/"
                      "outputs_a5000/ehzawad_fastconformer/predictions.json")

# ---------------------------------------------------------------------------
# Registered constants (SPEC §6), each asserted against the text of the SPEC.
# ---------------------------------------------------------------------------

# "decode 920 FLEURS-test recordings once … (349 clusters, 20,000 draws,
# seed 20260905, corpus WER = Σedits/Σref words)".
N_TEST = 920
N_CLUSTERS = 349
assert N_TEST == 920, "SPEC §2/§6: FLEURS official test = 920 recordings"
assert N_CLUSTERS == 349, "SPEC §2/§6: 349 distinct sentences"
assert DRAWS_FINAL == 20000 and SEED == 20260905, "SPEC §6: 20,000 draws, seed 20260905"

# "Success: WER < 16.54 % and the 95 % CI of (P4 − P2) lies entirely below 0.
#  'Strong': ≤16.04 %."
SUCCESS_WER = 0.1654
STRONG_WER = 0.1604
assert SUCCESS_WER == 0.1654, "SPEC §6 success threshold"
assert STRONG_WER == 0.1604, "SPEC §6 'strong' threshold"

# The two frozen baselines, as SPEC §6 states them.
P2_SPEC_WER = 0.1654
P1_SPEC_WER = 0.19672
assert P2_SPEC_WER == 0.1654 and P1_SPEC_WER == 0.19672, "SPEC §6 baseline WERs"

# Tolerance for "re-scores to the registered value": the SPEC quotes 2 decimal
# places in percentage points, so anything inside half a display unit is the
# same number and anything outside it is a different file or a different scorer.
RESCORE_TOL_PP = 0.005

# MEASURED, 2026-09-05, by re-scoring both files through tools/scoring
# (frozen_norm + jiwer) on the 920 split=="test" rows:
#     qwen3_adapter          16.5362 pp  -> SPEC's 16.54, matches
#     ehzawad_fastconformer  19.6723 pp  -> SPEC's 19.67, matches
#
# RESOLVED 2026-09-13. This tool was written against a draft of SPEC §6 that
# recorded the baseline at 19.57 pp, and it kept asserting that value after the SPEC
# itself was corrected ON THE SAME DAY. The current §6 text reads: "the baseline
# (**19.67 %** under the frozen scorer on the 920 test rows — the 19.57 %
# quoted earlier was a different harness; corrected 2026-09-05 after re-scoring
# outputs_a5000/ehzawad_fastconformer/predictions.json, which the benchmark
# summary also records as 0.19672)". So all three sources agree at 19.672 and
# the only stale value was the constant here, which manufactured a discrepancy
# that stopped the decode.
#
# Note the benchmark repo publishes TWO summaries: summary_a5000.json records
# wer_test 0.19672 (this card, the registered comparator) and the public
# summary.json records 0.19650 from an RTX 5080 run. Different hardware, not a
# contradiction; §6 registers the A5000 file, which is the one on disk here.
P1_MEASURED_WER_NOTE = ("SPEC §6 records the baseline at 19.67 pp (corrected "
                        "2026-09-05); the frozen predictions re-score to "
                        "19.6723 pp and summary_a5000.json records 0.19672. "
                        "Reconciled 2026-09-13.")

OUT_DIR = REPO / "experiments/final_test"
MARKER = "COMPLETED.json"

DEV_CORPUS = REPO / "data2/val_corpus.json"
# the baseline dev-panel hypotheses keyed by audio_filepath, written by
# bnasr/panel_eval.py --save-hyps. Used ONLY by --dry-run, so the
# rehearsal's bootstrap runs on genuine paired data rather than on a copy of the
# candidate's own output (which would make every rehearsal Δ exactly 0 and hide
# a broken pairing).
DEV_PAIR_HYPS = REPO / "experiments/eval/p1_panels.hyps.json"


# ---------------------------------------------------------------------------
# Baseline re-score gate
# ---------------------------------------------------------------------------

def _test_rows_from(preds_path):
    """The split=="test" rows of a benchmark predictions.json, keyed by file.

    TEXT ONLY. This reads hypotheses and references from a JSON file in another
    repository; it never touches the sealed audio, which is why it is legal to
    run at any time, including from a dry run.
    """
    rows = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        if r.get("split") != "test":
            continue
        if r["file"] in out:
            raise SystemExit(f"{preds_path}: duplicate file {r['file']!r}")
        out[r["file"]] = r
    return out


def rescore_frozen_baseline(preds_path, spec_wer, label):
    """Re-score one frozen baseline file and compare with its SPEC §6 value."""
    rows = _test_rows_from(preds_path)
    we = wn = ce = cn = 0
    clusters = set()
    for r in rows.values():
        ref, hyp = frozen_norm(r["raw_reference"]), frozen_norm(r["transcript"] or "")
        e, n = wer_counts(ref, hyp)
        c, m = cer_counts(ref, hyp)
        we, wn, ce, cn = we + e, wn + n, ce + c, cn + m
        clusters.add(ref)
    wer_pp = 100.0 * we / max(wn, 1)
    return {
        "label": label, "path": str(preds_path), "sha256": sha256_file(preds_path),
        "n": len(rows), "n_expected": N_TEST,
        "n_clusters": len(clusters), "n_clusters_expected": N_CLUSTERS,
        "wer_pp_rescored": round(wer_pp, 4),
        "cer_pp_rescored": round(100.0 * ce / max(cn, 1), 4),
        "wer_pp_spec": round(spec_wer * 100, 4),
        "delta_pp": round(wer_pp - spec_wer * 100, 4),
        "tolerance_pp": RESCORE_TOL_PP,
        "matches_spec": abs(wer_pp - spec_wer * 100) <= RESCORE_TOL_PP,
        "rows_ok": len(rows) == N_TEST and len(clusters) == N_CLUSTERS,
    }


def baseline_gate(ack_p1=False, enforce=True):
    """SPEC §6 baselines, re-scored before anything else in this process runs.

    ``enforce=False`` (a dry run or a self-test) reports without stopping — the
    rehearsal has to be able to run while SPEC §6's the baseline number is unresolved,
    or the machinery could never be tested. A REAL run stops.
    """
    p2 = rescore_frozen_baseline(P2_PREDICTIONS, P2_SPEC_WER, "phase2_qwen3_adapter")
    p1 = rescore_frozen_baseline(P1_PREDICTIONS, P1_SPEC_WER, "phase1_fastconformer")
    p1["note"] = P1_MEASURED_WER_NOTE
    gate = {"phase2": p2, "phase1": p1,
            "p1_discrepancy_acknowledged": bool(ack_p1),
            "enforced": bool(enforce)}
    if not enforce:
        return gate
    if not (p2["matches_spec"] and p2["rows_ok"]):
        raise SystemExit(
            f"ABORT: frozen Phase-2 hypotheses re-score to {p2['wer_pp_rescored']} pp "
            f"({p2['n']} rows, {p2['n_clusters']} clusters); SPEC §6 registers "
            f"{p2['wer_pp_spec']} pp over {N_TEST}/{N_CLUSTERS}. The headline "
            f"comparison is against this file — it must be the registered one.")
    if not p1["rows_ok"]:
        raise SystemExit(f"ABORT: frozen the baseline hypotheses are {p1['n']} rows / "
                         f"{p1['n_clusters']} clusters, not {N_TEST}/{N_CLUSTERS}")
    if not p1["matches_spec"] and not ack_p1:
        raise SystemExit(
            f"ABORT: frozen the baseline hypotheses re-score to {p1['wer_pp_rescored']} pp, "
            f"SPEC §6 registers {p1['wer_pp_spec']} pp (Δ {p1['delta_pp']} pp). "
            f"{P1_MEASURED_WER_NOTE} Resolve SPEC §6 (owner) or pass "
            f"--ack-p1-spec-discrepancy to run the ONE decode with the "
            f"discrepancy recorded in the bundle.")
    return gate


# ---------------------------------------------------------------------------
# Council Q defect 9: one phase-level held-decode event, with a frozen identity
# ---------------------------------------------------------------------------
# The bug this closes: ``decode_items`` resumed a prediction shard BY FILENAME
# ALONE and the checkpoint was only recorded afterwards, in the report. Decode
# 700 clips with checkpoint X, crash, restart the same directory with Y, and the
# tool kept X's 700 rows, decoded Y's remaining 220 and labelled the result Y.
# The completion marker was output-directory-local too, so ``--out-dir`` at a
# fresh path minted a second authorisation for what SPEC §6 allows exactly once.
#
# Council Q's required fix, implemented here: one phase-level event record, an
# exclusive evaluation lock, immutable selected-checkpoint / decoder / scorer /
# panel identities recorded BEFORE the first prediction, identical identities
# required on resume, and a fresh output directory that does NOT create another
# authorisation.

# The durable event record. It lives OUTSIDE any bundle directory on purpose:
# that is the whole point of a phase-level event.
EVENT_REGISTRY = REPO / "experiments/final_test_event.json"
EVENT_OVERRIDE_ENV = "P4_FINAL_TEST_SECOND_EVENT"
IDENTITY = "IDENTITY.json"


def event_registry_path():
    return EVENT_REGISTRY


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def identity_digest(ident):
    """One hash over the whole frozen identity — checkpoint, decoder, scorer,
    panels and the event id. Any field moving moves this."""
    return hashlib.sha256(_canon(ident).encode("utf-8")).hexdigest()


def _file_identity(path, label):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ABORT: {label} is missing: {p}")
    return {"path": str(p.resolve()), "sha256": sha256_file(p)}


def _rows_in(path, predicate=None):
    """Row count of a JSON manifest (``{"rows": [...]}`` or a bare list)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj["rows"] if isinstance(obj, dict) and "rows" in obj else obj
    if predicate is not None:
        rows = [r for r in rows if predicate(r)]
    return len(rows)


def panel_identities(final):
    """Every manifest the event reads, with its sha256 AND its row count.

    A file that is the registered one but has been truncated is a different
    panel; the count is what makes that visible without re-reading the file
    later.
    """
    if final:
        return {
            "test_manifest": dict(
                _file_identity(EVAL_MANIFEST, "eval_manifest.json"),
                rows=_rows_in(EVAL_MANIFEST, lambda r: r.get("split") == "test"),
                rows_expected=N_TEST),
            "phase2_frozen": dict(
                _file_identity(P2_PREDICTIONS, "frozen Phase-2 predictions"),
                rows=len(_test_rows_from(P2_PREDICTIONS)), rows_expected=N_TEST),
            "phase1_frozen": dict(
                _file_identity(P1_PREDICTIONS, "frozen the baseline predictions"),
                rows=len(_test_rows_from(P1_PREDICTIONS)), rows_expected=N_TEST),
        }
    # The dev corpus is a JSONL manifest, so it is counted through the same
    # reader the rehearsal itself uses rather than as a single JSON document.
    panels = {"dev_corpus": dict(_file_identity(DEV_CORPUS, "locked dev corpus"),
                                 rows=len(load_manifest(DEV_CORPUS)))}
    if DEV_PAIR_HYPS.exists():
        obj = json.loads(DEV_PAIR_HYPS.read_text(encoding="utf-8"))
        panels["dev_pair_hyps"] = dict(
            _file_identity(DEV_PAIR_HYPS, "dev pair hypotheses"),
            rows=len(obj.get("panels", {}).get("D", {})))
    return panels


def build_identity(ckpt, mode, event_id, draws=DRAWS_FINAL):
    """The immutable identity of ONE evaluation event, computed before the
    model is loaded and before a single prediction exists."""
    final = mode == "FINAL_TEST"
    return {
        "event_id": event_id,
        "mode": mode,
        "spec": "experiments/SPEC.md §6 — one decode of the sealed FLEURS test",
        "checkpoint": _file_identity(ckpt, "selected checkpoint"),
        "decoder": {
            "strategy": "greedy_ctc",
            "search": "none",
            "batch_size": 1,
            "segment_seconds": 25,
            "staging": "bench_assets/asr_core.py PCM16 staged WAV",
            "asr_core_sha256": sha256_file(BENCH / "asr_core.py"),
        },
        "scorer": {
            "surface": "tools/scoring.frozen_norm",
            "corpus_wer": "sum(edits)/sum(reference words)",
            "scoring_sha256": sha256_file(REPO / "bnasr/scoring.py"),
            "bench_score": _file_identity(BENCH / "bench_score.py", "bench_score.py"),
            "bootstrap": {
                "draws": int(draws), "seed": SEED,
                "tool_sha256": sha256_file(REPO / "bnasr/paired_bootstrap.py"),
            },
        },
        "panels": panel_identities(final),
    }


def _mismatches(saved, cur, prefix=""):
    out = []
    for k in sorted(set(saved) | set(cur)):
        a, b = saved.get(k, "<absent>"), cur.get(k, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(_mismatches(a, b, f"{prefix}{k}."))
        elif a != b:
            out.append(f"{prefix}{k}: recorded {a!r} != current {b!r}")
    return out


def require_identity_match(bundle_dir, ident):
    """Resume is permitted only against a byte-identical identity."""
    p = Path(bundle_dir) / IDENTITY
    if not p.exists():
        raise SystemExit(
            f"ABORT: {p} does not exist. The identity of this evaluation event "
            f"must be recorded before the first prediction; a directory holding "
            f"output but no identity cannot be resumed.")
    saved = json.loads(p.read_text(encoding="utf-8"))
    bad = _mismatches(saved, ident)
    if bad:
        raise SystemExit(
            "ABORT: this evaluation was authorised with a different identity. "
            "SPEC §6 allows ONE decode of the sealed test with ONE selected "
            "checkpoint, decoder, scorer and panel set. Mismatch(es): "
            + "; ".join(bad)
            + f". Recorded identity: {p}. Nothing was decoded.")
    return saved


def bind_identity(bundle_dir, ident, shard_path=None):
    """Record the identity, or require the recorded one to match exactly.

    Returns ``(identity, minted)``. A directory that already holds predictions
    but no identity is refused outright: those rows cannot be attributed to a
    checkpoint, so they can never be reused.
    """
    bundle_dir = Path(bundle_dir)
    p = bundle_dir / IDENTITY
    if p.exists():
        return require_identity_match(bundle_dir, ident), False
    if shard_path is not None:
        sp = Path(shard_path)
        if sp.exists() and sp.stat().st_size > 0:
            raise SystemExit(
                f"ABORT: {sp} holds predictions written before any identity was "
                f"recorded. They cannot be attributed to a checkpoint and will "
                f"not be reused. Nothing was decoded.")
    p.write_text(json.dumps(ident, indent=1, ensure_ascii=False), encoding="utf-8")
    return ident, True


class EvaluationLock:
    """Exclusive, process-lifetime lock on the phase-level evaluation event.

    Two evaluators must not open the sealed set at once — not in the same
    bundle and not in two different ones, which is why the lock is taken on the
    REGISTRY, not on an output directory.
    """

    def __init__(self, registry):
        self.path = Path(str(registry) + ".lock")
        self.fh = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.seek(0)
            holder = self.fh.read().strip()
            self.fh.close()
            self.fh = None
            raise SystemExit(
                f"ABORT: another this recipe held evaluation holds the exclusive "
                f"lock {self.path} ({holder or 'holder unknown'}). SPEC §6 "
                f"allows one decode of the sealed test; two processes must "
                f"never be inside it at once. Nothing was decoded.")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid {os.getpid()} since "
                      f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        self.fh.flush()
        return self

    def release(self):
        if self.fh is not None:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def read_event(registry):
    p = Path(registry)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_event_id(bundle_dir, registry):
    """The event id of a resume comes from the record, never from a new uuid."""
    ip = Path(bundle_dir) / IDENTITY
    if ip.exists():
        return json.loads(ip.read_text(encoding="utf-8"))["event_id"]
    rec = read_event(registry) if registry else None
    if rec and Path(rec["bundle_dir"]) == Path(bundle_dir).resolve():
        return rec["event_id"]
    return uuid.uuid4().hex


def _write_event(registry, rec):
    p = Path(registry)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return rec


def open_event(registry, bundle_dir, ident, override=None):
    """Mint or re-enter THE one authorised held-decode event.

    Returns ``(record, minted)``. A second event — a different identity, or the
    same identity pointed at a different output directory — is refused unless
    ``P4_FINAL_TEST_SECOND_EVENT=1`` is set. The refusal deliberately does not
    offer a directory as a way forward: a directory is not an authorisation.
    """
    registry = Path(registry)
    if override is None:
        override = os.environ.get(EVENT_OVERRIDE_ENV) == "1"
    dig = identity_digest(ident)
    here = Path(bundle_dir).resolve()
    prior = read_event(registry)
    if prior is not None:
        same_dir = Path(prior["bundle_dir"]) == here
        same_id = prior.get("identity_sha256") == dig
        if same_dir and same_id:
            # Council R (SPEC §8w.11): re-entry is for RESUMING an interrupted
            # decode, not for running the finished one again. A completed event
            # is closed; with --force --no-resume this branch previously allowed
            # a second decode of the sealed set without the override.
            if prior.get("status") in ("complete", "completed", "closed") \
                    and not override:
                raise SystemExit(
                    f"ABORT: the this recipe held decode registered in {registry} "
                    f"(event {prior['event_id']}, opened {prior['opened_utc']}) is "
                    f"already COMPLETE (status {prior.get('status')!r}"
                    + (f", WER {prior['wer']}" if prior.get("wer") is not None else "")
                    + "). SPEC §6 permits one decode of the sealed set; re-entry "
                      "exists to finish an interrupted one, not to repeat a "
                      "finished one. Neither --force nor a different --out-dir "
                      f"changes this. If it must be reopened, the owner records "
                      f"why and sets {EVENT_OVERRIDE_ENV}=1. Nothing was decoded.")
            return prior, False                      # a resume of the one event
        if not override:
            why = ("the same output directory under a DIFFERENT identity"
                   if same_dir else
                   f"a different output directory ({here})")
            raise SystemExit(
                f"ABORT: the this recipe held decode is a single phase-level event "
                f"and it is already registered in {registry} "
                f"(event {prior['event_id']}, opened {prior['opened_utc']}, "
                f"status {prior.get('status')}, bundle {prior['bundle_dir']}). "
                f"This invocation asks for {why}, which would be a SECOND "
                f"authorisation of an evaluation SPEC §6 permits once. A new "
                f"output directory does not create a new authorisation and "
                f"changing --out-dir will not change this answer. If the "
                f"registered event is genuinely not a real decode of the sealed "
                f"set, the owner must record why and set "
                f"{EVENT_OVERRIDE_ENV}=1. Nothing was decoded.")
    rec = {
        "spec": "experiments/SPEC.md §6 — one decode of the sealed FLEURS test",
        "tool": "bnasr/final_test.py",
        "event_id": ident["event_id"],
        "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opened_pid": os.getpid(),
        "bundle_dir": str(here),
        "identity_sha256": dig,
        "identity": ident,
        "status": "open",
    }
    if prior is not None:                            # override path only
        rec["supersedes"] = {"event_id": prior["event_id"],
                             "bundle_dir": prior["bundle_dir"],
                             "identity_sha256": prior.get("identity_sha256"),
                             "status": prior.get("status"),
                             "override_env": EVENT_OVERRIDE_ENV}
        return _write_event(registry, rec), True
    # No prior record: mint exclusively, so two racing processes cannot both
    # believe they minted the event.
    try:
        fd = os.open(str(registry), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(
            f"ABORT: {registry} appeared while this process was opening the "
            f"event — another evaluator minted it. Nothing was decoded.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    return rec, True


def close_event(registry, ident, report):
    """Mark the one event finished, with what it produced."""
    rec = read_event(registry)
    if rec is None or rec.get("identity_sha256") != identity_digest(ident):
        raise SystemExit(f"ABORT: {registry} no longer holds this event; the "
                         f"decode finished but the event cannot be closed.")
    rec["status"] = "completed"
    rec["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["wer_frozen_pp"] = report.get("wer_frozen_pp")
    rec["ckpt_sha256"] = report.get("ckpt_sha256")
    return _write_event(registry, rec)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def gather_test_items():
    """The sealed FLEURS test split. Reading this is the irreversible action."""
    manifest = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    if len(rows) != N_TEST:
        raise SystemExit(f"expected {N_TEST} test rows, found {len(rows)} — "
                         f"refusing to decode a test set that is not the "
                         f"registered one")
    keys = [r["file"] for r in rows]
    if len(set(keys)) != N_TEST:
        raise SystemExit("FLEURS test rows are not unique by file name")
    return [(r["file"], SEALED_TEST_DIR / r["file"], r["raw_reference"]) for r in rows]


def gather_dry_items(n):
    """Rehearsal input: the first n rows of the LOCKED DEV panel, paired against
    the baseline's stored dev hypotheses. Nothing here may resolve under the sealed
    directory — the caller asserts it."""
    rows = load_manifest(DEV_CORPUS)[:n]
    items = [(r["audio_filepath"], REPO / r["audio_filepath"], r["text"])
             for r in rows]
    pair = None
    if DEV_PAIR_HYPS.exists():
        obj = json.loads(DEV_PAIR_HYPS.read_text(encoding="utf-8"))
        d = obj.get("panels", {}).get("D", {})
        if all(k in d for k, _, _ in items):
            pair = [d[k] for k, _, _ in items]
    return items, pair


def decode_items(model, items, asr_core, shard_path, resume=True,
                 identity_sha=None):
    """Decode every item through the benchmark's own audio path, resumably.

    One JSON object per clip is appended to ``shard_path`` as soon as it is
    produced and the file handle is flushed, so a kill at clip 700 costs 700
    clips of GPU time and nothing else — in particular it does not cost a second
    opening of the sealed set, which SPEC §6 does not allow.

    Every row carries ``identity_sha256``: the digest of the frozen event
    identity (checkpoint, decoder, scorer, panels) that produced it. Resume
    matches on that digest as well as the file name, so a partial prediction set
    left by a different checkpoint is REFUSED rather than silently kept and
    relabelled — council Q defect 9.

    Timing wraps ``asr_core.decode`` only (the transcribe), because that is the
    quantity Phase 2's published latency was measured over; end-to-end wall is
    recorded alongside so the difference stays visible rather than implied.
    """
    # Council R (SPEC §8w.11): `--no-resume` used to IGNORE the existing shard
    # and decode again, appending - one clip became two rows, and the score was
    # then computed over a file with duplicates. A sealed set is scored once, so
    # a non-resuming pass over an existing shard is refused outright rather than
    # allowed to double-write it.
    if not resume and shard_path.exists() and shard_path.stat().st_size > 0:
        raise SystemExit(
            f"ABORT: --no-resume was given but {shard_path} already holds "
            f"predictions. Decoding again would append a second row for every "
            f"clip and the score would be computed over duplicates. The sealed "
            f"set is decoded ONCE (SPEC §6). Resume it, or record why the "
            f"existing predictions are invalid and move them aside deliberately. "
            f"Nothing was decoded.")
    done = {}
    if resume and shard_path.exists():
        for ln, line in enumerate(
                shard_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("identity_sha256") != identity_sha:
                raise SystemExit(
                    f"ABORT: {shard_path} line {ln} ({r.get('file')!r}) was "
                    f"produced under identity {r.get('identity_sha256')!r}, but "
                    f"this evaluation is identity {identity_sha!r}. Predictions "
                    f"from a different checkpoint, decoder, scorer or panel set "
                    f"are never reused. Nothing was decoded.")
            done[r["file"]] = r
    todo = [it for it in items if it[0] not in done]
    print(f"decode: {len(done)} already in {shard_path.name}, {len(todo)} to go",
          flush=True)
    with open(shard_path, "a", encoding="utf-8") as fh:
        for i, (key, path, ref) in enumerate(todo, 1):
            t_all = time.perf_counter()
            audio, note = asr_core.load_audio(str(path))
            pieces, forced = asr_core.chunks(audio)
            t0 = time.perf_counter()
            hyp = asr_core.decode(model, pieces)
            transcribe_s = time.perf_counter() - t0
            rec = {"file": key, "reference": ref, "transcript": hyp,
                   "identity_sha256": identity_sha,
                   "n_pieces": len(pieces), "forced_cuts": forced,
                   "downmix_note": note,
                   "transcribe_s": transcribe_s,
                   "end_to_end_s": time.perf_counter() - t_all,
                   "audio_seconds": len(audio) / asr_core.TARGET_SR}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            done[key] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    # Return in the item order, never in completion order: the paired bootstrap
    # is joined by key, but every downstream array (latency, counts) is
    # positional and a resumed run's shard is in a different order.
    return [done[k] for k, _, _ in items]


def score_rows(preds):
    """Frozen-surface per-row counts + cluster keys for the paired bootstrap."""
    we, wn, ce, cn, keys = [], [], [], [], []
    for p in preds:
        ref, hyp = frozen_norm(p["reference"]), frozen_norm(p["transcript"] or "")
        e, n = wer_counts(ref, hyp)
        c, m = cer_counts(ref, hyp)
        we.append(e)
        wn.append(n)
        ce.append(c)
        cn.append(m)
        keys.append(p["reference"])
    return (np.asarray(we), np.asarray(wn), np.asarray(ce), np.asarray(cn), keys)


def latency_stats(preds):
    """p50/p95 of the transcribe call at batch 1 — the Phase-3 SPEC Q4 protocol."""
    t = np.asarray([p["transcribe_s"] for p in preds]) * 1000.0
    e = np.asarray([p["end_to_end_s"] for p in preds]) * 1000.0
    audio_s = float(sum(p["audio_seconds"] for p in preds))
    return {
        "batch_size": 1,
        "protocol": "warmed in-process batch-1 transcribe through asr_core "
                    "staging (Phase-3 SPEC Q4 / benchmark protocol)",
        "transcribe_ms_p50": round(float(np.percentile(t, 50)), 3),
        "transcribe_ms_p95": round(float(np.percentile(t, 95)), 3),
        "transcribe_ms_mean": round(float(t.mean()), 3),
        "end_to_end_ms_p50": round(float(np.percentile(e, 50)), 3),
        "end_to_end_ms_p95": round(float(np.percentile(e, 95)), 3),
        "audio_seconds_total": round(audio_s, 1),
        "rtf_transcribe": round(float(t.sum() / 1000.0 / audio_s), 6) if audio_s else None,
    }


# ---------------------------------------------------------------------------
# Immutable bundle
# ---------------------------------------------------------------------------

def seal_bundle(bundle_dir, report):
    """Write the completion marker, then make every file in the bundle read-only.

    The marker carries a SHA-256 of every file it seals, so "this bundle was not
    edited afterwards" is checkable rather than asserted, and its presence is
    what a later invocation trips over: SPEC §6 allows one decode, and the file
    system is a more reliable memory of that than anybody's recollection.
    Read-only is a guard rail, not a security boundary — the owner can still
    chmod — but it turns an accidental overwrite into an error.
    """
    files = sorted(p for p in bundle_dir.iterdir() if p.is_file() and p.name != MARKER)
    marker = {
        "spec": "experiments/SPEC.md §6 — one decode of the sealed FLEURS test",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt": report["ckpt"], "ckpt_sha256": report["ckpt_sha256"],
        "event_id": report.get("event_id"),
        "identity_sha256": report.get("identity_sha256"),
        "wer_frozen_pp": report["wer_frozen_pp"],
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in files},
    }
    (bundle_dir / MARKER).write_text(
        json.dumps(marker, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in files + [bundle_dir / MARKER]:
        p.chmod(stat.S_IRUSR | stat.S_IRGRP)
    return marker


def check_not_already_done(bundle_dir, force):
    marker = bundle_dir / MARKER
    if marker.exists() and not force:
        raise SystemExit(
            f"{marker} exists — the sealed FLEURS test has already been decoded "
            f"for this bundle. SPEC §6: 'No second decode.' A different "
            f"--out-dir is NOT a way around this and does not create a second "
            f"authorisation. Pass --force only if this marker is not a real "
            f"prior decode of the sealed set.")


def _unlock(bundle_dir):
    """Make a forced re-run writable again. Only reached via --force."""
    for p in bundle_dir.iterdir():
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(ckpt, mode, bundle_dir, dry_n=8, device="cuda", force=False,
        ack_p1=False, draws=DRAWS_FINAL, resume=True, registry=None,
        override_event=None):
    """The one held-decode event, under an exclusive phase-level lock.

    The lock is taken on the EVENT REGISTRY rather than on the output
    directory, so two evaluators pointed at two different ``--out-dir`` values
    still serialise against each other.
    """
    final = mode == "FINAL_TEST"
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if not final:
        return _run_locked(ckpt, mode, bundle_dir, dry_n, device, force,
                           ack_p1, draws, resume, None, override_event)
    registry = Path(registry) if registry else event_registry_path()
    lock = EvaluationLock(registry).acquire()
    try:
        return _run_locked(ckpt, mode, bundle_dir, dry_n, device, force,
                           ack_p1, draws, resume, registry, override_event)
    finally:
        lock.release()


QUALIFIED_SELECTION = REPO / "experiments/qualified_selection.json"
SELECTION_OVERRIDE_ENV = "P4_ALLOW_UNQUALIFIED_FINAL"


def require_qualified_selection(ckpt):
    """The sealed decode runs on the SELECTED checkpoint of a QUALIFIED run.

    Council R (SPEC §8w.11): the entry point took any --ckpt. SPEC §8w.2 says
    the held decode is authorised only by a normal ending whose selected
    checkpoint passed qualification, so the tool must read that record rather
    than trust the caller. Absent record, non-normal ending, failed
    qualification, or a --ckpt that is not the selected one: refuse, and decode
    nothing. The run completes 14 September; until then this always refuses,
    which is the correct answer.
    """
    override = os.environ.get(SELECTION_OVERRIDE_ENV) == "1"
    if not QUALIFIED_SELECTION.exists():
        if override:
            print(f"WARNING {SELECTION_OVERRIDE_ENV}=1: decoding the sealed set with no "
                  f"qualified-selection record. This is not a registered this recipe result.",
                  flush=True)
            return None
        raise SystemExit(
            f"ABORT: no qualified-selection record at {QUALIFIED_SELECTION}. The sealed "
            f"decode is authorised by a NORMAL ending whose selected checkpoint passed "
            f"qualification (SPEC §8w.2), not by pointing --ckpt at a file. The production "
            f"run has not produced one. Nothing was decoded.")
    rec = json.loads(QUALIFIED_SELECTION.read_text(encoding="utf-8"))
    problems = []
    if rec.get("ending") != "normal":
        problems.append(f"ending is {rec.get('ending')!r}, not 'normal'")
    if not rec.get("qualified"):
        problems.append("the selected checkpoint did not pass qualification")
    want = rec.get("selected_checkpoint")
    if want and Path(want).resolve() != Path(ckpt).resolve():
        problems.append(f"--ckpt {ckpt} is not the selected checkpoint {want}")
    want_sha = rec.get("selected_sha256")
    if want_sha and Path(ckpt).exists() and sha256_file(Path(ckpt)) != want_sha:
        problems.append("the checkpoint's sha256 does not match the selected one")
    if problems and not override:
        raise SystemExit(
            "ABORT: the sealed decode is not authorised: " + "; ".join(problems) +
            f". Substituting another checkpoint to obtain a pass is what SPEC §8w.2 "
            f"forbids. Nothing was decoded.")
    if problems:
        print("WARNING decoding despite: " + "; ".join(problems), flush=True)
    return rec


def _run_locked(ckpt, mode, bundle_dir, dry_n, device, force, ack_p1, draws,
                resume, registry, override_event):
    final = mode == "FINAL_TEST"
    shard = bundle_dir / ("predictions.jsonl" if final
                          else "dryrun_predictions.jsonl")
    if final:
        require_qualified_selection(ckpt)
        check_not_already_done(bundle_dir, force)
        if force:
            _unlock(bundle_dir)

    # Identity BEFORE the first prediction: selected checkpoint + sha256,
    # decoder settings, scorer/surface, panel manifests and their row counts,
    # and the phase-level event id. A resume must match it exactly.
    event_id = resolve_event_id(bundle_dir, registry)
    ident = build_identity(ckpt, mode, event_id, draws=draws)
    bind_identity(bundle_dir, ident, shard)
    ident_sha = identity_digest(ident)
    event_rec = None
    if final:
        event_rec, _ = open_event(registry, bundle_dir, ident,
                                  override=override_event)

    gate = baseline_gate(ack_p1=ack_p1, enforce=final)

    sys.path.insert(0, str(BENCH))
    import asr_core  # noqa: E402  (bench_assets is not a package)
    from bench_score import norm as bench_norm  # noqa: E402

    if final:
        items = gather_test_items()
        p2 = _test_rows_from(P2_PREDICTIONS)
        p1 = _test_rows_from(P1_PREDICTIONS)
        for k, _, ref in items:
            for name, tbl in (("phase2", p2), ("phase1", p1)):
                if k not in tbl:
                    raise SystemExit(f"{name} hypotheses missing {k}")
                # A paired test is only paired if both sides were scored against
                # the same reference text.
                assert frozen_norm(tbl[k]["raw_reference"]) == frozen_norm(ref), (
                    f"{name} reference mismatch at {k}")
        pairs = {"phase2_qwen3_adapter_frozen": [p2[k]["transcript"] or "" for k, _, _ in items],
                 "phase1_fastconformer_frozen": [p1[k]["transcript"] or "" for k, _, _ in items]}
    else:
        items, pair = gather_dry_items(dry_n)
        for _, p, _ in items:
            rp = Path(p).resolve()
            assert rp != SEALED_TEST_DIR and SEALED_TEST_DIR not in rp.parents, (
                f"DRY RUN tried to touch the sealed test set: {p}")
        pairs = {"phase1_dev_hyps": pair} if pair else {}

    # The scoring surface this repo uses must agree with the benchmark's own on
    # the actual references, or the headline moves without the model moving.
    parity = all(bench_norm(ref) == frozen_norm(ref) for _, _, ref in items)

    from bnasr.panel_eval import load_ctc_model
    model = load_ctc_model(ckpt, device=device)

    preds = decode_items(model, items, asr_core, shard, resume=resume,
                         identity_sha=ident_sha)

    we, wn, ce, cn, keys = score_rows(preds)
    wer = float(we.sum() / max(wn.sum(), 1))
    boots = {}
    for label, hyps in pairs.items():
        pe, pn, _, _, _ = score_rows(
            [{"reference": r, "transcript": h}
             for (_, _, r), h in zip(items, hyps)])
        assert (pn == wn).all(), f"{label}: reference word counts differ"
        boots[label] = paired_bootstrap(we, pe, wn, keys, draws, SEED)

    report = {
        "tool": "bnasr/final_test.py",
        "spec": "experiments/SPEC.md §6",
        "mode": mode,
        "event_id": event_id,
        "identity_sha256": ident_sha,
        "identity": ident,
        "event_registry": str(registry) if registry else None,
        "ckpt": str(Path(ckpt).resolve()), "ckpt_sha256": sha256_file(ckpt),
        "device": device,
        "decode_path": "bench_assets/asr_core.py (25 s segmentation, PCM16 "
                       "staged WAV, transcribe batch_size=1)",
        "n": len(preds),
        "n_clusters": len(cluster_blocks([p["reference"] for p in preds])),
        "wer_frozen_pp": round(wer * 100, 4),
        "cer_frozen_pp": round(float(ce.sum() / max(cn.sum(), 1)) * 100, 4),
        "empty_hyps": int(sum(1 for p in preds if not p["transcript"].strip())),
        "multi_piece_clips": int(sum(1 for p in preds if p["n_pieces"] > 1)),
        "forced_cuts": int(sum(p["forced_cuts"] for p in preds)),
        "paired_bootstrap": boots,
        "latency": latency_stats(preds),
        "baseline_gate": gate,
        "scorer_parity_with_bench_score_norm": parity,
        "scorer_executed": {"path": str(BENCH / "bench_score.py"),
                            "sha256": sha256_file(BENCH / "bench_score.py")},
        "asr_core_sha256": sha256_file(BENCH / "asr_core.py"),
        "scoring_sha256": sha256_file(REPO / "bnasr/scoring.py"),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_tool_sha256": sha256_file(REPO / "bnasr/paired_bootstrap.py"),
        "thresholds": {"success_wer_pp": SUCCESS_WER * 100,
                       "strong_wer_pp": STRONG_WER * 100,
                       "rule": "SPEC §6: WER < 16.54 % AND 95 % CI of (P4 − P2) "
                               "entirely below 0"},
    }
    if final:
        p2b = boots["phase2_qwen3_adapter_frozen"]
        report["test_manifest_sha256"] = sha256_file(EVAL_MANIFEST)
        report["meets_success"] = bool(wer < SUCCESS_WER and p2b["ci_upper_below_zero"])
        report["meets_strong"] = bool(wer <= STRONG_WER and p2b["ci_upper_below_zero"])
    else:
        report["note"] = ("REHEARSAL on the locked dev panel. Not a headline; "
                          "the sealed FLEURS test was not opened.")
    if device.startswith("cuda"):
        import torch
        report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1 << 30), 3)

    name = "final_test.json" if final else "dryrun.json"
    (bundle_dir / name).write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    if final:
        report["completion_marker"] = seal_bundle(bundle_dir, report)
        report["event_record"] = close_event(registry, ident, report)
    return report


# ---------------------------------------------------------------------------
# Self-test: bundle / resume / marker machinery, no model and no real audio.
# ---------------------------------------------------------------------------

def selftest(tmp=None):
    """Exercise everything that is not the model, on synthetic data.

    A fake ``asr_core``-shaped object and a fake model stand in for the real
    ones, so the resume path, the marker, the read-only sealing and the paired
    bootstrap wiring are all tested without a GPU and without going anywhere
    near the sealed set.
    """
    import shutil
    tmp = Path(tmp or (REPO / "experiments/fixtures/final_test_selftest"))
    if tmp.exists():
        for p in tmp.iterdir():
            p.chmod(stat.S_IRUSR | stat.S_IWUSR)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    checks = {}

    # 1. The baseline gate is a real measurement on the real frozen files.
    gate = baseline_gate(enforce=False)
    checks["baseline_gate"] = gate
    assert gate["phase2"]["matches_spec"], gate["phase2"]

    # 2. Resume: a shard with 6 of 10 rows must decode only the missing 4, and
    #    must return rows in ITEM order, not completion order.
    class FakeCore:
        TARGET_SR = 16000

        @staticmethod
        def load_audio(p):
            return np.zeros(16000, dtype=np.float32), None

        @staticmethod
        def chunks(a):
            return [a], 0

        @staticmethod
        def decode(model, pieces):
            return model["texts"].pop(0)

    items = [(f"clip_{i:03d}.wav", tmp / f"clip_{i:03d}.wav", f"রেফ {i} শব্দ")
             for i in range(10)]
    shard = tmp / "resume.jsonl"
    with open(shard, "w", encoding="utf-8") as fh:
        for i in range(6):
            fh.write(json.dumps({"file": f"clip_{i:03d}.wav",
                                 "reference": f"রেফ {i} শব্দ",
                                 "transcript": f"রেফ {i} শব্দ",
                                 "n_pieces": 1, "forced_cuts": 0,
                                 "downmix_note": None, "transcribe_s": 0.01,
                                 "end_to_end_s": 0.02, "audio_seconds": 1.0},
                                ensure_ascii=False) + "\n")
    fake_model = {"texts": [f"রেফ {i} ভুল" for i in range(6, 10)]}
    out = decode_items(fake_model, items, FakeCore, shard, resume=True)
    assert len(out) == 10 and not fake_model["texts"], out
    assert [r["file"] for r in out] == [k for k, _, _ in items], "order not restored"
    assert out[0]["transcript"] == "রেফ 0 শব্দ" and out[9]["transcript"].endswith("ভুল")
    checks["resume_skips_completed_and_restores_order"] = True

    # 3. Bootstrap wiring on those rows: candidate (perfect on 6, wrong word on
    #    4) vs a baseline that is wrong everywhere -> CI entirely below 0.
    we, wn, ce, cn, keys = score_rows(out)
    base = [{"reference": r["reference"], "transcript": "ভুল ভুল ভুল"} for r in out]
    be, bn, _, _, _ = score_rows(base)
    assert (bn == wn).all()
    b = paired_bootstrap(we, be, wn, keys, 2000, SEED)
    assert b["ci_upper_below_zero"] and b["delta_pp_a_minus_b"] < 0, b
    checks["bootstrap_wiring"] = b

    # 4. Marker + read-only sealing + the second-decode refusal.
    rep = {"ckpt": "synthetic", "ckpt_sha256": "0" * 64, "wer_frozen_pp": 1.0}
    (tmp / "final_test.json").write_text(json.dumps(rep), encoding="utf-8")
    marker = seal_bundle(tmp, rep)
    assert (tmp / MARKER).exists() and "resume.jsonl" in marker["files"]
    mode = stat.S_IMODE((tmp / "final_test.json").stat().st_mode)
    assert not (mode & stat.S_IWUSR), f"bundle file still writable: {oct(mode)}"
    try:
        check_not_already_done(tmp, force=False)
    except SystemExit as exc:
        checks["second_decode_refused"] = str(exc)[:140]
    else:
        raise AssertionError("a completed bundle did not refuse a second decode")
    check_not_already_done(tmp, force=True)   # --force is the only way past
    checks["force_overrides"] = True
    checks["marker_files"] = sorted(marker["files"])

    # 4b. Identity + phase-level event on synthetic fixtures (council Q #9).
    ident = {"event_id": "e0", "mode": "FINAL_TEST",
             "checkpoint": {"path": "/x/step_1.nemo", "sha256": "a" * 64},
             "decoder": {"strategy": "greedy_ctc"},
             "scorer": {"surface": "frozen_norm"},
             "panels": {"test_manifest": {"rows": 920}}}
    other = json.loads(json.dumps(ident))
    other["checkpoint"]["sha256"] = "b" * 64
    idir = tmp / "ident"
    idir.mkdir()
    bind_identity(idir, ident, idir / "predictions.jsonl")
    require_identity_match(idir, ident)                 # matching resume: ok
    try:
        require_identity_match(idir, other)
    except SystemExit as exc:
        assert "sha256" in str(exc), exc
        checks["resume_with_other_ckpt_refused"] = str(exc)[:140]
    else:
        raise AssertionError("a different checkpoint sha resumed")
    reg = tmp / "event.json"
    open_event(reg, idir, ident, override=False)
    open_event(reg, idir, ident, override=False)        # same event: re-enter
    try:
        open_event(reg, tmp / "ident2", other, override=False)
    except SystemExit as exc:
        assert "directory" in str(exc)
        checks["second_event_refused"] = str(exc)[:200]
    else:
        raise AssertionError("a fresh output directory minted a second event")

    # 5. The sealed directory is never resolvable from the dry-run item set.
    dry_items, _ = gather_dry_items(3)
    for _, p, _ in dry_items:
        rp = Path(p).resolve()
        assert rp != SEALED_TEST_DIR and SEALED_TEST_DIR not in rp.parents
    checks["dry_items_outside_sealed_dir"] = [str(p) for _, p, _ in dry_items]
    return {"tool": "bnasr/final_test.py --selftest", "dir": str(tmp),
            "checks": checks}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearse on the LOCKED DEV panel; the sealed test set "
                         "is never opened")
    ap.add_argument("--dry-n", type=int, default=8)
    ap.add_argument("--draws", type=int, default=DRAWS_FINAL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing prediction shard and decode again")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a completed bundle — SPEC §6 allows no "
                         "second decode, so this needs a reason")
    ap.add_argument("--ack-p1-spec-discrepancy", action="store_true",
                    help="proceed with the ONE decode although the frozen "
                         "the baseline hypotheses do not re-score to SPEC §6's 19.57 pp")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print(json.dumps(selftest(), indent=1, ensure_ascii=False))
        return
    if not a.ckpt:
        raise SystemExit("--ckpt is required")
    if a.device.startswith("cuda"):
        pin_gpu()

    out_dir = Path(a.out_dir)
    if a.dry_run and out_dir == OUT_DIR:
        # A rehearsal must never land on the real result's path: it would both
        # trip the completion marker when the real event finally runs, and leave
        # a file that reads like the sealed-test result.
        out_dir = REPO / "experiments/final_test_dryrun"

    rep = run(a.ckpt, "dry_run" if a.dry_run else "FINAL_TEST", out_dir,
              dry_n=a.dry_n, device=a.device, force=a.force,
              ack_p1=a.ack_p1_spec_discrepancy, draws=a.draws,
              resume=not a.no_resume)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
