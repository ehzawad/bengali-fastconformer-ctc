#!/usr/bin/env python
"""Council Q defect 9: the one-shot held evaluator must not mix checkpoints.

    .venv/bin/python experiments/p4/tests/test_final_test_identity.py

CPU only, and deliberately blind: no GPU is opened, no model is loaded, no
audio is read and NOTHING under the sealed FLEURS directory is touched. Every
fixture here is synthetic and lives under a temp directory - the machinery
being checked is pure file bookkeeping, which is exactly the part that decided,
before this fix, whether 920 sealed recordings could be labelled with a
checkpoint that did not produce them.

The defect, as council Q stated it (SPEC §8w.7, defect 9): decode 700
recordings with checkpoint X, crash, restart the SAME output directory with
checkpoint Y, and ``decode_items`` - which resumed BY FILENAME ALONE - kept X's
700 predictions, decoded Y's remaining 220, and the report carried Y's hash.
Its completion marker was output-directory-local and its refusal text suggested
another directory, so ``--out-dir /somewhere/else`` minted a second
authorisation for an evaluation SPEC §6 permits once.

Its required fix, and therefore the coverage map here:
  identity_*   - immutable checkpoint/decoder/scorer/panel identity written
                 BEFORE the first prediction, matched exactly on resume
  resume_*     - a matching resume is accepted; a different checkpoint sha is
                 refused, by name
  shard_*      - a partial prediction set from a different checkpoint is never
                 reused
  event_*      - ONE phase-level event outside the output directory; a fresh
                 output directory is not a second authorisation, and the
                 refusal text does not offer one
  lock_*       - exclusive evaluation lock

Conventions are test_main_run_gate.py's (check/in_child/fresh); that file is
owned by another workstream and is neither imported nor edited here.
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from tools.p4 import p4_final_test as FT                 # noqa: E402

RESULTS = []


# --------------------------------------------------------------------------
# harness (test_main_run_gate.py's)
# --------------------------------------------------------------------------
def check(name, passed, evidence):
    RESULTS.append({"name": name, "passed": bool(passed) if passed is not None
                    else None, "evidence": evidence})
    tag = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    print(f"{tag}  {name}\n      {evidence}", flush=True)
    return bool(passed)


def in_child(fn, *a, **kw):
    """Run `fn` in a forked child and return (exit_code, captured_output)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # child
        try:
            os.close(r)
            os.dup2(w, 1)
            os.dup2(w, 2)
            os.close(w)
            fn(*a, **kw)
            sys.stdout.flush()
            os._exit(0)
        except SystemExit as exc:
            print(exc)
            sys.stdout.flush()
            os._exit(4)
        except BaseException:                       # noqa: BLE001
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(99)
    os.close(w)
    with os.fdopen(r, "rb") as fh:
        out = fh.read().decode("utf-8", "replace")
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out


def fresh(tag):
    return Path(tempfile.mkdtemp(prefix=f"p4ft_{tag}_"))


def refused(fn, *a, **kw):
    """(was_refused, message) for a call that must raise SystemExit."""
    try:
        fn(*a, **kw)
    except SystemExit as exc:
        return True, str(exc)
    return False, ""


# --------------------------------------------------------------------------
# synthetic fixtures - no real checkpoint, no panel, no audio
# --------------------------------------------------------------------------
def ident(ckpt_sha="a" * 64, event_id="EVENT0", draws=20000, rows=920):
    """An identity of the shape build_identity() produces, all-synthetic."""
    return {
        "event_id": event_id,
        "mode": "FINAL_TEST",
        "spec": "experiments/p4/SPEC.md §6 — one decode of the sealed FLEURS test",
        "checkpoint": {"path": "/synthetic/candidates/step_120000.nemo",
                       "sha256": ckpt_sha},
        "decoder": {"strategy": "greedy_ctc", "search": "none", "batch_size": 1,
                    "segment_seconds": 25, "asr_core_sha256": "c" * 64},
        "scorer": {"surface": "tools/p3_common.frozen_norm",
                   "p3_common_sha256": "d" * 64,
                   "bootstrap": {"draws": draws, "seed": 20260905}},
        "panels": {"test_manifest": {"path": "/synthetic/eval_manifest.json",
                                     "sha256": "e" * 64, "rows": rows,
                                     "rows_expected": 920}},
    }


class FakeCore:
    """asr_core's shape, with no audio and no model."""
    TARGET_SR = 16000

    @staticmethod
    def load_audio(p):
        import numpy as np
        return np.zeros(16000, dtype="float32"), None

    @staticmethod
    def chunks(a):
        return [a], 0

    @staticmethod
    def decode(model, pieces):
        return model["texts"].pop(0)


def items(n, root):
    return [(f"clip_{i:03d}.wav", root / f"clip_{i:03d}.wav", f"রেফ {i} শব্দ")
            for i in range(n)]


def shard_rows(path, n, identity_sha):
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps(
                {"file": f"clip_{i:03d}.wav", "reference": f"রেফ {i} শব্দ",
                 "transcript": f"রেফ {i} শব্দ", "identity_sha256": identity_sha,
                 "n_pieces": 1, "forced_cuts": 0, "downmix_note": None,
                 "transcribe_s": 0.01, "end_to_end_s": 0.02,
                 "audio_seconds": 1.0}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# (identity) frozen before the first prediction, matched exactly on resume
# --------------------------------------------------------------------------
def test_identity():
    d = fresh("ident")
    i = ident()
    got, minted = FT.bind_identity(d, i, d / "predictions.jsonl")
    check("identity_manifest_is_written_before_the_first_prediction",
          minted and (d / FT.IDENTITY).exists() and got == i
          and not (d / "predictions.jsonl").exists(),
          f"{d / FT.IDENTITY} written, no prediction shard yet")

    saved = json.loads((d / FT.IDENTITY).read_text(encoding="utf-8"))
    need = {"event_id", "checkpoint", "decoder", "scorer", "panels"}
    check("identity_records_ckpt_sha_decoder_scorer_panels_and_event_id",
          need <= set(saved) and saved["checkpoint"]["sha256"] == "a" * 64
          and saved["panels"]["test_manifest"]["rows"] == 920,
          f"keys={sorted(saved)}; ckpt sha {saved['checkpoint']['sha256'][:8]}…; "
          f"panel rows {saved['panels']['test_manifest']['rows']}")

    # every identity field is load-bearing: change one, the digest moves
    base = FT.identity_digest(i)
    moved = {}
    for path, val in (("checkpoint.sha256", "b" * 64),
                      ("decoder.strategy", "beam"),
                      ("scorer.p3_common_sha256", "f" * 64),
                      ("panels.test_manifest.rows", 919),
                      ("event_id", "EVENT1")):
        v = json.loads(json.dumps(i))
        keys = path.split(".")
        cur = v
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
        moved[path] = FT.identity_digest(v) != base
    check("identity_digest_covers_every_frozen_field",
          all(moved.values()), json.dumps(moved))

    # a directory with output but no identity cannot be resumed at all
    d2 = fresh("unbound")
    (d2 / "predictions.jsonl").write_text('{"file": "clip_000.wav"}\n',
                                          encoding="utf-8")
    ref, msg = refused(FT.bind_identity, d2, i, d2 / "predictions.jsonl")
    check("identity_absent_but_predictions_present_is_refused",
          ref and "not be reused" in msg, msg[:160])

    ref, msg = refused(FT.require_identity_match, fresh("empty"), i)
    check("identity_missing_manifest_refuses_a_resume",
          ref and "before the first prediction" in msg, msg[:160])


# --------------------------------------------------------------------------
# (resume) matching identity accepted, different checkpoint refused by name
# --------------------------------------------------------------------------
def test_resume():
    d = fresh("resume")
    i = ident()
    FT.bind_identity(d, i, d / "predictions.jsonl")

    got, minted = FT.bind_identity(d, ident(), d / "predictions.jsonl")
    check("resume_with_a_matching_identity_is_accepted",
          got == i and not minted,
          "second bind_identity of the identical identity returned the record")

    ref, msg = refused(FT.bind_identity, d, ident(ckpt_sha="b" * 64),
                       d / "predictions.jsonl")
    check("resume_with_a_different_checkpoint_sha_is_refused",
          ref and "checkpoint.sha256" in msg, msg[:200])
    check("resume_refusal_names_the_mismatched_field_and_both_values",
          ref and "a" * 64 in msg and "b" * 64 in msg,
          "refusal quotes recorded and current sha256")

    for label, other in (("decoder", {"decoder": {"strategy": "beam"}}),
                         ("panel_rows", {"panels": {"test_manifest": {"rows": 919}}}),
                         ("scorer", {"scorer": {"p3_common_sha256": "f" * 64}})):
        v = json.loads(json.dumps(i))
        for k, sub in other.items():
            for kk, vv in sub.items():
                if isinstance(vv, dict):
                    v[k][kk].update(vv)
                else:
                    v[k][kk] = vv
        r, m = refused(FT.require_identity_match, d, v)
        check(f"resume_with_a_different_{label}_is_refused", r, m[:140])


# --------------------------------------------------------------------------
# (shard) a partial prediction set from another checkpoint is never reused
# --------------------------------------------------------------------------
def test_shard():
    d = fresh("shard")
    x, y = ident(ckpt_sha="a" * 64), ident(ckpt_sha="b" * 64)
    xs, ys = FT.identity_digest(x), FT.identity_digest(y)
    it = items(10, d)

    # 6 rows decoded under X, then a restart under X: resume, decode only 4.
    shard = d / "predictions.jsonl"
    shard_rows(shard, 6, xs)
    model = {"texts": [f"রেফ {i} ভুল" for i in range(6, 10)]}
    out = FT.decode_items(model, it, FakeCore, shard, resume=True,
                          identity_sha=xs)
    check("shard_resume_under_the_same_identity_decodes_only_the_remainder",
          len(out) == 10 and not model["texts"]
          and [r["file"] for r in out] == [k for k, _, _ in it],
          f"{len(out)} rows returned in item order, 4 decoded")
    check("shard_rows_carry_the_identity_digest_that_produced_them",
          all(r["identity_sha256"] == xs for r in out),
          f"all 10 rows stamped {xs[:12]}…")

    # the council's sequence: same directory, checkpoint Y.
    d2 = fresh("mix")
    shard2 = d2 / "predictions.jsonl"
    shard_rows(shard2, 700, xs)
    it2 = items(920, d2)
    model2 = {"texts": [""] * 220}
    ref, msg = refused(FT.decode_items, model2, it2, FakeCore, shard2,
                       resume=True, identity_sha=ys)
    check("shard_partial_set_from_another_checkpoint_is_never_reused",
          ref and "never reused" in msg and len(model2["texts"]) == 220,
          msg[:200])
    check("shard_refusal_decodes_nothing_and_appends_nothing",
          ref and len(shard2.read_text(encoding="utf-8").splitlines()) == 700,
          "shard still holds exactly the 700 rows checkpoint X wrote")

    # an unstamped legacy shard is not silently adopted by an identified event
    d3 = fresh("legacy")
    shard3 = d3 / "predictions.jsonl"
    shard_rows(shard3, 5, None)
    ref, msg = refused(FT.decode_items, {"texts": []}, items(5, d3), FakeCore,
                       shard3, resume=True, identity_sha=xs)
    check("shard_rows_without_an_identity_are_refused_by_an_identified_event",
          ref and "None" in msg, msg[:160])


# --------------------------------------------------------------------------
# (event) one phase-level authorisation, outside the output directory
# --------------------------------------------------------------------------
def test_event():
    root = fresh("event")
    reg = root / "final_test_event.json"
    d1, d2 = root / "bundle1", root / "bundle2"
    d1.mkdir()
    d2.mkdir()
    i = ident()

    rec, minted = FT.open_event(reg, d1, i, override=False)
    check("event_record_is_minted_outside_the_output_directory",
          minted and reg.exists() and not (d1 / reg.name).exists()
          and rec["bundle_dir"] == str(d1.resolve())
          and rec["identity_sha256"] == FT.identity_digest(i),
          f"{reg} holds event {rec['event_id']} for {rec['bundle_dir']}")

    rec2, minted2 = FT.open_event(reg, d1, i, override=False)
    check("event_resume_of_the_same_event_and_identity_is_accepted",
          not minted2 and rec2["event_id"] == rec["event_id"],
          "same directory + same identity re-enters the one event")

    ref, msg = refused(FT.open_event, reg, d2, i, override=False)
    check("event_second_event_in_a_fresh_output_directory_is_refused",
          ref and "SECOND" in msg, msg[:220])
    check("event_refusal_does_not_suggest_using_another_directory",
          ref and "does not create a new authorisation" in msg
          and "will not change this answer" in msg,
          "refusal states a new --out-dir is not an authorisation")
    check("event_refusal_names_the_override_and_decodes_nothing",
          ref and FT.EVENT_OVERRIDE_ENV in msg and "Nothing was decoded" in msg,
          f"override is {FT.EVENT_OVERRIDE_ENV}=1 only")

    ref2, msg2 = refused(FT.open_event, reg, d1, ident(ckpt_sha="b" * 64),
                         override=False)
    check("event_same_directory_with_a_different_identity_is_refused",
          ref2 and "DIFFERENT identity" in msg2, msg2[:200])

    # the env var is the ONLY way to a second authorisation, and it is recorded
    before = json.loads(reg.read_text(encoding="utf-8"))
    os.environ[FT.EVENT_OVERRIDE_ENV] = "1"
    try:
        rec3, minted3 = FT.open_event(reg, d2, ident(event_id="EVENT1"))
    finally:
        os.environ.pop(FT.EVENT_OVERRIDE_ENV, None)
    check("event_override_env_is_required_and_supersession_is_recorded",
          minted3 and rec3["supersedes"]["event_id"] == before["event_id"]
          and rec3["bundle_dir"] == str(d2.resolve()),
          f"event {rec3['event_id']} supersedes {before['event_id']}")

    # closing marks the one event finished with what it produced
    closed = FT.close_event(reg, ident(event_id="EVENT1"),
                            {"wer_frozen_pp": 15.5, "ckpt_sha256": "a" * 64})
    check("event_close_records_completion_on_the_phase_level_record",
          closed["status"] == "completed" and closed["wer_frozen_pp"] == 15.5
          and json.loads(reg.read_text(encoding="utf-8"))["status"] == "completed",
          f"status={closed['status']} wer={closed['wer_frozen_pp']}")

    r, m = refused(FT.close_event, reg, ident(ckpt_sha="b" * 64),
                   {"wer_frozen_pp": 1.0})
    check("event_close_refuses_an_identity_that_is_not_the_registered_one",
          r, m[:140])

    # the completion marker's refusal no longer offers another directory
    d3 = root / "bundle3"
    d3.mkdir()
    (d3 / FT.MARKER).write_text("{}", encoding="utf-8")
    r, m = refused(FT.check_not_already_done, d3, False)
    check("event_completion_marker_refusal_does_not_suggest_another_directory",
          r and "is NOT a way around this" in m
          and "Point --out-dir at a" not in m, m[:200])


# --------------------------------------------------------------------------
# (lock) one evaluator at a time, across output directories
# --------------------------------------------------------------------------
def test_lock():
    root = fresh("lock")
    reg = root / "final_test_event.json"
    lock = FT.EvaluationLock(reg).acquire()

    def child():
        FT.EvaluationLock(reg).acquire()
        print("CHILD ACQUIRED")

    code, out = in_child(child)
    check("lock_a_second_evaluator_is_refused_while_the_lock_is_held",
          code == 4 and "CHILD ACQUIRED" not in out
          and "exclusive lock" in out, out.strip().splitlines()[-1][:200]
          if out.strip() else f"exit {code}")
    check("lock_names_the_holding_pid",
          f"pid {os.getpid()}" in out, f"holder recorded as pid {os.getpid()}")

    lock.release()
    code, out = in_child(child)
    check("lock_is_released_for_the_next_evaluator",
          code == 0 and "CHILD ACQUIRED" in out, f"exit {code}")


# --------------------------------------------------------------------------
# (wiring) the shipped run path uses all of it
# --------------------------------------------------------------------------
def test_wiring():
    src = (REPO / "tools/p4/p4_final_test.py").read_text(encoding="utf-8")
    check("wiring_the_event_registry_is_outside_every_bundle_directory",
          FT.event_registry_path().parent == (REPO / "experiments/p4")
          and FT.OUT_DIR not in FT.event_registry_path().parents,
          f"registry {FT.event_registry_path()} vs bundle {FT.OUT_DIR}")
    check("wiring_decode_items_requires_an_identity_digest",
          "identity_sha=ident_sha" in src,
          "run() passes the frozen digest into decode_items")
    check("wiring_identity_is_bound_before_the_model_is_loaded",
          src.index("bind_identity(bundle_dir, ident, shard)")
          < src.index("model = load_ctc_model"),
          "bind_identity precedes load_ctc_model in _run_locked")
    check("wiring_the_final_run_holds_the_exclusive_lock",
          "EvaluationLock(registry).acquire()" in src
          and "lock.release()" in src, "run() acquires and releases the lock")

    # build_identity against real repo-side files, if they are present; this
    # reads TEXT manifests only and never the sealed audio directory.
    tmp = fresh("build")
    ck = tmp / "step_000010.nemo"
    ck.write_bytes(b"not a real checkpoint")
    have = all(p.exists() for p in (FT.BENCH / "asr_core.py",
                                    FT.BENCH / "bench_score.py",
                                    FT.DEV_CORPUS))
    if not have:
        check("wiring_build_identity_records_real_panel_row_counts", None,
              "bench assets or dev corpus absent on this box")
        return
    i = FT.build_identity(ck, "dry_run", "EVENTX")
    check("wiring_build_identity_records_real_panel_row_counts",
          i["checkpoint"]["sha256"] == FT.sha256_file(ck)
          and i["panels"]["dev_corpus"]["rows"] > 0
          and i["decoder"]["strategy"] == "greedy_ctc",
          f"dev_corpus rows={i['panels']['dev_corpus']['rows']}, "
          f"ckpt sha {i['checkpoint']['sha256'][:12]}…")
    check("wiring_build_identity_never_names_the_sealed_audio_directory",
          str(FT.SEALED_TEST_DIR) not in json.dumps(i),
          "no path under eval_fleurs_bn appears in a dry-run identity")


# --------------------------------------------------------------------------
# SPEC 8w.11 (council R): the three ways the "one decode" rule could still be
# broken - re-decoding under --no-resume, re-entering a COMPLETED event, and
# reaching a protected one-shot panel through the other evaluator.
# --------------------------------------------------------------------------
def test_one_shot_is_really_one_shot():
    root = fresh("oneshot")

    # 1. --no-resume over an existing shard would append a second row per clip
    shard = root / "shard.jsonl"
    shard.write_text(json.dumps({"id": "clip1", "hyp": "x",
                                 "identity_sha256": "abc"}) + "\n", encoding="utf-8")
    before = shard.read_text(encoding="utf-8")
    refused = False
    try:
        FT.decode_items(None, [("clip1", "a.wav", "ref")], None, shard, resume=False)
    except SystemExit as exc:
        refused = "duplicates" in str(exc) or "already holds" in str(exc)
    check("no_resume_over_existing_predictions_is_refused",
          refused and shard.read_text(encoding="utf-8") == before,
          "one clip would have become two rows and the score computed over duplicates; "
          "the shard is untouched")

    # 2. a COMPLETED event is not re-enterable, even with the same identity/dir
    reg = root / "final_test_event.json"
    bundle = root / "bundle"; bundle.mkdir()
    i = ident()
    rec, minted = FT.open_event(reg, bundle, i, override=False)
    again, m2 = FT.open_event(reg, bundle, i, override=False)
    check("an_interrupted_event_can_still_be_resumed",
          minted and not m2 and again["event_id"] == rec["event_id"],
          "re-entry exists to finish an interrupted decode")
    done = json.loads(reg.read_text(encoding="utf-8")); done["status"] = "complete"
    done["wer"] = 18.4
    reg.write_text(json.dumps(done), encoding="utf-8")
    closed = False
    try:
        FT.open_event(reg, bundle, i, override=False)
    except SystemExit as exc:
        closed = "COMPLETE" in str(exc) and "Nothing was decoded" in str(exc)
    check("a_completed_event_cannot_be_re_entered",
          closed, "--force --no-resume previously allowed a second decode of the sealed set")
    ok, _ = FT.open_event(reg, bundle, i, override=True)
    check("the_owner_override_still_reopens_a_completed_event",
          ok["event_id"] == rec["event_id"],
          "an authorised reopen is possible, and it is explicit")


def test_qualified_selection_gate():
    root = fresh("qual")
    ck = root / "sel.nemo"; ck.write_bytes(b"weights")
    other = root / "other.nemo"; other.write_bytes(b"different")
    reg = root / "qualified_selection.json"
    orig = FT.QUALIFIED_SELECTION
    FT.QUALIFIED_SELECTION = reg
    try:
        refused = False
        try:
            FT.require_qualified_selection(ck)
        except SystemExit as exc:
            refused = "no qualified-selection record" in str(exc)
        check("no_qualified_selection_record_refuses_the_sealed_decode", refused,
              "the held decode is authorised by a qualified NORMAL ending, not by --ckpt")

        base = {"ending": "normal", "qualified": True,
                "selected_checkpoint": str(ck),
                "selected_sha256": FT.sha256_file(ck)}
        reg.write_text(json.dumps(base), encoding="utf-8")
        check("a_qualified_normal_ending_authorises_the_selected_checkpoint",
              FT.require_qualified_selection(ck)["qualified"] is True,
              "the one authorised path")

        for name, patch, needle in (
                ("a_terminal_ending_is_refused", {"ending": "terminal"}, "not 'normal'"),
                ("a_failed_qualification_is_refused", {"qualified": False}, "did not pass qualification")):
            reg.write_text(json.dumps(dict(base, **patch)), encoding="utf-8")
            r = False
            try:
                FT.require_qualified_selection(ck)
            except SystemExit as exc:
                r = needle in str(exc)
            check(name, r, "Nothing was decoded")

        reg.write_text(json.dumps(base), encoding="utf-8")
        swapped = False
        try:
            FT.require_qualified_selection(other)
        except SystemExit as exc:
            swapped = "not the selected checkpoint" in str(exc)
        check("substituting_another_checkpoint_is_refused", swapped,
              "SPEC 8w.2: do not substitute another checkpoint to obtain a pass")
    finally:
        FT.QUALIFIED_SELECTION = orig


def test_protected_panels():
    # The gate appends every permitted decode to PROTECTED_DECODE_LOG. Left
    # pointed at the real file, this test wrote synthetic rows into the
    # registered audit log on 2026-09-13. The 2026-09-15 rewrite (SPEC 8w.19)
    # also found that the redirect was applied to the package module while the
    # test called a FRESH module instance whose paths were still real - and
    # that once the real event + qualified selection existed the "refused"
    # cases would authorise and write a real line. Every path the gate reads
    # or writes is redirected on the instance actually under test.
    import tempfile as _tf, pathlib as _pl, json as _json, os as _os
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p4_panel_eval_under_test", REPO / "tools/p4/p4_panel_eval.py")
    PE = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(PE)
    except SystemExit:
        pass
    tmp = _pl.Path(_tf.mkdtemp(prefix='p4t_gatelog_'))
    real_log, real_reg, real_q, real_lock = PE.PROTECTED_DECODE_LOG, PE.FINAL_EVENT_REGISTRY, PE.QUALIFIED_SELECTION, PE.PROTECTED_LOCK
    PE.PROTECTED_DECODE_LOG = tmp / 'decodes.jsonl'
    PE.FINAL_EVENT_REGISTRY = tmp / 'absent_event.json'
    PE.QUALIFIED_SELECTION = tmp / 'absent_qualified.json'
    saved_env = {k: _os.environ.pop(k, None) for k in (PE.PROTECTED_OVERRIDE_ENV, PE.PROTECTED_REPEAT_ENV)}
    try:
        check("ordinary_panels_need_no_authorisation",
              PE.authorise_protected_panels(["D", "M", "K"], []) is None,
              "the selection panels are unaffected")
        for name, panels, mans in (
                ("holdout_by_name_is_refused_without_an_authorised_event", ["holdout"], []),
                ("a_vaani_split_via_manifest_override_is_refused", ["D"], ["/x/data4/vaani/manifest_test.jsonl"]),
                ("a_spring_split_via_manifest_override_is_refused", ["D"], ["/x/data4/spring_inx/R1/manifest_validation.jsonl"])):
            refused = False
            try:
                PE.authorise_protected_panels(panels, mans, argv=[])
            except SystemExit as exc:
                refused = "PROTECTED one-shot panel" in str(exc) and "Nothing was decoded" in str(exc)
            check(name, refused, "SPEC §2: decoded once, after final selection")
        check("refusal_writes_no_audit_record", not PE.PROTECTED_DECODE_LOG.exists(),
              "a refused request must not look like a decode")
        check("the_gate_is_the_same_record_the_held_decode_mints",
              real_reg == REPO / "experiments/p4/final_test_event.json",
              "'the held decode is authorised' has ONE meaning in Phase 4, not two")

        # --- 8w.19 (council AB hardening): intent / started / fact / correction records ---
        import hashlib as _hl
        PE.PROTECTED_LOCK = tmp / "decodes.lock"
        PE.FINAL_EVENT_REGISTRY.write_text(_json.dumps(
            {"event_id": "ev1", "status": "completed", "ckpt_sha256": "abc"}), encoding="utf-8")
        PE.QUALIFIED_SELECTION.write_text(_json.dumps(
            {"ending": "normal", "qualified": True, "selected_sha256": "abc"}), encoding="utf-8")
        rec = PE.authorise_protected_panels(["vaani_test"], [], argv=["x"], ckpt_sha256="abc", run_id="r1")
        check("authorised_request_writes_an_intent_record",
              rec is not None and rec["kind"] == "requested" and rec["prior_decodes"] == {"vaani_test": 0}
              and rec["run_id"] == "r1" and rec["repeat_approval"] is None,
              "the audit log distinguishes intent from fact")
        check("an_intent_record_is_not_a_decode",
              PE.prior_decode_counts(["vaani_test"]) == {"vaani_test": 0},
              "the 08:12Z line claimed seven decodes; two happened")
        PE.log_protected_started("vaani_test", {"rows_used": 3, "manifest_sha256": "m"}, "abc", "r1")
        c, unres = PE.prior_decode_counts(["vaani_test"], with_unresolved=True)
        check("a_started_record_spends_the_exposure_and_is_unresolved_until_decoded",
              c == {"vaani_test": 1} and unres == {"vaani_test"},
              "council AB: a crash between decode and the fact record must not read as never decoded")
        blocked = False
        try:
            PE.authorise_protected_panels(["vaani_test"], [], argv=[], ckpt_sha256="abc", run_id="r2")
        except SystemExit as exc:
            blocked = "no 'decoded' record" in str(exc) and "Nothing was decoded" in str(exc)
        check("an_unresolved_start_blocks_even_before_the_repeat_question", blocked, "reconcile first")
        PE.log_protected_decoded("vaani_test", {"n": 3, "wer_frozen": 0.1}, {"rows_used": 3, "manifest_sha256": "m"}, "abc", run_id="r1")
        c, unres = PE.prior_decode_counts(["vaani_test"], with_unresolved=True)
        check("the_decoded_record_of_the_same_run_resolves_its_start_and_counts_once",
              c == {"vaani_test": 1} and unres == set(), "fact record")
        check("a_non_protected_panel_writes_no_fact_record",
              PE.log_protected_decoded("D", {}, {}, "abc") is None and PE.log_protected_started("D", {}, "abc", "r") is None,
              "D is not one-shot")
        repeat_refused = False
        try:
            PE.authorise_protected_panels(["vaani_test"], [], argv=[], ckpt_sha256="abc", run_id="r3")
        except SystemExit as exc:
            repeat_refused = "already decoded" in str(exc) and "Nothing was decoded" in str(exc)
        check("a_repeat_decode_of_a_one_shot_is_an_ABORT_not_a_warning", repeat_refused,
              "council AA 2026-09-13: 'repeats merely warn' was the weaker gate")
        # approval file: wrong checkpoint / missing panel / count exceeded / good
        ap = tmp / "approval.json"
        def _approve(**kw):
            d = {"panels": ["vaani_test"], "ckpt_sha256": "abc", "max_decodes_per_panel": 2,
                 "reason": "recovery", "approved_by": "owner", "utc": "T", "spec": "8w.19"}
            d.update(kw); ap.write_text(_json.dumps(d), encoding="utf-8"); _os.environ[PE.PROTECTED_REPEAT_ENV] = str(ap)
        for label, kw, needle in (
                ("an_approval_for_another_checkpoint_is_refused", {"ckpt_sha256": "zzz"}, "approval is for checkpoint"),
                ("an_approval_not_naming_the_panel_is_refused", {"panels": ["holdout"]}, "does not name vaani_test"),
                ("an_approval_whose_count_is_exhausted_is_refused", {"max_decodes_per_panel": 1}, "max_decodes_per_panel"),
                ("an_approval_missing_a_field_is_refused", {"approved_by": ""}, "missing approved_by")):
            _approve(**kw); refused = False
            try:
                PE.authorise_protected_panels(["vaani_test"], [], argv=[], ckpt_sha256="abc", run_id="r4")
            except SystemExit as exc:
                refused = needle in str(exc)
            check(label, refused, "council AB: an approval is scoped to panels, checkpoint and count")
        _approve()
        rec2 = PE.authorise_protected_panels(["vaani_test"], [], argv=[], ckpt_sha256="abc", run_id="r5")
        _os.environ.pop(PE.PROTECTED_REPEAT_ENV, None)
        check("a_scoped_approval_authorises_the_repeat_and_is_recorded_by_hash",
              rec2 is not None and rec2["repeat_approval"]["sha256"] == _hl.sha256(ap.read_bytes()).hexdigest()
              and rec2["prior_decodes"] == {"vaani_test": 1}, "the override is visible in the record")
        # legacy (pre-8w.19) line naming several panels + validated corrections
        evid = tmp / "crash.log"; evid.write_text("Traceback\n"); esha = _hl.sha256(evid.read_bytes()).hexdigest()
        legacy = _json.dumps({"utc": "T0", "pid": 7, "panels": ["holdout", "spring_r1_test", "spring_r2_test"]})
        with open(PE.PROTECTED_DECODE_LOG, "a", encoding="utf-8") as fh:
            fh.write(legacy + "\n")
        lsha = _hl.sha256(legacy.encode()).hexdigest()
        check("a_legacy_line_counts_as_decoded_for_every_panel_it_names",
              PE.prior_decode_counts(["holdout", "spring_r1_test", "spring_r2_test"]) == {"holdout": 1, "spring_r1_test": 1, "spring_r2_test": 1},
              "conservative by default")
        def _corr(**kw):
            d = {"kind": "correction", "target_sha256": lsha, "decoded": ["holdout"],
                 "not_decoded": ["spring_r1_test", "spring_r2_test"],
                 "evidence": {"path": str(evid), "sha256": esha}, "approved_by": "x", "reason": "y"}
            d.update(kw); return _json.dumps(d)
        def _with_line(line, names):
            snap = PE.PROTECTED_DECODE_LOG.read_text(encoding="utf-8")
            PE.PROTECTED_DECODE_LOG.write_text(snap + line + "\n", encoding="utf-8")
            try:
                return PE.prior_decode_counts(names)
            except SystemExit as exc:
                return str(exc)
            finally:
                PE.PROTECTED_DECODE_LOG.write_text(snap, encoding="utf-8")
        for label, kw, needle in (
                ("a_correction_that_does_not_partition_the_target_is_refused", {"not_decoded": ["spring_r1_test"]}, "exhaustively"),
                ("a_correction_with_overlapping_lists_is_refused", {"decoded": ["holdout", "spring_r1_test"]}, "exhaustively"),
                ("a_correction_whose_evidence_hash_is_wrong_is_refused", {"evidence": {"path": str(evid), "sha256": "0" * 64}}, "does not hash"),
                ("a_correction_without_evidence_is_refused", {"evidence": {}}, "no evidence"),
                ("a_correction_without_approver_is_refused", {"approved_by": ""}, "approved_by"),
                ("a_correction_naming_no_target_is_refused", {"target_sha256": None}, "names no target"),
                ("a_correction_targeting_an_unknown_line_is_refused", {"target_sha256": "f" * 64}, "no such record")):
            out = _with_line(_corr(**kw), ["holdout"])
            check(label, isinstance(out, str) and needle in out and "Nothing was decoded" in out,
                  "council AB Q3: a correction cannot launder a repeat")
        check("a_valid_correction_removes_only_the_panels_it_says_were_not_decoded",
              _with_line(_corr(), ["holdout", "spring_r1_test", "spring_r2_test"]) == {"holdout": 1, "spring_r1_test": 0, "spring_r2_test": 0},
              "the incident record is the only thing that can lower a count, and it names the line it corrects by hash")
        dec_line = [l for l in PE.PROTECTED_DECODE_LOG.read_text(encoding="utf-8").splitlines() if '"kind": "decoded"' in l][0]
        out = _with_line(_corr(target_sha256=_hl.sha256(dec_line.encode()).hexdigest(), decoded=[], not_decoded=["vaani_test"]), ["vaani_test"])
        check("a_correction_can_never_subtract_a_decoded_fact", isinstance(out, str) and "facts are never subtracted" in out, "facts stand")
        two = _corr() + "\n" + _corr(reason="again")
        out = _with_line(two, ["holdout"])
        check("two_active_corrections_for_one_target_fail_closed", isinstance(out, str) and "two active corrections" in out, "no ambiguity")
        out = _with_line("this is not json", ["holdout"])
        check("a_malformed_audit_line_fails_closed", isinstance(out, str) and "is not JSON" in out, "a count that skips lines fails open")
        # a registered protected manifest under another panel's name is refused outright
        alias_refused = False
        try:
            PE._protected_selected(["D"], [str(PE.PANELS["vaani_test"]["manifest"])])
        except SystemExit as exc:
            alias_refused = "under that name only" in str(exc)
        check("a_registered_one_shot_manifest_under_an_alias_is_refused", alias_refused, "council AB: facts are keyed by panel name")
        # frozen manifest identity
        saved = PE.FROZEN_ONE_SHOT_MANIFEST_SHA256["vaani_test"]
        PE.FROZEN_ONE_SHOT_MANIFEST_SHA256["vaani_test"] = "0" * 64
        moved = False
        try:
            PE.preflight_panels(["vaani_test"], {}, limit=None)
        except SystemExit as exc:
            moved = "not the registered panel" in str(exc)
        finally:
            PE.FROZEN_ONE_SHOT_MANIFEST_SHA256["vaani_test"] = saved
        check("a_protected_manifest_whose_hash_moved_is_refused_in_preflight", moved,
              "SPEC §2 'manifest hashes recorded at freeze' finally has a writer and a reader")
        check("the_frozen_hashes_match_the_manifests_on_disk",
              all(PE.sha256_file(PE.PANELS[n]["manifest"]) == h for n, h in PE.FROZEN_ONE_SHOT_MANIFEST_SHA256.items()),
              "frozen 2026-09-15 from the files as first decoded")

        # --- 8w.19: preflight finds a manifest collision before any GPU work ---
        rows = [{"audio_filepath": "/a.wav", "text": "x", "duration": 1.0},
                {"audio_filepath": "/b.wav", "text": "y", "duration": 1.0},
                {"audio_filepath": "/a.wav", "text": "z", "duration": 2.0}]
        check("manifest_collisions_lists_every_row_index_of_a_repeated_path",
              PE.manifest_collisions(rows) == {"/a.wav": [0, 2]}, "Vaani validation has three such pairs")
        man = tmp / "d_collide.jsonl"
        man.write_text("".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        pf = PE.preflight_panels(["D"], {"D": str(man)}, limit=None)
        check("preflight_reads_and_flags_before_the_model_is_loaded",
              pf["D"]["collisions"] == {"/a.wav": [0, 2]} and len(pf["D"]["rows"]) == 3, "zero-cost check")
        missing_fatal = False
        try:
            PE.preflight_panels(["vaani_test"], {"vaani_test": str(tmp / "nope.jsonl")})
        except SystemExit as exc:
            missing_fatal = "Nothing was decoded" in str(exc)
        check("a_missing_one_shot_manifest_is_fatal_in_preflight", missing_fatal,
              "a SKIP after the audit record would be a false exposure line")
    finally:
        PE.PROTECTED_DECODE_LOG, PE.FINAL_EVENT_REGISTRY, PE.QUALIFIED_SELECTION, PE.PROTECTED_LOCK = real_log, real_reg, real_q, real_lock
        for k, v in saved_env.items():
            if v is not None:
                _os.environ[k] = v


def main():
    for fn in (test_identity, test_resume, test_shard, test_event, test_lock,
               test_wiring, test_one_shot_is_really_one_shot,
               test_qualified_selection_gate, test_protected_panels):
        print(f"\n=== {fn.__name__} " + "=" * (60 - len(fn.__name__)))
        fn()
    n_pass = sum(1 for r in RESULTS if r["passed"] is True)
    n_fail = sum(1 for r in RESULTS if r["passed"] is False)
    n_skip = sum(1 for r in RESULTS if r["passed"] is None)
    out = HERE / "results_final_test_identity.json"
    out.write_text(json.dumps(
        {"checks": RESULTS, "passed": n_pass, "failed": n_fail,
         "skipped": n_skip,
         "code_sha256": {"p4_final_test.py":
                         FT.sha256_file(REPO / "tools/p4/p4_final_test.py")}},
        indent=1), encoding="utf-8")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped -> {out}")
    for r in RESULTS:
        if r["passed"] is False:
            print("FAILED: " + r["name"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
