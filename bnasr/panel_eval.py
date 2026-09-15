#!/usr/bin/env python3
"""this recipe panel evaluation: one CTC checkpoint, any subset of the registered
panels, BOTH scoring surfaces, per-source and per-style breakdowns, the SPEC §2
selection score S and the SPEC §2 regression guards.

Why this tool exists at all
---------------------------
this recipe selects checkpoints on a WEIGHTED score over four panels
(`S = 0.60·D + 0.20·M + 0.10·R + 0.10·K`, SPEC §2). A weighted score is exactly
the shape of number that can improve while the thing you care about gets worse:
a 3.5 h MaCro supplement is ~29 % of the panel hours (council B #10), so a
model that learns MaCro and forgets conversational IndicVoices can post a
better S. SPEC answers that with guards, not with trust — no D source worse
than the baseline by >1.0 pp, the conversational slice of D worse by ≤0.5 pp, empty
outputs ≤0.1 % — and this module computes S and the guards in ONE place so a
caller cannot report the score without also reporting what it hides.

The two scoring surfaces, and why neither may be dropped
-------------------------------------------------------
* ``wer_frozen`` — ``tools/scoring.frozen_norm``, byte-for-byte
  ``bench_assets/bench_score.norm``. This is the surface every published
  baseline on panel D is quoted on (P2 17.03 / P1 20.35 / hishab 22.30,
  SPEC §2), so it is the ONLY surface on which those comparisons are legal, and
  it is the surface S is defined on ("frozen-surface corpus WER, pp").
* ``wer_nemo`` — NeMo's own ``word_error_rate`` on RAW text. This is the
  surface the the baseline ``val_wer`` curve was logged on (the baseline dev D = 20.83 %),
  so it is the only surface comparable with that curve.

A third number, ``wer_p4_surface``, is reported for information only: it scores
on the surface SPEC §1 puts the TRAINING TARGETS through,
``metric_surface(NFC(t), False)`` + whitespace collapse. SPEC §1 applies that
transform to targets and SPEC §2 defines the panels on "the frozen surface";
these are two different functions (metric_surface casefolds, maps every Unicode
P* rather than a 20-character class, and preserves ZWNJ/ZWJ explicitly), so a
this recipe model trained on §1 targets is scored on the §2 surface and the gap
between the two is a fact about the corpus that must be visible, not a knob.
Selection uses ``wer_frozen``. Always.

Reuse, not reimplementation
---------------------------
``frozen_norm`` / ``wer_counts`` / ``cer_counts`` / ``sha256_file`` / ``pin_gpu``
/ ``anonymize_process`` are imported from ``bnasr/scoring.py``; the scoring
surface has been frozen since the baseline and a second copy of it in this file is
how two this recipe numbers end up under one label. ``scoring.dev_eval`` is NOT
reused: it is transducer-only (it reaches into ``model.joint`` to disable fused
loss/WER) and this phase's model is ``EncDecCTCModelBPE``, which has no joint.
The decode loop below is the CTC equivalent and nothing else about the scoring
differs.

Usage
-----
    CUDA_VISIBLE_DEVICES=GPU-3ce8... bnasr/panel_eval.py \
        --ckpt experiments/fastconformer_ctc_bn_1kh/final.nemo \
        --panels D --batch-size 32 \
        --out experiments/eval/p1_dev.json --save-hyps experiments/eval/p1_dev.hyps.json

    # record the the baseline reference the guards are measured against
    bnasr/panel_eval.py --ckpt .../final.nemo --panels D K R \
        --record-baseline
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import (  # noqa: E402
    anonymize_process, cer_counts, frozen_norm, load_manifest, nemo_wer,
    pin_gpu, sha256_file, wer_counts,
)

anonymize_process()
# Must be set before torch initialises its allocator (shared-box rule).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import defaultdict  # noqa: E402

from bnasr.normalize import metric_surface  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants. Every one of these is a transcription of SPEC.md; the
# assertions next to them are the only thing that keeps a number in this file
# from drifting away from the contract it claims to implement.
# ---------------------------------------------------------------------------

# SPEC §2: "S = 0.60·D + 0.20·M + 0.10·R + 0.10·K if MADASR is admitted, else
# S = 0.60·D + 0.30·M + 0.10·K". Both formulas are carried, and which one is
# used is decided by the PRESENCE OF PANEL R in the evaluation — never by a
# flag a caller can set to whichever gives the nicer number.
S_WEIGHTS_WITH_R = {"D": 0.60, "M": 0.20, "R": 0.10, "K": 0.10}
S_WEIGHTS_NO_R = {"D": 0.60, "M": 0.30, "K": 0.10}
assert abs(sum(S_WEIGHTS_WITH_R.values()) - 1.0) < 1e-12, "SPEC §2 S weights"
assert abs(sum(S_WEIGHTS_NO_R.values()) - 1.0) < 1e-12, "SPEC §2 S weights"
assert S_WEIGHTS_WITH_R["D"] == S_WEIGHTS_NO_R["D"] == 0.60, "SPEC §2: D is 0.60 in both"

# SPEC §2 "Regression guards": no D source worse than P1 by >1.0 pp;
# conversational slice of D worse by ≤0.5 pp; empty-output rate ≤0.1 %.
GUARDS = {
    "per_source_max_regression_pp": 1.0,
    "conversational_max_regression_pp": 0.5,
    "max_empty_rate": 0.001,
}
assert GUARDS["per_source_max_regression_pp"] == 1.0, "SPEC §2 guard: >1.0 pp"
assert GUARDS["conversational_max_regression_pp"] == 0.5, "SPEC §2 guard: ≤0.5 pp"
assert GUARDS["max_empty_rate"] == 0.001, "SPEC §2 guard: ≤0.1 %"

# SPEC §2, panel D row: "(P2 17.03 / P1 20.35 / hishab 22.30, frozen surface)".
# These are the published numbers on the 5,385-row locked panel; they are
# reported next to every D evaluation so a result is never read in isolation.
# Cross-checked against p3_probe/devpanel_baselines.json (phase1 0.20346).
DEV_D_PUBLISHED = {"phase2_qwen": 0.1703, "phase1_ctc": 0.2035, "hishab": 0.2230}
assert DEV_D_PUBLISHED["phase1_ctc"] == 0.2035, "SPEC §2: P1 D = 20.35 % frozen"

# SPEC §4 zero-update restoration check: "dev WER of the loaded model equals
# the baseline's recorded 20.35 % (frozen) ±0.05". The NeMo-internal companion
# (20.83 %) is the the baseline val_wer curve value, carried so the second surface
# is checked too rather than merely printed.
RESTORATION_TOLERANCE_PP = 0.05
P1_DEV_NEMO_INTERNAL = 0.2083

# SPEC §2 panel table. `role` is carried because a one-shot panel must never be
# silently folded into S: `holdout` and the conversational slice are reports.
PANELS = {
    "D": {"manifest": REPO / "data2/val_corpus.json", "rows": 5385,
          "role": "selection+historical"},
    "M": {"manifest": REPO / "experiments/corpus/panel_M.jsonl", "rows": None,
          "role": "selection"},
    "R": {"manifest": REPO / "data4/respin/manifest_madasr_dev.jsonl", "rows": 1500,
          "role": "selection (only if MADASR admitted)"},
    "K": {"manifest": REPO / "data4/kathbath/manifest_valid.jsonl", "rows": 2236,
          "role": "selection"},
    "holdout": {"manifest": REPO / "data2/holdout_corpus.json", "rows": 6767,
                "role": "one-shot report"},
    # SPEC 2 one-shots that had NO decode path in this tool until 2026-09-15
    # (found by the end-of-run review): the workflow's --panels command would
    # have exited 2 at argparse. Row counts are the files' actual line counts;
    # Vaani matches SPEC 1's "validation 10,105 / test 12,695" exactly. SPRING
    # ships validation+test for R1 and R2 - there is no "eval" split.
    "vaani_validation":    {"manifest": REPO / "data4/vaani/manifest_validation.jsonl",
                            "rows": 10105, "role": "one-shot report"},
    "vaani_test":          {"manifest": REPO / "data4/vaani/manifest_test.jsonl",
                            "rows": 12695, "role": "one-shot report"},
    "spring_r1_validation": {"manifest": REPO / "data4/spring_inx/R1/manifest_validation.jsonl",
                            "rows": 15055, "role": "one-shot report"},
    "spring_r1_test":      {"manifest": REPO / "data4/spring_inx/R1/manifest_test.jsonl",
                            "rows": 1879,  "role": "one-shot report"},
    "spring_r2_validation": {"manifest": REPO / "data4/spring_inx/R2/manifest_validation.jsonl",
                            "rows": 10936, "role": "one-shot report"},
    "spring_r2_test":      {"manifest": REPO / "data4/spring_inx/R2/manifest_test.jsonl",
                            "rows": 1872,  "role": "one-shot report"},
}
ONE_SHOT_PANELS = tuple(n for n, sp in PANELS.items() if sp["role"] == "one-shot report")
assert PANELS["D"]["rows"] == 5385 and PANELS["R"]["rows"] == 1500 \
    and PANELS["K"]["rows"] == 2236, "SPEC §2 panel row counts"

# ---- PROTECTED one-shot panels (council R, SPEC §8w.11) ---------------------
# final_test.py binds the sealed FLEURS decode to a single phase-level event
# with a frozen identity. `holdout` and the Vaani validation/test splits are
# ALSO protected one-shot panels (SPEC §2 and the Vaani row of §1's table:
# "validation 10,105 and test 12,695 are PROTECTED one-shot panels ... decoded
# only after final selection"), but this tool could decode them at any time,
# any number of times, with none of that machinery. A one-shot set scored more
# than once is no longer a held-out measurement: repeat it and you can pick the
# run that looks best, which is exactly what holding it out was meant to stop.
PROTECTED_ONE_SHOT = ONE_SHOT_PANELS   # every role=="one-shot report" panel, by name
# Matched case-insensitively against the FULL manifest path. The review of the
# end-of-run workflow (2026-09-15) found the previous list did not match the
# files that actually exist on disk - data4/vaani/manifest_{validation,test}.jsonl
# and data4/spring_inx/R{1,2}/manifest_{validation,test}.jsonl - so a --manifest
# override at a real path would have been decoded with NO gate and NO log entry.
# Every registered one-shot (SPEC 2) is listed by its real on-disk name, plus the
# earlier aliases so nothing that used to be caught stops being caught.
PROTECTED_MANIFEST_SUBSTRINGS = (
    "holdout_corpus",
    "vaani/manifest_validation", "vaani/manifest_test",
    "spring_inx/r1/manifest_validation", "spring_inx/r1/manifest_test",
    "spring_inx/r2/manifest_validation", "spring_inx/r2/manifest_test",
    # aliases kept from the earlier list
    "vaani_val", "vaani_test", "vaani/validation", "vaani/test",
    "spring_inx/validation", "spring_inx/eval", "spring_val", "spring_eval", "spring_test",
)
QUALIFIED_SELECTION = REPO / "experiments/qualified_selection.json"
FINAL_EVENT_REGISTRY = REPO / "experiments/final_test_event.json"
PROTECTED_OVERRIDE_ENV = "P4_ALLOW_PROTECTED_PANEL"
# Council AB (2026-09-15): a boolean override records nothing about WHO allowed
# WHAT. The env var now names an APPROVAL FILE: {"panels": [...], "ckpt_sha256",
# "max_decodes_per_panel", "reason", "approved_by", "utc", "spec"}; the gate
# checks the requested panels, the checkpoint and the count against it and
# records the file's sha in the audit record.
PROTECTED_REPEAT_ENV = "P4_PROTECTED_REPEAT_APPROVAL"
PROTECTED_LOCK = REPO / "experiments/protected_panel_decodes.lock"
# SPEC §2 promised "manifest hashes recorded at freeze" and nothing recorded
# them (the same reader-without-writer class as §8w.14/16/19). Frozen here on
# 2026-09-15 from the files as first decoded (mtimes 2026-08-29 / 2026-09-06,
# unchanged since). A protected panel whose manifest hash moved is not the
# registered panel and is refused.
FROZEN_ONE_SHOT_MANIFEST_SHA256 = {
    "holdout": "75e882289d2628bf7387c674dc7b4d7153b5d74d91e095c5ad7b8fc237560854",
    "vaani_validation": "d81cd0308611c4a7b559823871909a74a6d0e41a1e324f0e488bb81a4de3f0a1",
    "vaani_test": "a2bb5029b1e5d564f17fe381e2cad9344619a793e405967367dca67b85531efa",
    "spring_r1_validation": "9e9ceeb0bde6a808b076f0e7e239eaf2c6685e2b7f660b8565b3c01536363108",
    "spring_r1_test": "e6b6b2184975741e73a0bacba77c5026adbd66dbdc6a472ba3bca2e89ed9c0a4",
    "spring_r2_validation": "77016aeca9fbb5ddcfc61e8dcfa91367b49cf7954616b1e00db25e288325eb39",
    "spring_r2_test": "a8fc116581298c6b204f9b03df67af0f687899da3878fa4c68d24f19de984ddd",
}
PROTECTED_DECODE_LOG = REPO / "experiments/protected_panel_decodes.jsonl"


def _protected_selected(names, manifests=()):
    """Protected panels among a request, by registered name AND by manifest path.

    Council AB: a protected manifest passed as `--manifest D=<path>` used to be
    refused as a path string, so if it was ever allowed the fact record would
    have been written under "D". A registered one-shot manifest is only ever
    decodable under its registered name: an override that resolves to one is
    an ABORT here, whatever the gate would otherwise say. An unregistered path
    that merely looks protected (substring) is still refused by the gate.
    """
    hit = {n for n in names if n in PROTECTED_ONE_SHOT}
    registered = {Path(sp["manifest"]).resolve(): n for n, sp in PANELS.items()
                  if n in PROTECTED_ONE_SHOT}
    for mp in manifests:
        rp = Path(mp).resolve()
        if rp in registered:
            raise SystemExit(f"ABORT: {mp} is the registered manifest of the PROTECTED "
                             f"one-shot panel {registered[rp]!r}; decode it under that "
                             f"name only, so the audit log accounts for it. Nothing was decoded.")
        low = str(mp).lower()
        if any(sub in low for sub in PROTECTED_MANIFEST_SUBSTRINGS):
            hit.add(str(mp))
    return sorted(hit)


def authorise_protected_panels(names, manifests=(), argv=None, ckpt_sha256=None, run_id=None):
    """Refuse a protected one-shot panel unless the phase-level event allows it.

    The gate is the SAME record final_test.py mints, so "the held decode has
    been authorised" has one meaning in this recipe rather than two. Every permitted
    decode is appended to PROTECTED_DECODE_LOG, so a repeat is visible even when
    it was authorised.
    """
    import json as _json
    import os as _os
    import time as _time
    hit = _protected_selected(names, manifests)
    if not hit:
        return None
    override = _os.environ.get(PROTECTED_OVERRIDE_ENV) == "1"
    ev = None
    if FINAL_EVENT_REGISTRY.exists():
        try:
            ev = _json.loads(FINAL_EVENT_REGISTRY.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ev = None
    if ev is None and not override:
        raise SystemExit(
            f"ABORT: {', '.join(hit)} is a PROTECTED one-shot panel and no "
            f"authorised held-decode event exists ({FINAL_EVENT_REGISTRY} is "
            f"absent). SPEC §2: these are decoded once, after final selection. "
            f"Scoring a one-shot set more than once stops it being held out. "
            f"If this is genuinely authorised, the owner records why and sets "
            f"{PROTECTED_OVERRIDE_ENV}=1. Nothing was decoded.")
    # Council AA (2026-09-13): an event EXISTING is not the same as the FINAL,
    # QUALIFIED event existing. The early FLEURS read (SPEC 8w.14) minted an
    # event on an unqualified checkpoint, and this check used to accept it.
    # The one-shot panels are decoded against the QUALIFIED selection only.
    if ev is not None and not override:
        problems = []
        if not QUALIFIED_SELECTION.exists():
            problems.append(f"no qualified selection record at {QUALIFIED_SELECTION}")
        else:
            try:
                q = _json.loads(QUALIFIED_SELECTION.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                q = {}; problems.append(f"unreadable qualified selection: {exc}")
            if q.get("ending") != "normal" or not q.get("qualified"):
                problems.append("qualified selection is not a qualified normal ending")
            want = q.get("selected_sha256")
            got = ev.get("ckpt_sha256")
            if want and got and want != got:
                problems.append(f"event checkpoint {got[:12]}… is not the qualified "
                                f"selection {want[:12]}… (an earlier/unqualified event)")
        if ev.get("status") != "completed":
            problems.append(f"event status is {ev.get('status')!r}, not 'completed'")
        if problems:
            raise SystemExit(
                f"ABORT: {', '.join(hit)} is a PROTECTED one-shot panel and the "
                f"existing held-decode event does not authorise it: "
                + "; ".join(problems) +
                f". Set {PROTECTED_OVERRIDE_ENV}=1 only with the owner's recorded "
                f"reason. Nothing was decoded.")
    # 2026-09-15 incident (SPEC 8w.19): the first real one-shot invocation wrote
    # ONE line naming all seven panels, then crashed after decoding two of them.
    # The log said "seven decoded"; the truth was two decoded, results lost.
    # Now: a "requested" record here (intent), a "decoded" record per panel
    # AFTER its decode completes (fact, written by log_protected_decoded), and a
    # prior-decode count that reads facts, not intents. A prior decode of a
    # one-shot panel is an ABORT, not a warning - the owner records the reason
    # and sets P4_ALLOW_PROTECTED_REPEAT=1 (the checkpoint being frozen and the
    # decode deterministic is such a reason; "I wanted another look" is not).
    _acquire_protected_lock()
    prior, unresolved = prior_decode_counts(hit, with_unresolved=True)
    if unresolved:
        raise SystemExit(
            f"ABORT: {PROTECTED_DECODE_LOG} has a 'started' record with no 'decoded' "
            f"record for {', '.join(sorted(unresolved))} (a decode that began and "
            f"never completed). Exposure may have been spent; reconcile it with a "
            f"'correction' record naming that started record before any repeat. "
            f"Nothing was decoded.")
    repeated = {n: c for n, c in prior.items() if c > 0}
    approval = None
    if repeated:
        approval = _load_repeat_approval(repeated, ckpt_sha256)
    rec = {"kind": "requested", "run_id": run_id,
           "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
           "panels": hit, "pid": _os.getpid(), "ckpt_sha256": ckpt_sha256,
           "event_id": (ev or {}).get("event_id"),
           "event_status": (ev or {}).get("status"),
           "override": override,
           "repeat_approval": approval,
           "prior_decodes": prior, "argv": list(argv or [])}
    _append_protected_record(rec)
    if repeated:
        print(f"WARNING: repeat decode of "
              + ", ".join(f"{n} (×{c} before)" for n, c in sorted(repeated.items()))
              + f" under approval {approval['path']} (sha {approval['sha256'][:12]}…): "
              f"recovery repeat; no selection or tuning permitted.", flush=True)
    return rec


_LOCK_FD = None


def _acquire_protected_lock():
    """One protected decode per box at a time: two invocations passing the gate
    together would each count the other's exposure as zero."""
    global _LOCK_FD
    import fcntl
    if _LOCK_FD is not None:
        return   # already held by this process (flock is per open file description)
    PROTECTED_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PROTECTED_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit(f"ABORT: another protected-panel decode holds {PROTECTED_LOCK}. "
                         f"Nothing was decoded.")
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    _LOCK_FD = fd   # held for the life of the process; released by exit


def _load_repeat_approval(repeated, ckpt_sha256):
    import json as _json
    path = os.environ.get(PROTECTED_REPEAT_ENV)
    if not path:
        raise SystemExit(
            f"ABORT: already decoded per {PROTECTED_DECODE_LOG}: "
            + ", ".join(f"{n}×{c}" for n, c in sorted(repeated.items()))
            + f". A one-shot panel scored repeatedly is no longer a held-out "
            f"measurement. Re-decoding requires the owner's approval record: "
            f"{PROTECTED_REPEAT_ENV}=<path to JSON with panels, ckpt_sha256, "
            f"max_decodes_per_panel, reason, approved_by, utc, spec>. Nothing was decoded.")
    ap = Path(path)
    if not ap.exists():
        raise SystemExit(f"ABORT: {PROTECTED_REPEAT_ENV}={path} does not exist. Nothing was decoded.")
    try:
        a = _json.loads(ap.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ABORT: approval {path} is not JSON: {exc}. Nothing was decoded.")
    problems = []
    for k in ("panels", "ckpt_sha256", "max_decodes_per_panel", "reason", "approved_by", "utc", "spec"):
        if not a.get(k):
            problems.append(f"missing {k}")
    if a.get("ckpt_sha256") and a["ckpt_sha256"] != ckpt_sha256:
        problems.append(f"approval is for checkpoint {a['ckpt_sha256'][:12]}…, this is {ckpt_sha256[:12]}…")
    not_covered = sorted(set(repeated) - set(a.get("panels", [])))
    if not_covered:
        problems.append(f"approval does not name {', '.join(not_covered)}")
    over = {n: c for n, c in repeated.items() if c + 1 > int(a.get("max_decodes_per_panel") or 0)}
    if over:
        problems.append("would exceed max_decodes_per_panel for "
                        + ", ".join(f"{n} (prior {c})" for n, c in sorted(over.items())))
    if problems:
        raise SystemExit(f"ABORT: repeat approval {path} does not authorise this decode: "
                         + "; ".join(problems) + ". Nothing was decoded.")
    return {"path": str(ap.resolve()), "sha256": sha256_file(ap),
            "approved_by": a["approved_by"], "utc": a["utc"], "spec": a["spec"],
            "max_decodes_per_panel": int(a["max_decodes_per_panel"])}


def _append_protected_record(rec):
    import json as _json
    PROTECTED_DECODE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROTECTED_DECODE_LOG, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_protected_records():
    """Every record of the audit log, each tagged with the sha256 of its own line
    (`_line_sha256`) so a correction can name exactly the line it corrects.
    A line that is not JSON is an ABORT: an audit log that cannot be read in
    full cannot be counted, and a count that skips lines fails open."""
    import json as _json
    import hashlib as _hashlib
    if not PROTECTED_DECODE_LOG.exists():
        return []
    out = []
    for i, line in enumerate(open(PROTECTED_DECODE_LOG, encoding="utf-8"), 1):
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        try:
            rec = _json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"ABORT: {PROTECTED_DECODE_LOG} line {i} is not JSON ({exc}); "
                             f"the exposure count cannot be trusted. Nothing was decoded.")
        if not isinstance(rec, dict):
            raise SystemExit(f"ABORT: {PROTECTED_DECODE_LOG} line {i} is not an object. Nothing was decoded.")
        rec["_line_sha256"] = _hashlib.sha256(raw.encode("utf-8")).hexdigest()
        rec["_line_no"] = i
        out.append(rec)
    return out


def _active_corrections(recs):
    """Corrections that are valid and not superseded, keyed by target line sha.

    A correction must: name its target by `target_sha256` (a `requested`,
    `started` or legacy record — NEVER a `decoded` fact); partition the target's
    panels exhaustively and disjointly into `decoded` / `not_decoded`; carry
    `evidence` {path, sha256} that exists on disk and hashes as stated; carry
    `approved_by` and `reason`. Two active corrections for one target is an
    ABORT. A legacy correction without `target_sha256` must be superseded by a
    valid one (`supersedes_correction_sha256`), else ABORT.
    """
    by_sha = {r["_line_sha256"]: r for r in recs}
    corrections = [r for r in recs if r.get("kind") == "correction"]
    superseded = {c.get("supersedes_correction_sha256") for c in corrections
                  if c.get("supersedes_correction_sha256")}
    active = {}
    for c in corrections:
        if c["_line_sha256"] in superseded:
            continue
        tsha = c.get("target_sha256")
        if not tsha:
            raise SystemExit(f"ABORT: correction at {PROTECTED_DECODE_LOG} line {c['_line_no']} "
                             f"names no target_sha256 and is not superseded. Nothing was decoded.")
        t = by_sha.get(tsha)
        if t is None:
            raise SystemExit(f"ABORT: correction line {c['_line_no']} targets {tsha[:12]}…, "
                             f"no such record. Nothing was decoded.")
        if t.get("kind") == "decoded":
            raise SystemExit(f"ABORT: correction line {c['_line_no']} targets a 'decoded' fact "
                             f"record; facts are never subtracted. Nothing was decoded.")
        tpanels = set(t.get("panels", [])) if t.get("kind") != "started" else {t.get("panel")}
        d, nd = set(c.get("decoded", [])), set(c.get("not_decoded", []))
        if (d | nd) != tpanels or (d & nd):
            raise SystemExit(f"ABORT: correction line {c['_line_no']} does not partition the "
                             f"target's panels {sorted(tpanels)} exhaustively and disjointly "
                             f"(decoded={sorted(d)}, not_decoded={sorted(nd)}). Nothing was decoded.")
        ev = c.get("evidence") or {}
        if not (ev.get("path") and ev.get("sha256")):
            raise SystemExit(f"ABORT: correction line {c['_line_no']} carries no evidence "
                             f"{{path, sha256}}. Nothing was decoded.")
        ep = Path(ev["path"]) if Path(ev["path"]).is_absolute() else REPO / ev["path"]
        if not ep.exists() or sha256_file(ep) != ev["sha256"]:
            raise SystemExit(f"ABORT: correction line {c['_line_no']} evidence {ev['path']} is "
                             f"missing or does not hash as stated. Nothing was decoded.")
        if not (c.get("approved_by") and c.get("reason")):
            raise SystemExit(f"ABORT: correction line {c['_line_no']} lacks approved_by/reason. "
                             f"Nothing was decoded.")
        if tsha in active:
            raise SystemExit(f"ABORT: two active corrections target record {tsha[:12]}… "
                             f"(lines {active[tsha]['_line_no']} and {c['_line_no']}). Nothing was decoded.")
        active[tsha] = c
    return active


def prior_decode_counts(names, with_unresolved=False):
    """How many times each named panel has had its exposure SPENT, per the log.

    Record kinds: "requested" (intent, not counted); "started" (written
    immediately before a protected panel's decode begins — exposure is spent
    the moment audio is decoded, whether or not scoring completes); "decoded"
    (fact, written after decode+score, resolves the started record of the same
    run_id); "correction" (validated by _active_corrections; the ONLY thing
    that can lower a count, and only for the panels it says were not decoded);
    legacy records with no "kind" (the pre-8w.19 format, one line per
    invocation naming every requested panel) count as decoded for every panel
    they name unless an active correction says otherwise.

    A "started" with no matching "decoded" is UNRESOLVED: it counts as spent
    AND (with_unresolved=True) is reported so the gate can refuse until a
    correction reconciles it.
    """
    recs = read_protected_records()
    active = _active_corrections(recs)
    counts = {n: 0 for n in names}
    unresolved = set()
    decoded_runs = {(r.get("run_id"), r.get("panel")) for r in recs if r.get("kind") == "decoded"}
    for r in recs:
        kind = r.get("kind")
        corr = active.get(r["_line_sha256"])
        if kind == "decoded":
            if r.get("panel") in counts:
                counts[r["panel"]] += 1
        elif kind == "started":
            n = r.get("panel")
            if (r.get("run_id"), n) in decoded_runs:
                continue   # resolved by its own decoded record
            if corr is not None:
                if n in corr.get("decoded", []) and n in counts:
                    counts[n] += 1
                continue   # reconciled either way
            if n in counts:
                counts[n] += 1
                unresolved.add(n)
        elif kind is None:
            excluded = set(corr.get("not_decoded", [])) if corr else set()
            for n in r.get("panels", []):
                if n in counts and n not in excluded:
                    counts[n] += 1
    return (counts, unresolved) if with_unresolved else counts


def log_protected_started(name, meta, ckpt_sha256, run_id, argv=None):
    """Written immediately BEFORE a protected panel's audio is decoded. From
    this line on the exposure is spent; the matching 'decoded' record (same
    run_id) resolves it. Council AB: a crash between decode and the fact record
    must not read as 'never decoded'."""
    import time as _time
    if name not in PROTECTED_ONE_SHOT:
        return None
    rec = {"kind": "started", "run_id": run_id,
           "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
           "panel": name, "pid": os.getpid(), "ckpt_sha256": ckpt_sha256,
           "rows_used": meta.get("rows_used"), "manifest_sha256": meta.get("manifest_sha256"),
           "argv": list(argv or [])}
    _append_protected_record(rec)
    return rec


def log_protected_decoded(name, res, meta, ckpt_sha256, argv=None, run_id=None):
    """The fact record: written by evaluate() AFTER a protected panel's decode
    and scoring complete, so the log can never claim more than happened."""
    import time as _time
    if name not in PROTECTED_ONE_SHOT:
        return None
    rec = {"kind": "decoded", "run_id": run_id,
           "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
           "panel": name, "pid": os.getpid(), "ckpt_sha256": ckpt_sha256,
           "rows_used": meta.get("rows_used"), "n_scored": res.get("n"),
           "wer_frozen": res.get("wer_frozen"),
           "manifest_sha256": meta.get("manifest_sha256"),
           "argv": list(argv or [])}
    _append_protected_record(rec)
    return rec

# The sealed FLEURS test directory. Nothing in this tool may resolve under it
# (SPEC §2: decoded once, by bnasr/final_test.py, and by nothing else).
SEALED_TEST_DIR = Path(os.environ.get("BNASR_FLEURS_DIR", "/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn")).resolve()

# Style metadata for panel D. SPEC §2 makes "conversational slice of D" both a
# guard and a one-shot report, so the slice has to be RECOVERABLE, not assumed.
# It is not in val_corpus.json (which carries only source/duration/text), and it
# is NOT in data2/val_indicvoices.json either — that file is a different
# IndicVoices slice and shares zero audio paths with D. The full IndicVoices
# manifest is the join key that actually works: all 2,276 D IndicVoices rows are
# present in it with a `scenario` field (measured: 1,282 Extempore, 774
# Conversation, 220 Read).
D_STYLE_MANIFEST = REPO / "data2/indicvoices/manifest.jsonl"
CONVERSATIONAL_SCENARIOS = ("Conversation",)

BASELINE_PATH = REPO / "data/baselines/panel_baselines.json"

SURFACES = ("frozen", "nemo_internal", "p4_target_surface")


def p4_surface(text):
    """The SPEC §1 TRAINING-TARGET surface: metric_surface(NFC(t), False) with
    whitespace collapsed. Reported, never selected on — see the module docstring.

    Identical expression to bnasr/ingest_macro.py and build_corpus.py,
    which is the point: if the targets and this diverge, the diagnostic number
    stops meaning anything.
    """
    return " ".join(metric_surface(unicodedata.normalize("NFC", text), False).split())


def resolve_audio(path):
    """Manifest audio paths are repo-relative; make them absolute, and refuse
    anything under the sealed FLEURS test directory.

    The guard is here rather than at the call site because every panel decode in
    this recipe funnels through this function, and "one accidental --panels pointing
    at eval_manifest.json" is precisely how a sealed set gets opened twice.
    """
    p = Path(path)
    p = p if p.is_absolute() else (REPO / p)
    rp = p.resolve()
    if rp == SEALED_TEST_DIR or SEALED_TEST_DIR in rp.parents:
        raise SystemExit(
            f"REFUSING to read the sealed FLEURS test set: {p}. SPEC §2 allows "
            f"one decode of it, by bnasr/final_test.py only.")
    return str(p)


def load_ctc_model(ckpt, device="cuda"):
    """Restore a this recipe CTC checkpoint and force the registered decoding mode.

    ``ASRModel.restore_from`` is used rather than a hard-coded class so a `.nemo`
    written by any this recipe stage restores as whatever it actually is; the class
    is then ASSERTED to be a CTC BPE model, because silently evaluating a
    transducer here would compare two different decoders and call it a data
    result.

    The decoding strategy is set explicitly to greedy (SPEC §6: "greedy CTC
    decoding … no decoder search"). A checkpoint saved mid-run can carry any
    decoding block, and a different decoder moves WER without moving a weight.
    """
    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    ckpt = str(ckpt)
    model = ASRModel.restore_from(ckpt, map_location="cpu")
    assert isinstance(model, EncDecCTCModelBPE), (
        f"{ckpt} restored as {type(model).__name__}; this recipe is a "
        f"FastConformer-CTC lineage (SPEC §0/§4) and this tool scores CTC only")
    try:
        from omegaconf import open_dict
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.strategy = "greedy_batch"
        model.change_decoding_strategy(model.cfg.decoding)
    except Exception as exc:  # noqa: BLE001
        # Loud, not silent: a checkpoint whose decoding block cannot be forced
        # to greedy is still evaluable, but the caller has to know the number
        # was produced under the archive's own decoder.
        print(f"WARNING: could not force greedy_batch decoding ({exc}); using "
              f"the archive's own decoding config", file=sys.stderr)
    model = model.to(device)
    model.eval()
    return model


def decode_panel(model, rows, batch_size=32):
    """Greedy batched decode of one panel, hypotheses in manifest row order.

    Row order is load-bearing: hypotheses are zipped back onto rows positionally
    and the locked D panel's order is part of what makes the P1/P2/P4 numbers
    comparable. ``num_workers=0`` because the box is shared and four ingestion
    jobs are running; a decode that forks 8 loader processes per call is how a
    co-tenant's build slows down for reasons nobody can attribute.
    """
    import torch
    paths = [resolve_audio(r["audio_filepath"]) for r in rows]
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.transcribe(paths, batch_size=batch_size, num_workers=0,
                               verbose=False)
    wall = time.perf_counter() - t0
    if isinstance(out, tuple):          # some configs return (best, all_beams)
        out = out[0]
    hyps = [(h.text if hasattr(h, "text") else str(h)) for h in out]
    assert len(hyps) == len(rows), f"{len(hyps)} hypotheses for {len(rows)} rows"
    return hyps, wall


def _corpus(pairs, counter):
    """Corpus-level rate from summed counts: Σerrors / Σreference units.

    Summing counts and dividing once is not the same as averaging per-utterance
    rates, and WER is a ratio of totals. Averaging rates would reweight a 3-word
    row like a 30-word row, which on panel R (read speech, tight length
    distribution) is a small lie and on panel D (16 s IndicVoices extempore next
    to 3 s Common Voice) is a large one.
    """
    e = n = 0
    for ref, hyp in pairs:
        a, b = counter(ref, hyp)
        e += a
        n += b
    return e, n


def d_style_map():
    """audio_filepath -> IndicVoices scenario, for the conversational slice of D.

    Returns {} (and the slice is reported as unavailable rather than as 0 rows)
    if the manifest is missing: an absent guard must be visible, never pass by
    default.
    """
    if not D_STYLE_MANIFEST.exists():
        return {}
    return {r["audio_filepath"]: r.get("scenario", "?")
            for r in load_manifest(D_STYLE_MANIFEST)}


def score_panel(rows, hyps, style_map=None):
    """All scored quantities for one panel, on all three surfaces.

    Everything a guard or a selection score needs comes out of this one function
    so that a per-source number and the panel total can never be computed from
    different hypotheses.
    """
    refs = [r["text"] for r in rows]
    surf = {
        "frozen": (frozen_norm, lambda t: frozen_norm(t)),
        "p4_target_surface": (p4_surface, p4_surface),
    }
    out = {}
    for name, (fr, fh) in surf.items():
        pairs = [(fr(r), fh(h)) for r, h in zip(refs, hyps)]
        we, wn = _corpus(pairs, wer_counts)
        ce, cn = _corpus(pairs, cer_counts)
        out[f"wer_{name}"] = round(we / max(wn, 1), 5)
        out[f"cer_{name}"] = round(ce / max(cn, 1), 5)
        out[f"ref_words_{name}"] = wn
        out[f"word_errors_{name}"] = we
    # NeMo's own WER, on RAW text, no normalisation — the baseline val_wer parity.
    out["wer_nemo_internal"] = round(nemo_wer(refs, hyps), 5)

    # Per-source, on the frozen surface only: S and the guards are defined there.
    per_src = defaultdict(lambda: [0, 0])
    for row, ref, hyp in zip(rows, refs, hyps):
        e, n = wer_counts(frozen_norm(ref), frozen_norm(hyp))
        acc = per_src[row.get("source", "?")]
        acc[0] += e
        acc[1] += n
    out["per_source"] = {s: round(a[0] / max(a[1], 1), 5)
                         for s, a in sorted(per_src.items())}
    out["per_source_ref_words"] = {s: a[1] for s, a in sorted(per_src.items())}

    empties = sum(1 for h in hyps if not h.strip())
    out["empty_hyps"] = empties
    out["empty_rate"] = round(empties / max(len(hyps), 1), 6)
    out["n"] = len(rows)

    if style_map:
        slices = defaultdict(lambda: [0, 0, 0])
        for row, ref, hyp in zip(rows, refs, hyps):
            st = style_map.get(row["audio_filepath"])
            if st is None:
                continue
            e, n = wer_counts(frozen_norm(ref), frozen_norm(hyp))
            acc = slices[st]
            acc[0] += e
            acc[1] += n
            acc[2] += 1
        out["style_slices"] = {
            st: {"wer_frozen": round(a[0] / max(a[1], 1), 5), "n": a[2],
                 "ref_words": a[1]}
            for st, a in sorted(slices.items())}
        conv = [a for st, a in slices.items() if st in CONVERSATIONAL_SCENARIOS]
        if conv:
            e = sum(a[0] for a in conv)
            n = sum(a[1] for a in conv)
            out["conversational"] = {
                "scenarios": list(CONVERSATIONAL_SCENARIOS),
                "wer_frozen": round(e / max(n, 1), 5),
                "n": sum(a[2] for a in conv), "ref_words": n}
    return out


def selection_score(panel_wers):
    """SPEC §2 selection score S, in percentage points, plus which formula ran.

    The formula is chosen by the PRESENCE OF R, as SPEC §2 words it ("if MADASR
    is admitted"). A missing panel is a refusal, not a renormalisation: quietly
    rescaling the remaining weights would produce a number called "S" that is
    not S, and every gate in SPEC §5 is stated against S.
    """
    weights = S_WEIGHTS_WITH_R if "R" in panel_wers else S_WEIGHTS_NO_R
    missing = [p for p in weights if p not in panel_wers]
    if missing:
        return {"S_pp": None, "formula": weights, "missing_panels": missing,
                "note": "S undefined: SPEC §2 weights are fixed and are never "
                        "renormalised over the panels that happen to exist"}
    s = sum(w * panel_wers[p] * 100.0 for p, w in weights.items())
    return {"S_pp": round(s, 4), "formula": weights,
            "madasr_admitted": "R" in weights,
            "components_pp": {p: round(panel_wers[p] * 100.0, 4) for p in weights}}


def check_guards(results, baseline):
    """SPEC §2 regression guards against a stored the baseline.

    Returns one record per guard with the observed delta, so a FAIL is
    actionable and a PASS is auditable. A guard whose baseline is absent is
    reported as ``"unavailable"`` and makes ``all_pass`` None — never True.
    """
    out = {"guards": [], "all_pass": True}

    def add(name, ok, **kw):
        if ok is None:
            out["all_pass"] = None if out["all_pass"] is not False else False
        elif not ok:
            out["all_pass"] = False
        out["guards"].append(dict(name=name, pass_=ok, **kw))

    for pname, res in results.items():
        add(f"empty_rate[{pname}]", res["empty_rate"] <= GUARDS["max_empty_rate"],
            observed=res["empty_rate"], limit=GUARDS["max_empty_rate"],
            spec="§2 empty-output rate ≤0.1 %")

    base_d = (baseline or {}).get("panels", {}).get("D")
    res_d = results.get("D")
    if res_d is not None:
        if base_d is None:
            add("D_per_source_regression", None, spec="§2 no D source worse than P1 by >1.0 pp",
                note="no stored the baseline D baseline; run --record-baseline first")
            add("D_conversational_regression", None,
                spec="§2 conversational slice of D worse by ≤0.5 pp",
                note="no stored the baseline D baseline")
        else:
            for src, wer in res_d["per_source"].items():
                b = base_d.get("per_source", {}).get(src)
                if b is None:
                    add(f"D_source[{src}]", None, note="source absent from baseline")
                    continue
                delta = (wer - b) * 100.0
                add(f"D_source[{src}]",
                    delta <= GUARDS["per_source_max_regression_pp"],
                    observed_pp=round(wer * 100, 4), baseline_pp=round(b * 100, 4),
                    delta_pp=round(delta, 4),
                    limit_pp=GUARDS["per_source_max_regression_pp"],
                    spec="§2 no D source worse than P1 by >1.0 pp")
            bc = base_d.get("conversational", {}).get("wer_frozen")
            rc = res_d.get("conversational", {}).get("wer_frozen")
            if bc is None or rc is None:
                add("D_conversational_regression", None,
                    note="conversational slice unavailable on one side "
                         f"(baseline={bc}, observed={rc})",
                    spec="§2 conversational slice of D worse by ≤0.5 pp")
            else:
                delta = (rc - bc) * 100.0
                add("D_conversational_regression",
                    delta <= GUARDS["conversational_max_regression_pp"],
                    observed_pp=round(rc * 100, 4), baseline_pp=round(bc * 100, 4),
                    delta_pp=round(delta, 4),
                    limit_pp=GUARDS["conversational_max_regression_pp"],
                    spec="§2 conversational slice of D worse by ≤0.5 pp")
    return out


def scorer_provenance():
    """SHA-256 of every file that can change a reported number.

    A WER without the hash of the code that computed it is an assertion, not a
    measurement; SPEC §6 requires the headline to carry it and there is no
    reason for the panel numbers to be held to a lower standard.
    """
    return {p.name: {"path": str(p), "sha256": sha256_file(p)} for p in [
        REPO / "bnasr/scoring.py",
        REPO / "bnasr/normalize.py",
        Path(__file__).resolve(),
    ]}


def manifest_collisions(rows):
    """{audio_filepath: [row indices]} for every path that appears more than once.

    Vaani validation (10,105 rows) carries three build-time filename collisions:
    two source utterances with DIFFERENT durations and DIFFERENT texts landed on
    the same output wav name. The wav on disk is one of them; the other row's
    reference cannot match its audio. That is a manifest defect to be recorded
    and scored around, not a reason to abort after the GPU work is done.
    """
    seen = defaultdict(list)
    for i, r in enumerate(rows):
        seen[r["audio_filepath"]].append(i)
    return {k: v for k, v in seen.items() if len(v) > 1}


def preflight_panels(panel_names, manifest_overrides=None, limit=None):
    """Everything that can be known about a panel BEFORE any GPU work.

    Runs before the model is loaded and before the audit 'requested' record is
    written, so a manifest problem costs no exposure and leaves no false record.
    Returns {name: {"manifest": Path, "rows": [...], "collisions": {...}}}.
    A missing manifest is fatal for a protected one-shot panel (a SKIP after
    the audit record would be a false exposure line) and a SKIP otherwise.
    """
    manifest_overrides = manifest_overrides or {}
    out = {}
    for name in panel_names:
        spec = PANELS[name]
        mpath = Path(manifest_overrides.get(name, spec["manifest"]))
        if not mpath.exists():
            if name in PROTECTED_ONE_SHOT:
                raise SystemExit(f"ABORT: one-shot panel {name} manifest {mpath} does "
                                 f"not exist. Nothing was decoded.")
            print(f"SKIP panel {name}: {mpath} does not exist yet", file=sys.stderr)
            continue
        rows = load_manifest(mpath)
        if name in PROTECTED_ONE_SHOT and not manifest_overrides.get(name):
            # Council AB: a protected panel's identity is not a warning. The
            # registered row count and the frozen manifest hash must both hold.
            got = sha256_file(mpath); want = FROZEN_ONE_SHOT_MANIFEST_SHA256.get(name)
            if want is None or got != want or len(rows) != spec["rows"]:
                raise SystemExit(f"ABORT: one-shot panel {name} is not the registered panel: "
                                 f"rows {len(rows)} (registered {spec['rows']}), manifest sha "
                                 f"{got[:12]}… (frozen {str(want)[:12]}…). Nothing was decoded.")
        elif spec["rows"] is not None and limit is None and len(rows) != spec["rows"]:
            # Not fatal — a panel can legitimately be rebuilt — but a locked
            # panel whose row count moved is not the panel the baselines were
            # measured on, and that has to be shouted about.
            print(f"WARNING: panel {name} has {len(rows)} rows, SPEC §2 "
                  f"registers {spec['rows']}", file=sys.stderr)
        if limit:
            rows = rows[:limit]
        coll = manifest_collisions(rows)
        if coll:
            print(f"WARNING: panel {name}: {len(coll)} audio_filepath value(s) appear "
                  f"more than once ({sum(len(v) for v in coll.values())} rows). Scored "
                  f"on all rows AND with the colliding rows excluded; hyps keyed by "
                  f"path keep the first occurrence and list the rest under "
                  f"'collisions'.", file=sys.stderr)
        out[name] = {"manifest": mpath, "rows": rows, "collisions": coll}
    return out


def evaluate(ckpt, panel_names, batch_size=32, limit=None, device="cuda",
             manifest_overrides=None, model=None, preflight=None, on_panel=None,
             on_decoded=None, run_id=None):
    """Decode and score every requested panel. Returns the full report dict.

    ``on_panel(name, res, meta, hyps_by_path)`` is called after EACH panel is
    scored so the caller can persist it: the 2026-09-15 incident lost two
    decoded one-shot panels because nothing was written until the end.
    """
    manifest_overrides = manifest_overrides or {}
    style = d_style_map()
    if preflight is None:
        preflight = preflight_panels(panel_names, manifest_overrides, limit)
    if model is None:
        model = load_ctc_model(ckpt, device=device)
    ckpt_sha = sha256_file(ckpt)

    results, hyps_out, panel_meta = {}, {}, {}
    for name in panel_names:
        if name not in preflight:
            continue   # SKIPped in preflight (non-protected, manifest absent)
        spec = PANELS[name]
        mpath = preflight[name]["manifest"]
        rows = preflight[name]["rows"]
        coll = preflight[name]["collisions"]
        meta0 = {"manifest": str(mpath), "manifest_sha256": sha256_file(mpath),
                 "rows_registered": spec["rows"], "rows_used": len(rows), "role": spec["role"]}
        log_protected_started(name, meta0, ckpt_sha, run_id, argv=sys.argv[1:])
        hyps, wall = decode_panel(model, rows, batch_size=batch_size)
        if on_decoded is not None:
            # Row-indexed hypotheses on disk BEFORE scoring: the exposure is
            # already spent, so from here nothing may be able to lose them.
            on_decoded(name, [{"i": i, "audio_filepath": r["audio_filepath"], "hyp": h}
                              for i, (r, h) in enumerate(zip(rows, hyps))])
        res = score_panel(rows, hyps, style_map=style if name == "D" else None)
        res["wall_s"] = round(wall, 1)
        res["batch_size"] = batch_size
        if coll:
            bad = {i for idx in coll.values() for i in idx}
            keep = [i for i in range(len(rows)) if i not in bad]
            res_x = score_panel([rows[i] for i in keep], [hyps[i] for i in keep],
                                style_map=style if name == "D" else None)
            res["excluding_collisions"] = {
                "n": res_x["n"], "rows_excluded": len(bad),
                "wer_frozen": res_x["wer_frozen"],
                "wer_nemo_internal": res_x["wer_nemo_internal"],
                "wer_p4_target_surface": res_x["wer_p4_target_surface"]}
        # Shared-box accounting: the co-tenant and other this recipe agents test on
        # this card concurrently, so every GPU action in this phase has to be
        # able to say what it actually cost. Reserved (not allocated) is the
        # number that matters — it is what the allocator holds and therefore
        # what another process cannot have. Reset per panel so a long panel's
        # peak is not attributed to the short one that follows it.
        if device.startswith("cuda"):
            import torch
            res["peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved() / (1 << 30), 3)
            torch.cuda.reset_peak_memory_stats()
        results[name] = res
        panel_meta[name] = dict(meta0, collisions={k: {"rows": v, "hyps": [hyps[i] for i in v]}
                                                   for k, v in coll.items()})
        # Keyed by path for bnasr/paired_bootstrap.py (aligns BY KEY).
        # A colliding path keeps its first hypothesis here; every hypothesis
        # for it is in panel_meta[name]["collisions"], so nothing is dropped.
        first = {}
        for r, h in zip(rows, hyps):
            first.setdefault(r["audio_filepath"], h)
        hyps_out[name] = first
        assert len(hyps_out[name]) + sum(len(v) - 1 for v in coll.values()) == len(rows)
        line = (f"{name:8s} n={res['n']:5d}  frozen {res['wer_frozen']*100:6.2f}%  "
                f"nemo {res['wer_nemo_internal']*100:6.2f}%  "
                f"p4surf {res['wer_p4_target_surface']*100:6.2f}%  "
                f"empty {res['empty_hyps']}  {res['wall_s']}s")
        if coll:
            line += (f"  [excl. {len(bad)} colliding rows: frozen "
                     f"{res['excluding_collisions']['wer_frozen']*100:6.2f}%]")
        print(line, flush=True)
        log_protected_decoded(name, res, panel_meta[name], ckpt_sha, argv=sys.argv[1:], run_id=run_id)
        if on_panel is not None:
            on_panel(name, res, panel_meta[name], hyps_out[name])

    report = {
        "tool": "bnasr/panel_eval.py",
        "spec": "experiments/SPEC.md §2 (panels, S, guards), §4 (restoration)",
        "ckpt": str(Path(ckpt).resolve()),
        "ckpt_sha256": ckpt_sha,
        "device": device,
        "surfaces": {
            "frozen": "tools/scoring.frozen_norm == bench_assets/bench_score.norm "
                      "(SELECTION surface, SPEC §2)",
            "nemo_internal": "nemo word_error_rate on RAW text (the baseline val_wer parity)",
            "p4_target_surface": "metric_surface(NFC(t), False) + whitespace collapse "
                                 "(SPEC §1 TRAINING-TARGET surface; diagnostic only)",
        },
        "panels": panel_meta,
        "results": results,
        "selection_score": selection_score(
            {k: v["wer_frozen"] for k, v in results.items() if k in ("D", "M", "R", "K")}),
        "dev_D_published_baselines": DEV_D_PUBLISHED,
        "scorer": scorer_provenance(),
        "limit": limit,
    }
    if "D" in results:
        d = results["D"]["wer_frozen"]
        report["restoration_check"] = {
            "spec": "§4 zero-update restoration: dev D frozen WER == 20.35 % ±0.05 pp",
            "observed_pp": round(d * 100, 4),
            "target_pp": round(DEV_D_PUBLISHED["phase1_ctc"] * 100, 4),
            "delta_pp": round((d - DEV_D_PUBLISHED["phase1_ctc"]) * 100, 4),
            "tolerance_pp": RESTORATION_TOLERANCE_PP,
            "passes": abs(d - DEV_D_PUBLISHED["phase1_ctc"]) * 100
            <= RESTORATION_TOLERANCE_PP,
            "nemo_internal_observed_pp": round(results["D"]["wer_nemo_internal"] * 100, 4),
            "nemo_internal_target_pp": round(P1_DEV_NEMO_INTERNAL * 100, 4),
        }
    return report, hyps_out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--panels", nargs="+", default=["D"], choices=list(PANELS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows of each panel — a shape check, never a gate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-hyps", default=None,
                    help="write {panel: {audio_filepath: hypothesis}} — the input "
                         "bnasr/paired_bootstrap.py aligns BY KEY")
    ap.add_argument("--manifest", action="append", default=[],
                    metavar="PANEL=PATH",
                    help="override a panel manifest (e.g. M=experiments/corpus/panel_M.jsonl)")
    ap.add_argument("--baseline", default=str(BASELINE_PATH),
                    help="stored the baseline the SPEC §2 guards compare against")
    ap.add_argument("--record-baseline", action="store_true",
                    help="write this evaluation as the the baseline guard baseline "
                         "(refuses to overwrite without --force)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        pin_gpu()

    overrides = {}
    for kv in a.manifest:
        k, _, v = kv.partition("=")
        if k not in PANELS:
            raise SystemExit(f"--manifest {kv}: unknown panel {k!r}")
        overrides[k] = v

    # Council R (SPEC §8w.11): refuse a PROTECTED one-shot panel - by name or by
    # an overridden manifest that points at one - unless the phase-level held
    # decode has been authorised. This runs BEFORE the model is loaded, so a
    # refusal costs no GPU and decodes nothing.
    # End-of-run review (2026-09-15): (a) `--panels holdout holdout` decoded a
    # one-shot twice in one invocation with a single log line; (b) nothing bound
    # --ckpt to the QUALIFIED selection, so the gate could pass on the event and
    # the panels be decoded with some other archive. Both closed here.
    if len(set(a.panels)) != len(a.panels):
        raise SystemExit(f"ABORT: --panels lists a panel more than once: {a.panels}")
    if _protected_selected(a.panels, list(overrides.values())) \
            and os.environ.get(PROTECTED_OVERRIDE_ENV) != "1":
        import json as _json
        if not QUALIFIED_SELECTION.exists():
            raise SystemExit(f"ABORT: one-shot panels require {QUALIFIED_SELECTION}; none exists")
        q = _json.loads(QUALIFIED_SELECTION.read_text(encoding="utf-8"))
        want = q.get("selected_sha256"); got = sha256_file(Path(a.ckpt))
        if want != got:
            raise SystemExit(f"ABORT: --ckpt sha {got[:12]}… is not the qualified selection "
                             f"{str(want)[:12]}… ({q.get('selected_checkpoint')}). One-shot panels are "
                             f"decoded with the SELECTED checkpoint only.")
    # SPEC 8w.19: every manifest is read and checked BEFORE the audit record
    # and before the model is loaded, so a manifest defect is found at zero
    # cost; and every panel is persisted the moment it is scored.
    import uuid
    run_id = uuid.uuid4().hex
    protected = _protected_selected(a.panels, list(overrides.values()))
    if protected and os.environ.get(PROTECTED_OVERRIDE_ENV) != "1":
        # Council AB: a protected decode without a durable destination is a
        # decode whose result can be lost. Both outputs are required, both
        # directories must be writable, and neither file may already exist.
        if not (a.out and a.save_hyps):
            raise SystemExit("ABORT: one-shot panels require --out AND --save-hyps. Nothing was decoded.")
        for pth in (Path(a.out), Path(a.save_hyps)):
            if pth.exists():
                raise SystemExit(f"ABORT: {pth} exists; a one-shot result is never overwritten. Nothing was decoded.")
            pth.parent.mkdir(parents=True, exist_ok=True)
            probe = pth.parent / f".probe_{run_id}"
            try:
                probe.write_text("x"); probe.unlink()
            except OSError as exc:
                raise SystemExit(f"ABORT: cannot write in {pth.parent}: {exc}. Nothing was decoded.")
    preflight = preflight_panels(a.panels, overrides, a.limit)
    authorise_protected_panels(a.panels, list(overrides.values()), argv=sys.argv[1:],
                               ckpt_sha256=sha256_file(Path(a.ckpt)), run_id=run_id)

    partial = {"results": {}, "panels": {}, "hyps": {}, "raw_hyps": {}}
    partial_out = Path(a.out).with_suffix(".partial.json") if a.out else None

    def _atomic_write(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False)
            fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def _write_partial():
        if partial_out is not None:
            partial_out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(partial_out, {"tool": "bnasr/panel_eval.py", "run_id": run_id,
                                        "status": "partial", "ckpt": str(Path(a.ckpt).resolve()),
                                        "ckpt_sha256": sha256_file(Path(a.ckpt)),
                                        "panels": partial["panels"],
                                        "results": partial["results"],
                                        "hyps": partial["hyps"],
                                        "raw_hyps": partial["raw_hyps"]})

    def on_decoded(name, rows_hyps):
        partial["raw_hyps"][name] = rows_hyps
        _write_partial()

    def on_panel(name, res, meta, hyps_by_path):
        partial["results"][name] = res
        partial["panels"][name] = meta
        partial["hyps"][name] = hyps_by_path
        partial["raw_hyps"].pop(name, None)   # now carried by hyps + collisions
        _write_partial()

    report, hyps = evaluate(a.ckpt, a.panels, batch_size=a.batch_size,
                            limit=a.limit, device=a.device,
                            manifest_overrides=overrides, preflight=preflight,
                            on_panel=on_panel, on_decoded=on_decoded, run_id=run_id)
    report["run_id"] = run_id

    baseline = None
    bpath = Path(a.baseline)
    if bpath.exists():
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        report["baseline_source"] = {"path": str(bpath),
                                     "sha256": sha256_file(bpath),
                                     "ckpt_sha256": baseline.get("ckpt_sha256")}
    report["guard_report"] = check_guards(report["results"], baseline)

    if a.record_baseline:
        if a.limit:
            raise SystemExit("--record-baseline with --limit would freeze a "
                             "truncated panel as the guard reference")
        if bpath.exists() and not a.force:
            raise SystemExit(f"{bpath} exists; the guards are measured against it. "
                             f"Pass --force only if you mean to move the reference.")
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps(
            {"tool": report["tool"], "ckpt": report["ckpt"],
             "ckpt_sha256": report["ckpt_sha256"], "panels": report["results"],
             "panel_meta": report["panels"], "scorer": report["scorer"],
             "spec": "SPEC §2 regression-guard reference (the baseline final.nemo)"},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"baseline written: {bpath}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(Path(a.out), report)
    if a.save_hyps:
        Path(a.save_hyps).parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(Path(a.save_hyps),
                      {"ckpt_sha256": report["ckpt_sha256"], "panels": hyps,
                       "collisions": {n: m.get("collisions", {})
                                      for n, m in report["panels"].items()}})
    if partial_out is not None and partial_out.exists() and a.out and Path(a.out).exists():
        partial_out.unlink()   # the complete report supersedes the crash-safety copy
    print(json.dumps({k: v for k, v in report.items() if k != "results"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
