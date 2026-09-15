#!/usr/bin/env python
"""Workstream-2 harness: does the main run actually stop when SPEC says stop?

    .venv/bin/python experiments/p4/tests/test_main_run_gate.py

CPU only. No GPU is opened, no model is loaded, no panel is decoded: every
check here is about the REGISTERED DECISION LOGIC, which is pure arithmetic
over a validation record plus some file bookkeeping. That is the part a week
of GPU time rests on - SPEC §8k Q8 prices the 20,000-update gate at "~15-19
GPU-h to reach the decision" against "the failure it prevents ~157-210 GPU-h".

What "verified by execution" means here
--------------------------------------
Every check calls the shipped function. Nothing asserts about source text.
Where a rule ends the process - and most registered stops do, via
``hard_exit`` - the call is made in a forked CHILD, and the parent checks the
child's exit status and the record the child left on disk. Exit code 4 is a
REGISTERED STOP and exit code 3 is an ABORT (p4_train.STOP_EXIT / ABORT_EXIT),
and run_supervised.sh treats those two outcomes completely differently, so a
test that only checked "it stopped" would not be checking the thing that
decides whether 150,000 updates get resurrected.

The supervisor's own terminal-semantics block is not re-implemented either: it
is EXTRACTED from experiments/p4/run_supervised.sh at run time (that file
belongs to another workstream and is read, never edited) and executed against
STOP/ABORT records that the real ``stop()`` and ``abort()`` wrote.

Coverage map (letters are the workstream-2 brief):
  (a) gate20k_*        - fires at exactly 20,000, all three conditions, STOP
  (b) comparator_*     - continuation guards use the FROZEN P1 baseline
  (c) eligibility_*    - a guard failure is ineligibility, not a reset
  (d) plateau_*        - anchor 50k / 0.20 pp cumulative / 10 / earliest 70k
  (e) retention_*      - the selection survives rotation; ties go earlier
  (f) resume_*         - horizon, scheduler, optimizer, sampler, RNG
  (g) supervisor_*     - terminal STOP vs graceful_stop vs infrastructure ABORT
"""
import importlib.machinery
import importlib.util
import json
import os
import signal
import uuid
import subprocess
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import p4_fixtures as F                                   # noqa: E402
from tools.p4 import p4_common as P4                      # noqa: E402
from tools.p4 import p4_train as T                        # noqa: E402

RESULTS = []
CFG = F.cfg()


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def check(name, passed, evidence):
    RESULTS.append({"name": name, "passed": bool(passed), "evidence": evidence})
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {evidence}",
          flush=True)
    return bool(passed)


def in_child(fn, *a, **kw):
    """Run `fn` in a forked child and return (exit_code, captured_output).

    The registered stops call ``hard_exit`` (``os._exit``) on purpose - SPEC
    §4's watchdog exists because a trainer that merely raises leaves a zombie
    holding several GiB of a SHARED card. So the only way to test them is to
    let a process die and look at the corpse: the exit status and the record.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # child
        try:
            os.close(r)
            os.dup2(w, 1)
            os.dup2(w, 2)
            os.close(w)
            # dup2 moves the FILE DESCRIPTORS, but `print` writes to
            # `sys.stdout`, which pytest has replaced with an in-memory capture
            # object that never touches fd 1. Without this rebind the child's
            # output landed in the PARENT's capture buffer, the pipe returned
            # empty, and every caller that parses the last line raised
            # IndexError - the test failed for a reason unrelated to the
            # behaviour under test.
            sys.stdout = os.fdopen(1, "w", closefd=False)
            sys.stderr = os.fdopen(2, "w", closefd=False)
            fn(*a, **kw)
            sys.stdout.flush()
            os._exit(0)
        except BaseException:                       # noqa: BLE001
            traceback.print_exc()
            # print_exc writes to the rebound sys.stderr, which on a pipe is
            # BLOCK-buffered; os._exit skips every atexit flush, so without this
            # the traceback text never reached the parent and
            # stop_rejects_an_unknown_ending_class failed on an empty capture.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(99)
    os.close(w)
    with os.fdopen(r, "rb") as fh:
        out = fh.read().decode("utf-8", "replace")
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out


def fresh(tag):
    d = Path(tempfile.mkdtemp(prefix=f"p4t_{tag}_"))
    return F.run_dir(d)


def stop_record(run):
    p = Path(run) / "STOP"
    return json.loads(p.read_text().splitlines()[0]) if p.exists() else None


def new_sel():
    return {"best_eligible_s": None, "anchor": None, "nonreset": 0}


def call_stops(val, update, run, sel=None):
    """The shipped check_registered_stops, in this process (no stop expected)."""
    return T.check_registered_stops(val, update, None, run, sel or new_sel())


# --------------------------------------------------------------------------
# (a) the 20,000-update progress gate
# --------------------------------------------------------------------------
def test_gate20k():
    # SPEC §8k Q8: "S <= 19.530 AND D <= 19.35 AND all main-run guards".
    check("gate20k_thresholds_are_council_F's",
          abs((F.BASE["S_pp"] - P4.GATE20K_S_PP) - P4.GATE20K_S_ABS_PP) <= P4.GATE20K_ABS_TOL_PP
          and abs((P4.P1_DEV_FROZEN * 100 - P4.GATE20K_D_PP) - P4.GATE20K_D_ABS_PP) < 1e-9,
          f"S(P1)={F.BASE['S_pp']} - {P4.GATE20K_S_PP} = "
          f"{round(F.BASE['S_pp'] - P4.GATE20K_S_PP, 4)} vs ruling "
          f"{P4.GATE20K_S_ABS_PP}; D(P1)=20.35 - 1.0 = 19.35 vs ruling "
          f"{P4.GATE20K_D_ABS_PP}")

    # PASS: S 19.40 (<= 19.5296), D 19.30 % (<= 19.35), guards clean.
    run = fresh("gate_pass")
    v = F.val_with_S(20000, 19.40, D=0.1930)
    code, out = in_child(call_stops, v, 20000, run)
    check("gate20k_passes_when_all_three_conditions_hold",
          code == 0 and "GATE20K PASS" in out and not (run / "STOP").exists(),
          f"S={v['S_pp']} D=19.30% guards={v['guards']['pass']} -> exit {code}, "
          f"STOP file {(run / 'STOP').exists()}")

    # FAIL on S alone: 19.54 > 19.5296, D and guards fine.
    run = fresh("gate_S")
    v = F.val_with_S(20000, 19.54, D=0.1930)
    code, _ = in_child(call_stops, v, 20000, run)
    rec = stop_record(run)
    check("gate20k_fails_on_S_alone",
          code == T.STOP_EXIT and rec and "progress gate failed" in rec["stop"],
          f"S={v['S_pp']} > limit {round(F.BASE['S_pp'] - 1.5, 4)}; exit {code} "
          f"(STOP_EXIT={T.STOP_EXIT}); record={rec and rec['stop']}")

    # FAIL on D alone: S 19.40 but D 19.40 % > 19.35 %.
    run = fresh("gate_D")
    v = F.val_with_S(20000, 19.40, D=0.1940)
    code, _ = in_child(call_stops, v, 20000, run)
    rec = stop_record(run)
    check("gate20k_fails_on_D_alone",
          code == T.STOP_EXIT and rec and rec["D_pp"] == 19.40,
          f"D=19.40% > 19.35%; exit {code}; record D_pp={rec and rec['D_pp']} "
          f"limit {rec and rec['D_limit_pp']}")

    # FAIL on the guards alone: S and D pass, conversational 22.10 > 22.075.
    run = fresh("gate_G")
    v = F.val_with_S(20000, 19.40, D=0.1930, conversational=0.2210)
    code, _ = in_child(call_stops, v, 20000, run)
    rec = stop_record(run)
    check("gate20k_fails_on_guards_alone",
          code == T.STOP_EXIT and rec and rec["guards_pass"] is False,
          f"conversational 22.10% > limit {P4.conversational_limit_pp(F.BASE)}%;"
          f" S={v['S_pp']} D=19.30% still pass; exit {code}")

    # A gate failure is a registered STOP, never an ABORT.
    check("gate20k_failure_is_a_STOP_record_not_an_ABORT",
          (run / "STOP").exists() and not (run / "ABORT").exists()
          and T.STOP_EXIT == 4 and T.ABORT_EXIT == 3,
          f"STOP written, ABORT absent, exit codes STOP={T.STOP_EXIT} "
          f"ABORT={T.ABORT_EXIT}")

    # EXACTLY at 20,000: a catastrophic S one update either side does nothing.
    for u in (19999, 20001):
        run = fresh(f"gate_{u}")
        v = F.val_with_S(u, 25.00, D=0.2340)
        code, _ = in_child(call_stops, v, u, run)
        check(f"gate20k_does_not_fire_at_{u}",
              code == 0 and not (run / "STOP").exists(),
              f"S={v['S_pp']} (far above 19.530) at update {u} -> exit {code}")

    # The gate cannot be switched off by --val-every.
    ok = (T.should_validate(20000, 0, 150000, set())
          and T.should_validate(20000, 3000, 150000, set())
          and not T.should_validate(20000, 0, 8000, set()))
    check("gate20k_validation_is_forced_even_with_val_every_0", ok,
          "should_validate(20000, val_every=0, stop_at=150000)=True; "
          "val_every=3000 -> True; a pilot with stop_at=8000 -> False")

    # A comparator that is not the frozen P1 record stops the run.
    run = fresh("gate_prov")
    tmpbase = run / "oref.json"
    F.oref_shaped_baseline(tmpbase)

    def _wrong_comparator():
        P4.BASELINE_JSON = tmpbase          # child only
        T.check_registered_stops(F.val_with_S(20000, 19.40, D=0.1930),
                                 20000, None, run, new_sel())
    code, _ = in_child(_wrong_comparator)
    rec = stop_record(run)
    check("gate20k_refuses_an_O_ref_shaped_comparator",
          code == T.STOP_EXIT and rec and "comparator rejected" in rec["stop"],
          f"baseline without init A's sha256 -> exit {code}: "
          f"{rec and rec['stop'][:90]}")


# --------------------------------------------------------------------------
# (b) the continuation/final guards compare against the FROZEN P1 baseline
# --------------------------------------------------------------------------
def test_comparator():
    lim = P4.conversational_limit_pp(F.BASE)
    check("comparator_conversational_limit_is_22.075",
          lim == 22.075,
          f"P1 conversational {F.P1_CONV * 100:.3f}% + "
          f"{P4.GUARD_CONVERSATIONAL_PP} pp = {lim}% (SPEC §8k Q2)")

    just_under = F.make_val(60000, conversational=0.22074)
    just_over = F.make_val(60000, conversational=0.22076)
    check("comparator_conversational_guard_bites_at_the_limit",
          P4.is_eligible(just_under) and not P4.is_eligible(just_over),
          f"22.074% eligible={P4.is_eligible(just_under)}, "
          f"22.076% eligible={P4.is_eligible(just_over)}")

    # Council F Q2 states the consequence outright: MaCro's 8k conversational
    # 21.892 IS eligible under the P1 rule. Under an O_ref-based rule
    # (21.47 + 0.5 = 21.97) it would be too, but 22.00 would not - which is
    # the whole reason the comparator has to be named.
    macro = F.make_val(8000, conversational=0.21892)
    mid = F.make_val(8000, conversational=0.2200)
    check("comparator_matches_council_F_on_MaCro's_21.892",
          P4.is_eligible(macro) and P4.is_eligible(mid),
          f"21.892% eligible={P4.is_eligible(macro)} (SPEC §8k Q2 says so); "
          f"22.000% eligible={P4.is_eligible(mid)} under P1 (+0.5 = 22.075) "
          f"but would fail an O_ref rule (21.47 + 0.5 = 21.97)")

    check("comparator_source_guard_is_P1_plus_1.0pp",
          P4.is_eligible(F.make_val(60000, source_delta_pp=0.99))
          and not P4.is_eligible(F.make_val(60000, source_delta_pp=1.01)),
          "every D source at P1 +0.99 pp eligible; at P1 +1.01 pp ineligible "
          f"(limit {P4.GUARD_SOURCE_PP} pp)")

    check("comparator_empty_rate_guard_is_0.1_percent",
          P4.is_eligible(F.make_val(60000, empty_rate=0.001))
          and not P4.is_eligible(F.make_val(60000, empty_rate=0.0011)),
          f"0.100% eligible, 0.110% ineligible (limit "
          f"{P4.GUARD_EMPTY_RATE * 100}%)")

    try:
        P4.assert_frozen_p1_baseline(json.loads(
            F.oref_shaped_baseline(Path(tempfile.mkdtemp()) / "b.json")
            .read_text()))
        rejected = False
        why = "accepted an O_ref-shaped baseline"
    except ValueError as exc:
        rejected, why = True, str(exc)[:110]
    check("comparator_provenance_check_rejects_a_non_init_A_baseline",
          rejected, why)
    check("comparator_accepts_the_real_baseline_p1.json",
          P4.assert_frozen_p1_baseline(F.BASE) is F.BASE,
          f"init_sha256={F.BASE['init_sha256'][:8]}… D={F.BASE['D']['wer_frozen']} "
          f"S={F.BASE['S_pp']}")


# --------------------------------------------------------------------------
# (c) a guard failure is INELIGIBILITY, and it does not reset patience
# --------------------------------------------------------------------------
def test_eligibility():
    sel = new_sel()
    # anchor armed at 50,000 from an eligible S of 21.00
    sel = call_stops(F.val_with_S(50000, 21.00), 50000, fresh("elig0"), sel)
    armed = (sel["anchor"], sel["nonreset"])
    # a MUCH better S, but the conversational guard fails -> must not reset
    bad = F.val_with_S(52000, 19.00, conversational=0.2300)
    sel = call_stops(bad, 52000, fresh("elig1"), sel)
    check("eligibility_a_failing_guard_does_not_reset_plateau_patience",
          not P4.is_eligible(bad) and sel["anchor"] == armed[0]
          and sel["nonreset"] == 1,
          f"S fell 21.00 -> 19.00 (2.00 pp, ten times the 0.20 pp reset) but "
          f"the conversational guard failed: anchor stayed {sel['anchor']}, "
          f"nonreset {armed[1]} -> {sel['nonreset']}")
    check("eligibility_an_ineligible_S_does_not_become_best_eligible",
          sel["best_eligible_s"] == 21.00,
          f"best_eligible_s={sel['best_eligible_s']} (the ineligible 19.00 "
          f"was not adopted)")
    good = F.val_with_S(54000, 20.79)
    sel2 = call_stops(good, 54000, fresh("elig2"), dict(sel))
    check("eligibility_a_passing_guard_does_reset_it",
          sel2["anchor"] == 20.79 and sel2["nonreset"] == 0,
          f"eligible S 20.79 <= anchor 21.00 - 0.20 -> anchor "
          f"{sel2['anchor']}, nonreset {sel2['nonreset']}")

    # ...and it can never be selected, even when its S is the lowest by far.
    hist = [(21.00, 50000, True), (19.00, 52000, False)]
    check("eligibility_an_ineligible_checkpoint_is_never_selected",
          P4.select_best(hist) == (21.00, 50000),
          f"history {hist} -> select_best={P4.select_best(hist)}")
    check("eligibility_no_eligible_checkpoint_means_no_selection",
          P4.select_best([(19.0, 1000, False)]) is None,
          "select_best over ineligible-only history returns None, not a "
          "checkpoint the guards rejected")


# --------------------------------------------------------------------------
# (d) the plateau rule as council F re-ruled it
# --------------------------------------------------------------------------
def test_plateau():
    check("plateau_constants_are_council_F's",
          (P4.PLATEAU_ANCHOR_AT, P4.PLATEAU_NONRESET, P4.PLATEAU_MIN_PP,
           P4.PLATEAU_EARLIEST_STOP) == (50000, 10, 0.20, 70000),
          f"anchor {P4.PLATEAU_ANCHOR_AT}, {P4.PLATEAU_NONRESET} non-reset "
          f"validations, {P4.PLATEAU_MIN_PP} pp cumulative, earliest "
          f"{P4.PLATEAU_EARLIEST_STOP} (SPEC §4 / §8k Q4)")
    check("plateau_old_rule_is_gone",
          not hasattr(P4, "PLATEAU_FROM") and not hasattr(P4, "PLATEAU_VALIDATIONS"),
          "p4_common no longer defines PLATEAU_FROM (30,000) or "
          "PLATEAU_VALIDATIONS (5)")

    # Not armed before 50,000: 12 flat validations from 30,000 do nothing.
    run = fresh("plat_early")

    def _early():
        sel = new_sel()
        for u in range(30000, 54000, 2000):
            sel = T.check_registered_stops(F.val_with_S(u, 21.00), u, None,
                                           run, sel)
        print("SEL " + json.dumps(sel))
    code, out = in_child(_early)
    sel = json.loads(out.strip().splitlines()[-1][4:])
    check("plateau_not_armed_before_50000",
          code == 0 and sel["anchor"] == 21.00 and sel["nonreset"] == 1,
          f"11 flat validations 30,000-52,000: exit {code}, anchor armed at "
          f"50,000 to {sel['anchor']}, counter {sel['nonreset']} (the old "
          f"rule would have stopped at 40,000)")

    # The anchor makes 0.20 pp CUMULATIVE: 0.15 pp steps never reset a
    # best-so-far comparison but do cross a sticky anchor.
    run = fresh("plat_cum")

    def _cumulative():
        sel = new_sel()
        s = 21.00
        for i, u in enumerate(range(50000, 58000, 2000)):
            sel = T.check_registered_stops(F.val_with_S(u, s), u, None, run, sel)
            print(f"AT {u} S={s} anchor={sel['anchor']} n={sel['nonreset']}")
            s = round(s - 0.15, 4)
        print("SEL " + json.dumps(sel))
    code, out = in_child(_cumulative)
    sel = json.loads(out.strip().splitlines()[-1][4:])
    trail = [l for l in out.splitlines() if l.startswith("AT ")]
    check("plateau_reset_is_cumulative_against_the_anchor",
          code == 0 and sel["anchor"] == 20.70 and sel["nonreset"] == 1
          and "AT 52000 S=20.85 anchor=21.0 n=1" in trail
          and "AT 54000 S=20.7 anchor=20.7 n=0" in trail,
          "S 21.00, 20.85, 20.70, 20.55 in 0.15 pp steps; every step is "
          "smaller than the 0.20 pp reset, so a best-so-far comparison never "
          "resets on this sequence. Against the sticky anchor the CUMULATIVE "
          "21.00 -> 20.70 does: " + " | ".join(trail))

    # Exactly 10 consecutive non-reset validations, and not one earlier.
    run = fresh("plat_stop")

    def _flat():
        sel = new_sel()
        for u in range(50000, 72000, 2000):
            sel = T.check_registered_stops(F.val_with_S(u, 21.00), u, None,
                                           run, sel)
            print(f"SURVIVED {u} n={sel['nonreset']}")
    code, out = in_child(_flat)
    rec = stop_record(run)
    last = [l for l in out.splitlines() if l.startswith("SURVIVED")][-1]
    check("plateau_stops_after_10_nonreset_validations_at_70000",
          code == T.STOP_EXIT and rec and rec["update"] == 70000
          and rec["nonreset_validations"] == 10 and "68000" in last,
          f"flat S 21.00 from 50,000: last survived validation {last.split()[1]}"
          f", STOP at update {rec and rec['update']} with "
          f"{rec and rec['nonreset_validations']} non-reset validations")

    # The earliest-stop floor is on the UPDATE, not on the counter: --val-at
    # can add validations, and ten of those must not stop the run before 70k.
    run = fresh("plat_floor")

    def _dense():
        sel = new_sel()
        us = list(range(50000, 51100, 100))          # 11 dense validations
        for u in us:
            sel = T.check_registered_stops(F.val_with_S(u, 21.00), u, None,
                                           run, sel)
        print(f"SURVIVED n={sel['nonreset']} last={us[-1]}")
    code, out = in_child(_dense)
    check("plateau_never_stops_before_update_70000",
          code == 0 and "SURVIVED n=10" in out,
          f"10 non-reset validations reached by update 51,000 (--val-at "
          f"cadence): exit {code}, {out.strip().splitlines()[-1]} - no STOP, "
          f"because SPEC §4 registers the earliest stop as 70,000")


# --------------------------------------------------------------------------
# (e) the selected checkpoint survives rotation; ties resolve earlier
# --------------------------------------------------------------------------
def _rotation_scenario(retain_fn, run, cfg):
    """Best ELIGIBLE at 10,000, then three INELIGIBLE checkpoints with lower S.

    This is the shape council F Q2 (ii) is about and it is not exotic: S has
    no guard term, so a checkpoint that wins on the composite while breaking
    the conversational or a per-source guard outranks a clean one, and there
    are only three best/ slots and two ckpt/ slots.
    """
    hist = []
    plan = [(10000, 20.00, True), (20000, 19.00, False),
            (22000, 19.10, False), (24000, 19.20, False)]
    for u, s, elig in plan:
        F.write_ckpt(run, u)
        try:
            hist = retain_fn(run, u, s, cfg, hist, eligible=elig)
        except TypeError:                       # the superseded signature
            hist = retain_fn(run, u, s, cfg, hist)
    return hist


def test_retention():
    cfg = json.loads(json.dumps(CFG))
    run = fresh("ret_new")
    hist = _rotation_scenario(T.retain, run, cfg)
    survived = (run / "best" / "step_10000.pt").exists()
    check("retention_keeps_the_best_eligible_against_three_lower_S_rivals",
          survived and P4.select_best(hist) == (20.00, 10000),
          f"best/ = {sorted(p.name for p in (run / 'best').glob('*.pt'))}, "
          f"ckpt/ = {sorted(p.name for p in (run / 'ckpt').glob('*.pt'))}; "
          f"selection = {P4.select_best(hist)}")

    # The same scenario against the SUPERSEDED implementation, to show the
    # defect was real rather than hypothetical. Skipped (not passed) when the
    # pre-fix file is not available.
    orig = os.environ.get("P4_SUPERSEDED_TRAINER", "") or str(
        HERE / "superseded" / "p4_train_pre_D4_2026-09-07.py")
    if orig and Path(orig).exists():
        loader = importlib.machinery.SourceFileLoader("p4_train_superseded",
                                                      orig)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        run2 = fresh("ret_old")
        _rotation_scenario(mod.retain, run2, cfg)
        lost = not any((run2 / d / "step_10000.pt").exists()
                       for d in ("best", "ckpt", "milestones"))
        check("retention_defect_reproduced_on_the_superseded_code", lost,
              f"pre-fix retain() left best/ = "
              f"{sorted(p.name for p in (run2 / 'best').glob('*.pt'))} and "
              f"ckpt/ = {sorted(p.name for p in (run2 / 'ckpt').glob('*.pt'))}"
              f" - the best eligible checkpoint step_10000.pt is GONE")
    else:
        RESULTS.append({"name": "retention_defect_reproduced_on_the_superseded_code",
                        "passed": None, "evidence": "skipped: set "
                        "P4_SUPERSEDED_TRAINER to the pre-fix p4_train.py"})

    # Ties inside 0.05 pp go to the EARLIER checkpoint; outside it, to the
    # lower S. A plain sort by (S, update) gets the first of these wrong.
    check("retention_tie_inside_0.05pp_resolves_to_the_earlier_checkpoint",
          P4.select_best([(20.50, 60000, True), (20.47, 80000, True)])
          == (20.50, 60000),
          "S 20.50 @60,000 vs 20.47 @80,000 (0.03 pp apart) -> "
          f"{P4.select_best([(20.50, 60000, True), (20.47, 80000, True)])}; "
          "sorted() by (S, update) would return (20.47, 80000)")
    check("retention_tie_outside_0.05pp_resolves_to_the_lower_S",
          P4.select_best([(20.50, 60000, True), (20.40, 80000, True)])
          == (20.40, 80000),
          "S 20.50 @60,000 vs 20.40 @80,000 (0.10 pp apart) -> "
          f"{P4.select_best([(20.50, 60000, True), (20.40, 80000, True)])}")
    check("retention_tie_band_is_measured_from_the_best_not_chained",
          P4.select_best([(20.60, 10000, True), (20.56, 20000, True),
                          (20.52, 30000, True)]) == (20.56, 20000),
          "20.60 / 20.56 / 20.52: the band is [20.52, 20.57], so 20.60 is "
          "out and the earliest inside it wins -> "
          f"{P4.select_best([(20.60, 10000, True), (20.56, 20000, True), (20.52, 30000, True)])}")

    # A long run: the selection is still on disk after 40 rotations.
    run = fresh("ret_long")
    hist, cfg2 = [], json.loads(json.dumps(CFG))
    for i, u in enumerate(range(2000, 82000, 2000)):
        F.write_ckpt(run, u)
        s = 21.0 - min(i, 5) * 0.30 + (0.4 if i % 3 else 0.0)
        hist = T.retain(run, u, round(s, 4), cfg2, hist,
                        eligible=(i % 4 != 1))
    sel = P4.select_best(hist)
    alive = any((run / d / f"step_{sel[1]}.pt").exists()
                for d in ("best", "milestones", "ckpt"))
    check("retention_selection_survives_40_rotations", alive,
          f"40 validations, keep_last={cfg2['ckpt']['keep_last']} "
          f"keep_best={cfg2['ckpt']['keep_best']}; selection = "
          f"S {sel[0]} @{sel[1]}; on disk = {alive}; best/ = "
          f"{sorted(p.name for p in (run / 'best').glob('*.pt'))}")
    check("retention_milestones_are_never_rotated",
          (run / "milestones" / "step_20000.pt").exists()
          and (run / "milestones" / "step_40000.pt").exists()
          and (run / "milestones" / "step_8000.pt").exists(),
          f"milestones/ = "
          f"{sorted(p.name for p in (run / 'milestones').glob('*.pt'))}")


# --------------------------------------------------------------------------
# (f) resume
# --------------------------------------------------------------------------
class _FakeSampler:
    """The two things ``sampler_state_dict`` touches on a Lhotse sampler."""

    def __init__(self):
        self.constraint = object()
        self._sd = {"diagnostics": {"current_epoch": 0,
                                    "stats_per_epoch": {0: {"kept_batches": 0,
                                                            "discarded_batches": 7}}}}

    def state_dict(self):
        assert self.constraint is None, (
            "sampler_state_dict must blank the constraint before serialising")
        return self._sd


def test_resume():
    import numpy as np
    import torch
    run = fresh("resume")
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(P4.make_param_groups(model), lr=P4.lr_at(0),
                            betas=CFG["optim"]["betas"],
                            eps=float(CFG["optim"]["eps"]))
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    sampler = _FakeSampler()
    sel = {"best_eligible_s": 20.4, "anchor": 20.6, "nonreset": 3}
    hist = [(20.6, 50000, True), (20.4, 60000, True), (19.9, 62000, False)]
    torch.manual_seed(11)
    path = T.save_checkpoint(run / "ckpt" / "step_60000.pt", model, opt,
                             sampler, 60000, 2, CFG, 123456,
                             ["cut_a", "cut_b"],
                             extra={"sel": sel, "best_hist": hist})
    before = (torch.randn(4), np.random.rand(3).tolist())
    state = torch.load(path, map_location="cpu", weights_only=False)

    check("resume_checkpoint_carries_H_150000",
          state["horizon"] == P4.HORIZON == 150000,
          f"checkpoint horizon={state['horizon']}, registered H={P4.HORIZON}")
    check("resume_checkpoint_carries_optimizer_and_sampler_and_RNG",
          bool(state["optimizer"]["state"]) and "sampler" in state
          and set(state["rng"]) == {"torch", "cuda", "python", "numpy"},
          f"optimizer state entries={len(state['optimizer']['state'])}, "
          f"rng keys={sorted(state['rng'])}, "
          f"sampler cursor kept_batches="
          f"{state['sampler']['diagnostics']['stats_per_epoch'][0]['kept_batches']}")
    check("resume_sampler_cursor_is_the_loop_count_minus_the_prefetch_lead",
          state["sampler"]["diagnostics"]["stats_per_epoch"][0]["kept_batches"]
          == 123455 and state["last_micro_ids"] == ["cut_a", "cut_b"],
          "consumed=123,456 -> kept_batches=123,455 with the last microbatch's "
          "cut ids stored for the resume handshake (tools/p3_train docstring)")
    check("resume_scheduler_position_is_a_pure_function_of_update_and_H",
          abs(P4.lr_at(60000, state["horizon"]) - P4.lr_at(60000)) < 1e-18
          and abs(P4.lr_at(7999) - 0.000198967465612861) < 1e-15,
          "lr_at(60000, H from the checkpoint) == lr_at(60000); lr_at(7999) "
          "reproduces the LR recorded at update 8000 of the live runs "
          "O_A5000 and pilot_O+macro_filtered (0.000198967465612861)")

    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    after = (torch.randn(4), np.random.rand(3).tolist())
    check("resume_RNG_restore_reproduces_the_stream",
          torch.equal(before[0], after[0]) and before[1] == after[1],
          f"torch.randn(4) before={[round(float(x), 5) for x in before[0]]} "
          f"after restore={[round(float(x), 5) for x in after[0]]}")

    ex = state["extra"]
    hist2, legacy = P4.normalise_history(ex["best_hist"])
    check("resume_restores_the_selection_state",
          ex["sel"] == sel and hist2 == hist and legacy == 0
          and P4.select_best(hist2) == (20.4, 60000),
          f"sel round-tripped {ex['sel']}; history {len(hist2)} entries, "
          f"selection {P4.select_best(hist2)}")
    legacy_hist, n = P4.normalise_history([[20.1, 40000]])
    check("resume_legacy_history_without_a_guard_record_is_not_selectable",
          n == 1 and legacy_hist == [(20.1, 40000, False)]
          and P4.select_best(legacy_hist) is None,
          "a pre-eligibility (S, update) pair restores as NOT eligible, so a "
          "checkpoint whose guards were never recorded cannot be selected")

    # Refusals.
    run2 = fresh("resume_h")
    bad = dict(state)
    bad["horizon"] = 100000
    code, _ = in_child(T.check_resume_state, bad, CFG, run2)
    rec = json.loads((run2 / "ABORT").read_text().splitlines()[0]) \
        if (run2 / "ABORT").exists() else None
    check("resume_refuses_a_different_horizon",
          code == T.ABORT_EXIT and rec
          and rec["checkpoint_horizon"] == 100000
          and rec["config_horizon"] == 150000,
          f"H=100,000 checkpoint against the H=150,000 config -> exit {code} "
          f"(ABORT_EXIT={T.ABORT_EXIT}) and a recorded ABORT: "
          f"{rec and rec['abort'][:70]}")
    run3 = fresh("resume_b")
    bad2 = dict(state)
    bad2["effective_batch"] = 128
    code, _ = in_child(T.check_resume_state, bad2, CFG, run3)
    check("resume_refuses_a_different_effective_batch",
          code == T.ABORT_EXIT and (run3 / "ABORT").exists(),
          f"effective_batch 128 vs the registered 252 -> exit {code}")
    code, _ = in_child(T.check_resume_state, state, CFG, fresh("resume_ok"))
    check("resume_accepts_a_matching_checkpoint", code == 0,
          f"H=150,000 / effective batch 252 -> exit {code}")


# --------------------------------------------------------------------------
# (g) terminal semantics, end to end with the real supervisor block
# --------------------------------------------------------------------------
SUP = REPO / "experiments/p4/run_supervised.sh"


def supervisor_block():
    """The terminal-outcome decision, EXTRACTED from run_supervised.sh.

    Located by its own comment marker and by the line that follows it, so the
    test tracks the file rather than a line number. run_supervised.sh belongs
    to another workstream: it is read here, never written.
    """
    src = SUP.read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(src) if "Terminal scientific outcomes" in l)
    j = next(k for k, l in enumerate(src)
             if k > i and l.strip().startswith("u=$(last_update); if"))
    return "\n".join(src[i:j]), i + 1, j


def run_supervisor_block(run):
    """Execute the extracted block against `run`, in a private working tree.

    A private tree per invocation, not a shared one: the block APPENDS its
    verdict to $L, so a supervisor log reused across cases would let one
    case's "TERMINAL STOP" line satisfy the next case's assertion. (It did,
    on the first run of this harness - which is exactly the kind of false
    pass a gate test cannot afford, so the isolation is part of the test.)
    """
    block, _, _ = supervisor_block()
    work = Path(tempfile.mkdtemp(prefix="p4sup_"))
    (work / "experiments/p4/runs").mkdir(parents=True)
    os.symlink(run, work / "experiments/p4/runs" / run.name)
    script = ("set -uo pipefail\n"
              f'cd "{work}"\n'
              f"RUN={run.name}\n"
              "L=$PWD/sup.log\n"
              "last_update() { echo 137000; }\n"
              "release_card() { :; }\n"
              + block + "\necho CONTINUE\n")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    log = work / "sup.log"
    return p, (log.read_text() if log.exists() else "")


def test_supervisor():
    block, i, j = supervisor_block()
    check("supervisor_block_located_in_the_real_file",
          "graceful_stop) : ;;" in block and "TERMINAL STOP" in block
          and "CUDA OOM" in block,
          f"{SUP.relative_to(REPO)} lines {i}-{j} extracted at run time "
          f"({j - i + 1} lines), containing the STOP and ABORT cases")

    # A real gate STOP, written by the real stop() in a child process.
    run = fresh("sup_gate")

    def _gate():
        T.check_registered_stops(F.val_with_S(20000, 19.60, D=0.1930), 20000,
                                 None, run, new_sel())
    in_child(_gate)
    p, log = run_supervisor_block(run)
    check("supervisor_does_not_restart_a_terminal_gate_STOP",
          "CONTINUE" not in p.stdout and "TERMINAL STOP" in log,
          f"STOP record {json.dumps(stop_record(run))[:80]}… -> supervisor "
          f"log: {log.strip().splitlines()[-1][:110] if log.strip() else '(empty)'}")

    # A plateau STOP, likewise.
    run = fresh("sup_plateau")

    def _plateau():
        sel = new_sel()
        for u in range(50000, 72000, 2000):
            sel = T.check_registered_stops(F.val_with_S(u, 21.00), u, None,
                                           run, sel)
    in_child(_plateau)
    p, log = run_supervisor_block(run)
    check("supervisor_does_not_restart_a_terminal_plateau_STOP",
          "CONTINUE" not in p.stdout and "TERMINAL STOP" in log,
          f"plateau STOP at update {stop_record(run)['update']} -> "
          f"{log.strip().splitlines()[-1][:110]}")

    # A watchdog STOP: registered, and therefore also terminal.
    run = fresh("sup_watchdog")
    in_child(T.stop, "reserved memory over the 13.0 GiB watchdog (SPEC §4, §8i)",
             run, update=51234)
    p, log = run_supervisor_block(run)
    check("supervisor_does_not_restart_a_watchdog_STOP",
          "CONTINUE" not in p.stdout and "TERMINAL STOP" in log,
          f"{log.strip().splitlines()[-1][:110]}")

    # The graceful card hand-over is the ONE resumable stop.
    run = fresh("sup_graceful")
    (run / "STOP").write_text(json.dumps(
        {"reason": "graceful_stop", "signal": 15, "update": 137000,
         "wall_utc": "2026-09-07T00:00:00"}))
    p, log = run_supervisor_block(run)
    check("supervisor_resumes_a_graceful_stop",
          "CONTINUE" in p.stdout and "TERMINAL" not in log,
          "STOP {\"reason\": \"graceful_stop\"} -> the block falls through to "
          "the re-queue path (CONTINUE printed)")

    # Infrastructure aborts are resumable and the record is consumed.
    run = fresh("sup_oom")
    in_child(T.abort, "CUDA OOM", run, stage="update", update=137000,
             detail="CUDA out of memory. Tried to allocate 512.00 MiB")
    p, log = run_supervisor_block(run)
    check("supervisor_resumes_an_infrastructure_ABORT",
          "CONTINUE" in p.stdout and not (run / "ABORT").exists()
          and "infrastructure ABORT" in log,
          f"ABORT 'CUDA OOM' -> requeued and the record consumed: "
          f"{log.strip().splitlines()[-1][:110]}")

    # A numerical abort is terminal and needs a human.
    run = fresh("sup_nonfinite")
    in_child(T.abort, "non-finite loss", run, stage="update", update=137000,
             detail="loss=nan at update 137000")
    p, log = run_supervisor_block(run)
    check("supervisor_does_not_restart_a_numerical_ABORT",
          "CONTINUE" not in p.stdout and "TERMINAL ABORT" in log,
          f"{log.strip().splitlines()[-1][:110]}")

    # Fail-safe direction: an unparsable STOP is treated as terminal.
    run = fresh("sup_garbled")
    in_child(T.stop, "20,000-step progress gate failed", run, update=20000)
    with open(run / "STOP", "a") as fh:
        fh.write('{"stop": "a second appended record"}\n')
    p, log = run_supervisor_block(run)
    check("supervisor_treats_an_unparsable_STOP_as_terminal",
          "CONTINUE" not in p.stdout and "TERMINAL STOP" in log,
          "two appended JSON records make json.load fail; the case falls to "
          "*) which exits without restarting - the safe direction")


# --------------------------------------------------------------------------
# A real checkpoint from a live run (opt-in: it reads 1.4 GB)
# --------------------------------------------------------------------------
def test_live_checkpoint():
    """Re-run (f)'s refusals against a checkpoint an actual arm wrote.

        P4_LIVE_CKPT=experiments/p4/runs/<run>/ckpt/step_<n>.pt \
            .venv/bin/python experiments/p4/tests/test_main_run_gate.py

    Opt-in because it reads a 1.4 GB file on a box that is training: the
    synthetic checkpoint above exercises the same code, but only a real one
    proves that the state the RUNNING trainer is writing today survives the
    resume contract this workstream just changed.
    """
    import torch
    path = os.environ.get("P4_LIVE_CKPT", "")
    if not path or not Path(path).exists():
        RESULTS.append({"name": "resume_accepts_a_live_run_checkpoint",
                        "passed": None,
                        "evidence": "skipped: set P4_LIVE_CKPT to a "
                                    "runs/<run>/ckpt/step_*.pt path"})
        return
    st = torch.load(path, map_location="cpu", weights_only=False)
    hist, legacy = P4.normalise_history(st.get("extra", {}).get("best_hist", []))

    # Match the config to the checkpoint's OWN init, not to whichever arm
    # happened to be live when this case was written. Until 2026-09-08 it
    # hard-coded init A and `horizon == 150000`; run against the init-N
    # production checkpoint it failed with exit 3 - which is check_resume_state
    # doing its job (section (k) proves that refusal is correct), not a defect
    # in the trainer. A live-checkpoint case that cannot read the live run is
    # worse than no case at all.
    init = st.get("init", (st.get("extra", {}).get("sel") or {}).get("init", "A"))
    cfg = json.loads(json.dumps(CFG))
    cfg["init"] = init
    cfg["sched"] = dict(P4.sched_for(init))
    if init == "N":
        cfg["ckpt"]["milestones"] = list(P4.MILESTONES_N)
    want_h = P4.sched_for(init)["horizon"]

    code, _ = in_child(T.check_resume_state, st, cfg, fresh("live"))
    check("resume_accepts_a_live_run_checkpoint",
          code == 0 and st["horizon"] == want_h
          and st["effective_batch"] == 252,
          f"{path}: init={init} update={st['update']} "
          f"consumed={st['consumed_batches']} H={st['horizon']} (expected "
          f"{want_h}) effective_batch={st['effective_batch']} "
          f"extra={sorted(st.get('extra', {}))} -> exit {code}; legacy "
          f"history entries normalised={legacy}")


# --------------------------------------------------------------------------
# (h) D11: a FRESH supervisor invocation must not resurrect a terminal outcome
#
# The block in (g) only runs after a worker exits, so it cannot see a STOP left
# by a PREVIOUS invocation. Until 2026-09-07 the launch path reached
# `rm -f .../STOP` first and deleted the record. These cases run the real
# script end to end with an impossible memory requirement (P4_NEED_MIB huge),
# so pick_card can never succeed and no trainer can be launched: the only two
# reachable outcomes are "refused at startup" (exit 5) and "waiting for a card".
# --------------------------------------------------------------------------
REFUSE_EXIT = 5


def run_supervisor_startup(stop=None, abort=None, timeout=8.0):
    """-> (exit code or None if it went on waiting for a card, log text)."""
    # The supervisor resolves run dirs under experiments/p4/runs, so this
    # fixture has to live there. The leading DOT keeps it out of the
    # `runs/*/` glob the live monitors use: an un-hidden fixture writing a
    # test ABORT raised a false "CUDA OOM" alarm on the running production
    # job, which is how a monitor stops being believed.
    run = REPO / "experiments/p4/runs" / (".t_d11_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    try:
        if stop is not None:
            (run / "STOP").write_text(stop, encoding="utf-8")
        if abort is not None:
            (run / "ABORT").write_text(abort, encoding="utf-8")
        env = dict(os.environ, P4_NEED_MIB="999999", P4_ONLY_PREF="1")
        proc = subprocess.Popen(
            ["bash", str(SUP), "O", "8000", run.name, "a5000"], env=env,
            cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        log = run / "supervisor.log"
        return code, (log.read_text(encoding="utf-8") if log.exists() else ""), \
            (run / "STOP").exists()
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_supervisor_startup_guard():
    for name, rec in (("gate", '{"stop": "20,000-step progress gate failed (SPEC \u00a74)"}'),
                      ("plateau", '{"stop": "plateau: 10 consecutive validations"}'),
                      ("unparsable", "this is not json")):
        code, log, survived = run_supervisor_startup(stop=rec)
        check(f"startup_refuses_a_terminal_STOP_{name}",
              code == REFUSE_EXIT and "REFUSING TO START" in log and survived,
              f"exit {code}, STOP file survived={survived}: "
              f"{log.strip().splitlines()[-1][:120] if log.strip() else '(empty)'}")

    code, log, _ = run_supervisor_startup(
        abort='{"abort": "non-finite loss", "detail": "loss=nan at update 137000"}')
    check("startup_refuses_a_numerical_ABORT",
          code == REFUSE_EXIT and "terminal ABORT" in log,
          f"exit {code}: {log.strip().splitlines()[-1][:120] if log.strip() else '(empty)'}")

    code, log, _ = run_supervisor_startup(
        abort='{"abort": "CUDA OOM", "detail": "CUDA out of memory"}')
    check("startup_does_not_refuse_an_infrastructure_ABORT",
          code is None and "REFUSING" not in log,
          "an OOM ABORT is resumable, so the supervisor goes on to wait for a "
          f"card instead of refusing (exit {code})")

    code, log, survived = run_supervisor_startup(stop='{"reason": "graceful_stop"}')
    check("startup_does_not_refuse_a_graceful_stop",
          code is None and "REFUSING" not in log,
          f"a hand-over record is resumable (exit {code}); it is consumed by the "
          "launch path, which this case cannot reach")

    code, log, _ = run_supervisor_startup()
    check("startup_allows_a_clean_run",
          code is None and "REFUSING" not in log,
          f"no STOP and no ABORT -> waits for a card (exit {code})")

    src = SUP.read_text(encoding="utf-8")
    check("launch_path_only_consumes_a_graceful_stop",
          '[ "$(stop_reason)" = graceful_stop ] && rm -f' in src
          and "rm -f experiments/p4/runs/$RUN/STOP  " not in src,
          "the unconditional `rm -f .../STOP` before a launch is gone; the "
          "record is deleted only when its reason is graceful_stop")


# --------------------------------------------------------------------------
# (i) council H Q11: a resume must not skip the 20,000 gate, and the gate's
#     PASS must be durable. The loop writes the checkpoint at a validation
#     step BEFORE validating and deciding, so a crash between the two leaves a
#     step_20000.pt that owes its decision; the resume starts at 20,001 and an
#     equality-triggered gate would never fire again.
# --------------------------------------------------------------------------
ARGS = types.SimpleNamespace(val_every=2000)


def _val_file(run, *vals):
    with open(run / "val.jsonl", "a", encoding="utf-8") as fh:
        for v in vals:
            fh.write(json.dumps(v) + "\n")


def _replay(run, update, val_or_none, sel=None, stop_at=150000, hist=None,
            decode_calls=None):
    """Drive replay_owed_validation with a decode_fn that either returns the
    given fixture or refuses to be called."""
    def decode(u):
        if decode_calls is not None:
            decode_calls.append(u)
        if val_or_none is None:
            raise AssertionError("decode_fn must not be called")
        return val_or_none
    return T.replay_owed_validation(update, run, sel or new_sel(), hist or [],
                                    json.loads(json.dumps(CFG)), ARGS, stop_at,
                                    set(), decode)


def test_resume_replay():
    # --- the boundary council H named: 19.5300 must FAIL, 19.5296 must pass.
    for s_pp, expect_pass in ((19.5296, True), (19.5297, False), (19.5300, False)):
        run = fresh("bound")
        v = F.val_with_S(20000, s_pp, D=0.1935)          # D exactly at its limit
        code, out = in_child(call_stops, v, 20000, run)
        ok = (code == 0 and "GATE20K PASS" in out) if expect_pass else \
             (code == T.STOP_EXIT and stop_record(run) is not None)
        check(f"gate_boundary_S_{str(s_pp).replace('.', '_')}_{'passes' if expect_pass else 'fails'}",
              ok, f"S={s_pp} vs limit {round(F.BASE['S_pp'] - 1.5, 4)}, D=19.35 -> exit {code}")

    # --- the D boundary: exactly 19.35 passes (the rule says <= 19.35, and the
    #     binary limit is 19.349999999999998), 19.351 fails.
    for D, expect_pass in ((0.1935, True), (0.19351, False)):
        run = fresh("dbound")
        v = F.val_with_S(20000, 19.40, D=D)
        code, out = in_child(call_stops, v, 20000, run)
        ok = (code == 0 and "GATE20K PASS" in out) if expect_pass else \
             (code == T.STOP_EXIT and stop_record(run) is not None and stop_record(run)["D_pp"] > 19.35)
        check(f"gate_boundary_D_{str(D).replace('.', '_')}_{'passes' if expect_pass else 'fails'}",
              ok, f"D={D*100:.3f} vs limit 19.35 (binary {P4.P1_DEV_FROZEN*100-1.0!r}) -> exit {code}")

    # --- a PASS is durable: file + sel flag.
    run = fresh("durable")
    v = F.val_with_S(20000, 19.40, D=0.1930)
    sel = call_stops(v, 20000, run)
    rec = json.loads((run / T.GATE20K_FILE).read_text()) if (run / T.GATE20K_FILE).exists() else None
    check("gate_pass_writes_a_durable_record",
          rec is not None and rec["result"] == "pass" and rec["update"] == 20000
          and sel.get("gate20k") == "pass" and T.gate20k_recorded(run, {}),
          f"GATE20K={rec}; sel.gate20k={sel.get('gate20k')}")

    # --- replay from an EXISTING record: no decode, decision + retention applied.
    run = fresh("replay_rec"); F.write_ckpt(run, 20000)
    v = F.val_with_S(20000, 19.40, D=0.1930); _val_file(run, v)
    calls = []
    got, sel, hist = _replay(run, 20000, None, decode_calls=calls)
    check("resume_replay_reuses_the_existing_record_without_decoding",
          got is not None and calls == [] and (run / T.GATE20K_FILE).exists()
          and sel.get("gate20k") == "pass" and any(u == 20000 for _, u, _ in hist),
          f"decode calls={calls}; GATE20K exists={(run / T.GATE20K_FILE).exists()}; hist={hist}")

    # --- replay with NO record: decodes the resumed state exactly once.
    run = fresh("replay_dec"); F.write_ckpt(run, 20000)
    v = F.val_with_S(20000, 19.40, D=0.1930)
    calls = []
    got, sel, hist = _replay(run, 20000, v, decode_calls=calls)
    check("resume_replay_decodes_the_resumed_state_when_no_record_exists",
          got is v and calls == [20000] and (run / T.GATE20K_FILE).exists(),
          f"decode calls={calls}; GATE20K exists={(run / T.GATE20K_FILE).exists()}")

    # --- replay whose decision FAILS the gate is terminal, exactly as in-run.
    run = fresh("replay_fail"); F.write_ckpt(run, 20000)
    v = F.val_with_S(20000, 19.54, D=0.1930)
    code, _ = in_child(_replay, run, 20000, v)
    rec = stop_record(run)
    check("resume_replay_gate_fail_is_terminal",
          code == T.STOP_EXIT and rec and "progress gate failed" in rec["stop"]
          and not (run / T.GATE20K_FILE).exists(),
          f"exit {code}; STOP={rec and rec['stop'][:60]}; GATE20K absent={not (run / T.GATE20K_FILE).exists()}")
    # ... and a fresh supervisor stays stopped on that record.
    code2, log, survived = run_supervisor_startup(stop=(run / "STOP").read_text())
    check("fresh_supervisor_stays_stopped_after_a_replayed_gate_fail",
          code2 == REFUSE_EXIT and survived,
          f"supervisor exit {code2}: {log.strip().splitlines()[-1][:100] if log.strip() else '(empty)'}")

    # --- nothing is owed at a non-validation step.
    run = fresh("replay_none"); calls = []
    got, _, _ = _replay(run, 20500, None, decode_calls=calls)
    check("resume_replay_owes_nothing_at_a_non_validation_step",
          got is None and calls == [],
          f"update 20500 -> replayed={got is not None}, decode calls={calls}")

    # --- a plain 2k step past the plateau anchor replays its plateau tick.
    run = fresh("replay_tick"); F.write_ckpt(run, 52000)
    v = F.val_with_S(52000, 20.00, D=0.1930); _val_file(run, v)
    sel0 = {"best_eligible_s": 19.90, "anchor": 19.90, "nonreset": 2, "gate20k": "pass"}
    got, sel, _ = _replay(run, 52000, None, sel=sel0)
    check("resume_replay_restores_a_lost_plateau_tick",
          got is not None and sel["nonreset"] == 3 and sel["anchor"] == 19.90,
          f"nonreset 2 -> {sel['nonreset']}, anchor {sel['anchor']} (S 20.00 does not beat 19.90 by 0.20)")

    # --- the invariant: past 20k with no PASS record is refused (terminal ABORT).
    run = fresh("inv_refuse")
    code, _ = in_child(T.check_gate_invariant_on_resume, 20500, 150000, run, new_sel())
    ab = json.loads((run / "ABORT").read_text().splitlines()[0]) if (run / "ABORT").exists() else None
    check("resume_invariant_refuses_past_20k_without_a_recorded_pass",
          code == T.ABORT_EXIT and ab and "no recorded 20,000" in ab["abort"],
          f"exit {code} (ABORT_EXIT={T.ABORT_EXIT}); ABORT={ab and ab['abort'][:70]}")
    # ... the supervisor treats that ABORT as terminal, not infrastructure.
    code2, log, _ = run_supervisor_startup(abort=(run / "ABORT").read_text())
    check("supervisor_treats_the_gate_invariant_ABORT_as_terminal",
          code2 == REFUSE_EXIT and "terminal ABORT" in log,
          f"supervisor exit {code2}")

    run = fresh("inv_file"); (run / T.GATE20K_FILE).write_text('{"result": "pass", "update": 20000}\n')
    code, _ = in_child(T.check_gate_invariant_on_resume, 20500, 150000, run, new_sel())
    check("resume_invariant_accepts_a_GATE20K_file", code == 0, f"exit {code}")

    run = fresh("inv_sel")
    code, _ = in_child(T.check_gate_invariant_on_resume, 20500, 150000, run, dict(new_sel(), gate20k="pass"))
    check("resume_invariant_accepts_the_checkpointed_sel_flag", code == 0, f"exit {code}")

    for name, upd, stop_at in (("inert_for_a_pilot", 8000, 8000), ("inert_at_exactly_20k", 20000, 150000),
                               ("inert_below_20k", 18000, 150000)):
        run = fresh("inv_" + name)
        code, _ = in_child(T.check_gate_invariant_on_resume, upd, stop_at, run, new_sel())
        check(f"resume_invariant_{name}", code == 0, f"update {upd}, stop_at {stop_at} -> exit {code}")

    # --- THE CONTINUATION'S FIRST ACTION: a resume from the winner's 8,000
    #     checkpoint (a validation step) against its REAL pilot record. The
    #     pilot completed decision + retention after writing that checkpoint,
    #     so the replay must reuse the record (no decode), re-establish the
    #     selection state and touch nothing terminal.
    real = REPO / "experiments/p4/runs/pilot_O+macro_filtered/val.jsonl"
    if real.exists():
        run = fresh("real8k"); F.write_ckpt(run, 8000)
        recs = [l for l in real.read_text(encoding="utf-8").splitlines() if l.strip()]
        (run / "val.jsonl").write_text("\n".join(recs) + "\n", encoding="utf-8")
        r8 = [json.loads(l) for l in recs if json.loads(l)["update"] == 8000][0]
        calls = []
        got, sel, hist = _replay(run, 8000, None, decode_calls=calls, stop_at=150000)
        check("continuation_resume_replays_macro_8k_from_its_real_record",
              got is not None and calls == [] and not (run / "STOP").exists()
              and not (run / "ABORT").exists()
              and sel["best_eligible_s"] == r8["S_pp"] and sel["anchor"] is None
              and any(u == 8000 and e for _, u, e in hist),
              f"S_pp={r8['S_pp']} eligible={P4.is_eligible(r8)}; decode calls={calls}; "
              f"sel={sel}; hist={hist}; STOP={(run / 'STOP').exists()}")
    else:
        check("continuation_resume_replays_macro_8k_from_its_real_record", None,
              "skipped: MaCro pilot val.jsonl not present")

    # --- two records for one update is ambiguous and refused.
    run = fresh("dup")
    v = F.val_with_S(20000, 19.40, D=0.1930); _val_file(run, v, v)
    code, _ = in_child(T.existing_val_record, run, 20000)
    check("existing_val_record_refuses_duplicate_records",
          code == T.ABORT_EXIT, f"two records for 20000 -> exit {code}")


# --------------------------------------------------------------------------
# (j) SPEC §8p: every STOP record classifies its ending. A plateau stop is a
#     NORMAL ending (selection + headline follow); the 20k gate failure and the
#     §5 pilot aborts are TERMINAL (no headline). The supervisor restarts
#     neither, which (g) already proves.
# --------------------------------------------------------------------------
def test_ending_class():
    run = fresh("end_plateau")

    def _plateau():
        sel = new_sel()
        for u in range(50000, 72000, 2000):
            sel = T.check_registered_stops(F.val_with_S(u, 21.00), u, None, run, sel)
    in_child(_plateau)
    rec = stop_record(run)
    check("plateau_STOP_is_a_normal_ending",
          rec is not None and rec.get("ending") == "normal" and "plateau" in rec["stop"],
          f"ending={rec and rec.get('ending')}; reason={rec and rec['stop'][:50]}")

    run = fresh("end_gate")
    in_child(call_stops, F.val_with_S(20000, 19.60, D=0.1930), 20000, run)
    rec = stop_record(run)
    check("gate_failure_STOP_is_terminal",
          rec is not None and rec.get("ending") == "terminal",
          f"ending={rec and rec.get('ending')}")

    run = fresh("end_pilot")
    in_child(call_stops, F.val_with_S(4000, 21.00, D=0.2600), 4000, run)   # D 26.0 > P1 + 5
    rec = stop_record(run)
    check("pilot_abort_STOP_is_terminal",
          rec is not None and rec.get("ending") == "terminal" and "pilot abort" in rec["stop"],
          f"ending={rec and rec.get('ending')}; reason={rec and rec['stop'][:50]}")

    run = fresh("end_bad")
    code, out = in_child(T.stop, "x", run, ending="sideways")
    check("stop_rejects_an_unknown_ending_class",
          code == 99 and "AssertionError" in out and not (run / "STOP").exists(),
          f"exit {code}; STOP written={(run / 'STOP').exists()}")


# --------------------------------------------------------------------------
# (k) init N (SPEC §8q/§8s): its own schedule and its own gates, and never A's.
# --------------------------------------------------------------------------
def _sel_n(**kw):
    d = dict(new_sel(), init="N", gate_n40=None, gate_n60=None); d.update(kw); return d


def _stops_n(val, update, run, sel=None):
    return T.check_registered_stops(val, update, None, run, sel or _sel_n())


def test_init_n():
    # schedule: council J's numbers (4.980e-4 @8k, 4.245e-4 @40k, 3.366e-4 @60k, 2.861e-4 @70k)
    SN = P4.SCHED_N
    got = {u: P4.lr_at(u, sched=SN) for u in (0, 2000, 40000, 60000, 100000, 110000, 120000)}
    check("init_n_schedule_matches_council_N_H120k",
          abs(got[2000] - 5e-4) < 1e-12 and abs(got[40000] - 3.84953e-4) < 1e-8 and abs(got[60000] - 2.61522e-4) < 1e-8
          and abs(got[100000] - 4.39193e-5) < 1e-9 and abs(got[110000] - 1.86319e-5) < 1e-9 and abs(got[120000] - 1e-5) < 1e-12
          and abs(got[0] - 1e-5) < 1e-12 and SN["horizon"] == 120000,
          " ".join(f"{u}:{v:.3e}" for u, v in got.items()))
    check("init_a_schedule_unchanged",
          abs(P4.lr_at(1000) - 2e-4) < 1e-12 and abs(P4.lr_at(8000) - 1.990e-4) < 1e-6,
          f"A: {P4.lr_at(1000):.3e} @1k, {P4.lr_at(8000):.4e} @8k")

    # configure(): N needs SCHED_N; A with SCHED_N is refused and vice versa
    cfgA = json.loads(json.dumps(CFG)); cfgN = json.loads(json.dumps(CFG))
    cfgN["init"] = "N"; cfgN["sched"] = dict(P4.SCHED_N); cfgN["ckpt"]["milestones"] = list(P4.MILESTONES_N)
    cfgN["paths"]["init_N"] = "/dev/null"; cfgN["paths"]["init_N_sha256_prefix"] = P4.INIT_N_SHA256_PREFIX
    ok_n = P4.configure(json.loads(json.dumps(cfgN)))["init"] == "N"
    bad = json.loads(json.dumps(cfgN)); bad["sched"] = dict(P4.SCHED)
    try: P4.configure(bad); rej1 = False
    except AssertionError: rej1 = True
    bad2 = json.loads(json.dumps(cfgA)); bad2["sched"] = dict(P4.SCHED_N)
    try: P4.configure(bad2); rej2 = False
    except AssertionError: rej2 = True
    check("configure_binds_schedule_to_init", ok_n and rej1 and rej2,
          f"N+SCHED_N ok={ok_n}; N+SCHED_A refused={rej1}; A+SCHED_N refused={rej2}")

    # the 20k gate does NOT fire for N (a value that would fail A's gate passes through)
    run = fresh("n_no20k")
    code, out = in_child(_stops_n, F.val_with_S(20000, 21.50, D=0.2100), 20000, run)
    check("init_n_is_not_subject_to_the_20k_gate",
          code == 0 and not (run / "STOP").exists() and not (run / T.GATE20K_FILE).exists(),
          f"S=21.50 at 20k under init N -> exit {code}, STOP={(run/'STOP').exists()}, GATE20K={(run/T.GATE20K_FILE).exists()}")
    # ... and the pilot aborts do not fire for N (60 % empties, D 30 %)
    run = fresh("n_noabort")
    code, _ = in_child(_stops_n, F.val_with_S(2000, 30.0, D=0.30, empty_rate=0.6), 2000, run)
    check("init_n_is_not_subject_to_pilot_aborts", code == 0 and not (run / "STOP").exists(),
          f"empties 60 %, D 30 % at 2k under init N -> exit {code}")
    # ... and the plateau never fires for N
    run = fresh("n_noplateau")
    def _plat():
        sel = _sel_n()
        for u in range(50000, 90000, 2000):
            sel = T.check_registered_stops(F.val_with_S(u, 19.40, D=0.1930), u, None, run, sel)
        print("SEL", json.dumps(sel))
    code, out = in_child(_plat)
    check("init_n_has_no_plateau_stop", code == 0 and not (run / "STOP").exists() and '"anchor": null' in out,
          f"20 non-improving validations 50k-88k under init N -> exit {code}, anchor stays None")

    # 40k screen: pass, fail on S, fail on D, fail on empties; the A-relative clause when supplied
    run = fresh("n40_pass"); sel = _stops_n(F.val_with_S(40000, 22.00, D=0.2130), 40000, run)
    rec = json.loads((run / T.GATE_N40_FILE).read_text()) if (run / T.GATE_N40_FILE).exists() else None
    check("init_n_40k_screen_passes_on_absolute_clauses",
          rec is not None and rec["result"] == "pass" and sel["gate_n40"] == "pass" and not (run / T.GATE20K_FILE).exists(),
          f"S 22.00 <= 22.0296, D 21.30 <= 21.348 -> {rec and rec['result']}; no GATE20K written")
    for name, kw in (("S", dict(D=0.2130)), ("D", dict(D=0.2140))):
        run = fresh("n40_fail" + name)
        v = F.val_with_S(40000, 22.10 if name == "S" else 22.00, **kw)
        code, _ = in_child(_stops_n, v, 40000, run); r = stop_record(run)
        check(f"init_n_40k_screen_fails_on_{name}", code == T.STOP_EXIT and r and "40,000" in r["stop"] and r.get("ending") == "terminal",
              f"exit {code}; {r and r['stop'][:50]}; ending={r and r.get('ending')}")
    # council K Q3: an A-relative reference, even if someone supplies one, is IGNORED
    run = fresh("n40_rel")
    sel = _stops_n(F.val_with_S(40000, 21.90, D=0.2100), 40000, run, _sel_n(a_ref_40k={"S_pp": 19.80, "D_pp": 19.20}))
    rec = json.loads((run / T.GATE_N40_FILE).read_text())
    check("init_n_40k_screen_ignores_any_A_relative_reference",
          rec["result"] == "pass" and abs(rec["S_limit_pp"] - P4.GATE_N40_S_ABS_PP) < 1e-9 and "a_ref_40k" not in rec,
          f"S 21.90 passes against the absolute 22.0296 although a 19.80 A-ref was supplied; limit recorded {rec['S_limit_pp']}")
    # ---- a fresh start must never overwrite an existing run ----
    run = fresh("fresh_empty")
    check("fresh_start_allowed_into_an_empty_run_dir",
          T.fresh_start_refusal(run, None, env={}) is None, "nothing to lose")
    (run / "ckpt").mkdir(exist_ok=True)
    (run / "ckpt" / "step_7000.pt").write_bytes(b"x")
    (run / "steps.jsonl").write_text('{"update": 7000}\n')
    ref = T.fresh_start_refusal(run, None, env={})
    check("fresh_start_refused_over_checkpoints_and_a_steps_log",
          ref is not None and ref["checkpoints"] == 1
          and ref["newest_checkpoint"] == "step_7000.pt" and ref["steps_log"] is True
          and "--resume auto" in ref["reason"],
          "a unit whose ExecStart predates the first checkpoint would otherwise "
          "retrain the production run from update 0 over its own artefacts")
    check("fresh_start_allowed_with_resume",
          T.fresh_start_refusal(run, "auto", env={}) is None, "--resume auto is the fix, not the override")
    check("fresh_start_override_is_explicit_and_env_gated",
          T.fresh_start_refusal(run, None, env={"P4_ALLOW_FRESH_OVER_EXISTING": "1"}) is None
          and T.fresh_start_refusal(run, None, env={"P4_ALLOW_FRESH_OVER_EXISTING": "0"}) is not None,
          "only the exact value 1 overrides")

    run2 = fresh("fresh_other_evidence")
    (run2 / "val.jsonl").write_text('{"update": 2000}\n')
    ref2 = T.fresh_start_refusal(run2, None, env={})
    check("fresh_start_refused_on_validation_evidence_alone",
          ref2 is not None and "val.jsonl" in ref2["other_evidence"],
          "council Q #5: a directory with validation records but no ckpt/ or steps.jsonl "
          "used to read as fresh")
    run3 = fresh("fresh_terminal_stop"); (run3 / "STOP").write_text('{"reason": "x"}\n')
    check("fresh_start_refused_over_a_terminal_stop",
          (T.fresh_start_refusal(run3, None, env={}) or {}).get("other_evidence") == ["STOP"],
          "a run the protocol ended is never silently restarted from zero")

    # council Q #3: milestone evidence must belong to the update it claims to decide
    run4 = fresh("gate_wrong_update")
    (run4 / T.GATE_N40_FILE).write_text('{"result": "pass", "update": 17}\n')
    code, _ = in_child(T.check_gate_invariant_on_resume, 40500, 120000, run4, _sel_n())
    check("milestone_evidence_from_the_wrong_update_is_rejected", code == T.ABORT_EXIT,
          f"a GATE_N40 record stamped update 17 no longer authorises a resume past 40,000 (exit {code})")

    # council Q #6: the S weight vector comes from the registered admission
    check("madasr_admission_is_registered_not_inferred",
          P4.madasr_admitted() is True and P4.madasr_admitted({}) is True,
          "a vanished panel R can no longer switch S's weights and manufacture ~1.5 pp")
    s_full, m_full = P4.selection_score({"D": 0.20, "M": 0.18, "R": 0.33, "K": 0.15}, True)
    s_drop, m_drop = P4.selection_score({"D": 0.20, "M": 0.18, "K": 0.15}, True)
    check("a_missing_panel_makes_S_undefined_not_cheaper",
          abs(s_full - 20.4) < 1e-9 and s_drop is None and m_drop["missing"] == ["R"],
          f"S={s_full} with R; without R S is None (the old path would have returned 18.9)")

    # ---- SPEC §8w (council P): the same numbers under the DIAGNOSTIC policy ----
    # The default is unchanged and terminal; only an explicit n40_policy flips it.
    # SPEC 8w.9 / council R: a MISSING key is a policy change by omission and
    # must be refused, not silently resolved to "terminal".
    try:
        P4.n40_policy_of({}); missing_ok = True
    except KeyError:
        missing_ok = False
    check("a_missing_n40_policy_is_refused_not_defaulted",
          missing_ok is False and P4.n40_policy_of({"n40_policy": "terminal"}) == "terminal",
          "an absent n40_policy raises (the trainer turns it into a recorded ABORT, "
          "never an uncaught raise that the supervisor re-queues forever)")
    check("policy_generations_pin_the_semantics",
          P4.n40_policy_generation("diagnostic") == "n40/2026-09-09/diagnostic-no-floor"
          and P4.n40_policy_generation("terminal") == "n40/2026-09-08/terminal-screen",
          "'diagnostic' meant two different things in one evening; the generation says which")
    fwd = P4.n40_migration_authority(P4.n40_policy_generation("terminal"),
                                     P4.n40_policy_generation("diagnostic"))
    back = P4.n40_migration_authority(P4.n40_policy_generation("diagnostic"),
                                      P4.n40_policy_generation("terminal"))
    check("only_listed_policy_migrations_are_authorised",
          bool(fwd) and back is None,
          f"terminal->diagnostic: {fwd}; diagnostic->terminal: {back} (unauthorised)")
    check("a_legacy_bare_name_resolves_through_the_frozen_map",
          P4.N40_LEGACY_NAME_GENERATION["diagnostic"] == "n40/2026-09-09/diagnostic-no-floor"
          and set(P4.N40_LEGACY_NAME_GENERATION) == set(P4.N40_POLICY_GENERATION),
          "never through a 'latest version' lookup, which would silently adopt a bump")
    try:
        P4.n40_policy_of({"n40_policy": "lenient"}); rej = False
    except ValueError:
        rej = True
    check("n40_policy_rejects_an_unknown_value", rej, "only 'terminal'/'diagnostic' are accepted")

    def _gate_file(run, fname, update, result):
        """Write the durable milestone record. SPEC 8w.9: a bare in-memory
        sel flag is no longer evidence - the decision must be on disk."""
        (run / fname).write_text(json.dumps({"result": result, "update": update}) + "\n")

    def _sel_diag(**kw):
        return _sel_n(n40_policy="diagnostic", **kw)

    def _seed_val(run, rows):
        """Write this run's own val.jsonl history (update -> frozen D as a fraction)."""
        with open(run / "val.jsonl", "w", encoding="utf-8") as fh:
            for u, wer in rows:
                fh.write(json.dumps({"update": u, "S_pp": 30.0,
                                     "panels": {"D": {"n": 5385, "wer_frozen": wer,
                                                      "wer_nemo": wer + 0.014,
                                                      "empty_rate": 0.0, "empties": 0}}}) + "\n")

    # a MISS under the diagnostic policy: recorded, reported, and NOT terminal
    run = fresh("n40_diag_miss"); _seed_val(run, [(6000, 0.29012), (20000, 0.2700)])
    v = F.val_with_S(40000, 26.00, D=0.2600)           # would fail the terminal screen on both
    code, out = in_child(T.check_registered_stops, v, 40000, None, run, _sel_diag())
    rec = json.loads((run / T.GATE_N40_FILE).read_text()) if (run / T.GATE_N40_FILE).exists() else None
    check("n40_diagnostic_miss_is_recorded_and_does_not_stop",
          code == 0 and not (run / "STOP").exists() and rec is not None
          and rec["result"] == "missed" and rec["former_N40_satisfied"] is False
          and rec["diagnostic_only"] is True and rec["n40_policy"] == "diagnostic",
          f"S 26.00 / D 26.00 -> exit {code}, result={rec and rec['result']}, "
          f"former_N40_satisfied={rec and rec['former_N40_satisfied']}, STOP written="
          f"{(run / 'STOP').exists()}")

    # the identical value under the TERMINAL policy still ends the run
    run = fresh("n40_term_miss")
    code, _ = in_child(_stops_n, F.val_with_S(40000, 26.00, D=0.2600), 40000, run)
    r = stop_record(run)
    check("the_same_value_under_the_terminal_policy_still_ends_phase_4",
          code == T.STOP_EXIT and r and r.get("ending") == "terminal",
          f"exit {code}; ending={r and r.get('ending')} - the THRESHOLDS did not move, only the consequence")

    # a MET diagnostic record, and the thresholds are byte-identical to the screen's
    run = fresh("n40_diag_met"); _seed_val(run, [(6000, 0.29012), (20000, 0.2400)])
    sel_d = T.check_registered_stops(F.val_with_S(40000, 22.00, D=0.2130), 40000, None, run, _sel_diag())
    rec = json.loads((run / T.GATE_N40_FILE).read_text())
    check("n40_diagnostic_met_uses_the_unchanged_thresholds",
          rec["result"] == "met" and rec["former_N40_satisfied"] is True
          and sel_d["gate_n40"] == "met"
          and abs(rec["S_limit_pp"] - P4.GATE_N40_S_ABS_PP) < 1e-9
          and abs(rec["D_limit_pp"] - P4.GATE_N40_D_ABS_PP) < 1e-9,
          f"S 22.00 vs {rec['S_limit_pp']}, D 21.30 vs {rec['D_limit_pp']} -> {rec['result']}")

    # ---- §8w.2a WITHDRAWN (council Q Q1): no terminal early-performance rule ----
    # Council Q's counterexample: 21.000 by 18k, 19.400 by 38k, 21.050 at 40k was
    # terminated by the floor although a QUALIFYING checkpoint already existed at
    # 38k; and exact equality terminated a model stable at 18.000.
    run = fresh("n40_no_floor"); _seed_val(run, [(6000, 0.29012), (16000, 0.2200),
                                                 (18000, 0.2100), (20000, 0.2160),
                                                 (36000, 0.1960), (38000, 0.1940)])
    code, _ = in_child(T.check_registered_stops, F.val_with_S(40000, 26.00, D=0.21050),
                       40000, None, run, _sel_diag())
    rec = json.loads((run / T.GATE_N40_FILE).read_text()) if (run / T.GATE_N40_FILE).exists() else None
    check("council_Q_counterexample_no_longer_terminates",
          code == 0 and not (run / "STOP").exists() and rec is not None and "floor_pass" not in rec,
          "D 21.050 at 40k after a 21.000 minimum through 20k -> continues; the withdrawn "
          "floor would have killed a run holding a qualifying 38k checkpoint")
    check("no_terminal_early_performance_rule_remains_for_N",
          "learning floor" not in (REPO / "tools/p4/p4_train.py").read_text(encoding="utf-8")
          .split("WITHDRAWN")[-1],
          "the only mention of the floor left in the gate is the withdrawal note")

    # the replacement is a WARNING: cadence-matched medians, equal observation counts
    check("delta_40_is_computed_and_is_only_a_flag",
          rec["delta_40"] is not None and rec["delta_40_flag"] is False
          and abs(rec["delta_40"] - (19.60 - 21.60)) < 1e-6,
          f"median(19.60,19.40,21.05)=19.60 minus median(22.00,21.00,21.60)=21.60 "
          f"-> {rec['delta_40']} pp, flag={rec['delta_40_flag']}")
    run = fresh("n40_delta_flag"); _seed_val(run, [(16000, 0.2200), (18000, 0.2100),
                                                   (20000, 0.2160), (36000, 0.2300), (38000, 0.2250)])
    code, out = in_child(T.check_registered_stops, F.val_with_S(40000, 26.00, D=0.2280),
                         40000, None, run, _sel_diag())
    rec2 = json.loads((run / T.GATE_N40_FILE).read_text())
    check("delta_40_flags_a_flat_trajectory_without_stopping",
          code == 0 and not (run / "STOP").exists() and rec2["delta_40_flag"] is True
          and "WARNING delta_40" in out,
          f"delta_40={rec2['delta_40']} pp >= 0 -> warning printed, exit {code}, no STOP")

    # council Q #3: a poisoned history record is refused, not skipped
    run = fresh("n40_poison"); _seed_val(run, [(16000, 0.2200), (18000, 0.2100), (20000, 0.2160)])
    with open(run / "val.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"update": -1, "panels": {"D": {"wer_frozen": 0.0}}}) + "\n")
    try:
        T._d_frozen_history(run); poisoned = False
    except ValueError:
        poisoned = True
    check("val_history_refuses_a_negative_update_record", poisoned,
          "an `update: -1, D: 0` record could poison a minimum to zero; it now raises")

    # resume past 40k: the diagnostic policy needs the RECORD, not a passing outcome
    for name, upd, sl, rec, want in (
            ("N_at_40500_diagnostic_with_a_MISSED_record_is_accepted", 40500, _sel_diag(), "missed", 0),
            ("N_at_40500_diagnostic_with_a_MET_record_is_accepted", 40500, _sel_diag(), "met", 0),
            ("N_at_40500_diagnostic_with_NO_record_is_still_refused", 40500, _sel_diag(), None, T.ABORT_EXIT),
            ("N_at_40500_terminal_with_a_MISSED_record_is_refused", 40500, _sel_n(), "missed", T.ABORT_EXIT)):
        run = fresh(name)
        if rec:
            _gate_file(run, T.GATE_N40_FILE, 40000, rec)
        code, _ = in_child(T.check_gate_invariant_on_resume, upd, 120000, run, sl)
        check(name, code == want, f"exit {code} (wanted {want})")

    # SPEC 8w.9 / council R: a bare in-memory flag with NO durable record is
    # not evidence that the milestone fired.
    run = fresh("n40_bare_flag")
    code, _ = in_child(T.check_gate_invariant_on_resume, 40500, 120000, run, _sel_diag(gate_n40="missed"))
    check("a_bare_sel_flag_without_a_record_no_longer_authorises_a_resume",
          code == T.ABORT_EXIT,
          f"sel says 'missed' but no GATE_N40 file exists -> exit {code}")
    run = fresh("n40_wrong_run")
    (run / T.GATE_N40_FILE).write_text(json.dumps(
        {"result": "missed", "update": 40000, "run_uid": "somebody-elses-run"}) + "\n")
    code, _ = in_child(T.check_gate_invariant_on_resume, 40500, 120000, run,
                       dict(_sel_diag(), run_uid="this-run"))
    check("milestone_evidence_from_another_run_uid_is_rejected", code == T.ABORT_EXIT,
          f"record carries another run's uid -> exit {code}")

    # crashes at N's gates replay the owed decision (council K acceptance case)
    for upd, fname in ((40000, T.GATE_N40_FILE), (60000, T.GATE_N60_FILE)):
        run = fresh(f"n_replay{upd}"); F.write_ckpt(run, upd)
        v = F.val_with_S(upd, 19.40, D=0.1930); _val_file(run, v)
        calls = []
        got, sel, hist = T.replay_owed_validation(upd, run, _sel_n(gate_n40="pass" if upd == 60000 else None), [],
                                                  json.loads(json.dumps(CFG)), ARGS, 150000, set(),
                                                  lambda u: (_ for _ in ()).throw(AssertionError("no decode")))
        check(f"init_n_resume_replays_the_owed_{upd}_record",
              got is not None and (run / fname).exists() and calls == [],
              f"record reused, {fname} written={(run / fname).exists()}")

    # 60k is a DIAGNOSTIC milestone (SPEC §8t.3): recorded either way, never terminal
    run = fresh("n60_met"); sel = _stops_n(F.val_with_S(60000, 19.40, D=0.1930), 60000, run)
    rec = json.loads((run / T.GATE_N60_FILE).read_text())
    check("init_n_60k_milestone_met_is_recorded", rec["result"] == "met" and rec["diagnostic_only"] and sel["gate_n60"] == "met" and not (run / "STOP").exists(),
          f"S 19.40 D 19.30 guards ok -> {rec['result']}")
    run = fresh("n60_S"); code, _ = in_child(_stops_n, F.val_with_S(60000, 19.54, D=0.1930), 60000, run)
    rec = json.loads((run / T.GATE_N60_FILE).read_text()) if (run / T.GATE_N60_FILE).exists() else None
    check("init_n_60k_milestone_missed_on_S_is_reported_not_terminal", code == 0 and rec and rec["result"] == "missed" and not (run / "STOP").exists(),
          f"S 19.54 -> exit {code}, recorded {rec and rec['result']}, STOP={(run/'STOP').exists()}")
    run = fresh("n60_G"); code, _ = in_child(_stops_n, F.val_with_S(60000, 19.40, D=0.1930, conversational=0.2210), 60000, run)
    rec = json.loads((run / T.GATE_N60_FILE).read_text()) if (run / T.GATE_N60_FILE).exists() else None
    check("init_n_60k_milestone_missed_on_guards_is_reported_not_terminal", code == 0 and rec and rec["result"] == "missed" and rec["conditions"]["all_P1_guards"] is False,
          f"conv 22.10 -> exit {code}, recorded {rec and rec['result']}")
    # ... and a later eligible checkpoint is still selectable after a missed 60k
    run = fresh("n60_later"); sel = _stops_n(F.val_with_S(60000, 19.54, D=0.1930), 60000, run)
    sel = _stops_n(F.val_with_S(62000, 19.30, D=0.1920), 62000, run, sel)
    check("init_n_later_eligible_checkpoint_selectable_after_a_missed_60k", sel["best_eligible_s"] == 19.30 and sel["gate_n60"] == "missed",
          f"best_eligible_s after 62k = {sel['best_eligible_s']}, 60k recorded {sel['gate_n60']}")

    # resume invariants per init
    # SPEC 8w.9: the 40k milestone is authorised by its durable RECORD, never by
    # a bare sel flag, so rows that expect acceptance write the record.
    for name, upd, sel, rec40, expect in (
            ("N_at_20500_without_GATE20K_is_accepted", 20500, _sel_n(), None, 0),
            ("N_at_40500_without_GATE_N40_is_refused", 40500, _sel_n(), None, T.ABORT_EXIT),
            ("N_at_40500_with_its_record_is_accepted", 40500, _sel_n(), "pass", 0),
            ("N_at_60500_without_GATE_N60_is_ACCEPTED_(60k_is_a_report)", 60500, _sel_n(), "pass", 0),
            ("N_at_60500_with_a_missed_60k_is_accepted", 60500, _sel_n(gate_n60="missed"), "pass", 0),
            ("A_at_20500_without_GATE20K_is_still_refused", 20500, new_sel(), None, T.ABORT_EXIT)):
        run = fresh("inv_" + name[:12])
        if rec40:
            (run / T.GATE_N40_FILE).write_text(json.dumps({"result": rec40, "update": 40000}) + "\n")
        code, _ = in_child(T.check_gate_invariant_on_resume, upd, 150000, run, sel)
        check("resume_invariant_" + name, code == expect, f"exit {code} (expected {expect})")
    run = fresh("inv_nfile"); (run / T.GATE_N40_FILE).write_text('{"result": "pass", "update": 40000}\n')
    code, _ = in_child(T.check_gate_invariant_on_resume, 40500, 150000, run, _sel_n())
    check("resume_invariant_N_accepts_a_GATE_N40_file", code == 0, f"exit {code}")

    # should_validate forces N's gates, not A's
    sv = T.should_validate
    check("should_validate_forces_N_gates_and_not_20k",
          sv(40000, 0, 150000, set(), init="N") and sv(60000, 0, 150000, set(), init="N") and not sv(20000, 0, 150000, set(), init="N")
          and sv(20000, 0, 150000, set(), init="A") and not sv(40000, 0, 150000, set(), init="A"),
          "N: 40k/60k forced, 20k not; A: 20k forced, 40k not (with --val-every 0)")

    # check_resume_state refuses an init mismatch
    st = dict(F.resume_state(20000)) if hasattr(F, "resume_state") else None
    if st is None:
        st = {"update": 8000, "horizon": 150000, "effective_batch": 252}
    st["init"] = "N"
    code, _ = in_child(T.check_resume_state, st, CFG, fresh("init_mismatch"))
    check("resume_refuses_an_init_mismatch", code == T.ABORT_EXIT, f"N checkpoint under an A config -> exit {code}")
    st["init"] = "A"
    code, _ = in_child(T.check_resume_state, st, CFG, fresh("init_match"))
    check("resume_accepts_a_matching_init", code == 0, f"A checkpoint under an A config -> exit {code}")


# --------------------------------------------------------------------------
# (l) SPEC §8w: the Phase-1 comparison curve. Councils I, J and O were given a
#     curve built by enumerating lightning_logs.txt "New best score" lines and
#     labelling event i as update (i+1)*1000. Those are EarlyStopping improvement
#     events, not a 1,000-update grid, so the series was the running-minimum
#     envelope on a wrong x-axis - understating the true update by up to 16,005
#     (84%). No gate ever read it (SPEC: the 40k thresholds are "not ... estimates
#     derived from the historical cold-start curve"), but nothing must re-emit it.
# --------------------------------------------------------------------------
BAD_ENVELOPE = [56.6, 40.3, 39.8, 32.8, 29.5, 27.6, 26.2, 25.9, 25.5, 24.5,
                23.7, 22.9, 22.6, 22.3, 21.9, 21.5, 21.3, 21.2, 20.8]


def test_phase1_curve():
    curve = P4.phase1_val_curve()
    check("phase1_curve_artefact_exists", curve is not None and len(curve) == 39,
          f"{0 if curve is None else len(curve)} validations")
    if not curve:
        return

    # 1. the artefact agrees with a fresh extraction from the event protobufs.
    # The Phase-1 event files are raw data from an earlier phase and are not
    # carried into a Phase-4-only checkout. Their absence is REPORTED as a
    # skipped cross-check, never as a silent pass: checks 2-4 below still hold
    # the artefact to its known anchors from the committed file alone.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p4_phase1_curve", REPO / "tools/p4/p4_phase1_curve.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    try:
        fresh_curve = mod.val_wer_by_update()
    except (FileNotFoundError, NotADirectoryError):
        fresh_curve = None
    if fresh_curve:
        check("phase1_curve_matches_tfevents",
              all(abs(fresh_curve[u] - v) < 1e-6 for u, v in curve.items())
              and set(fresh_curve) == set(curve),
              f"{len(fresh_curve)} points re-extracted, max delta "
              f"{max(abs(fresh_curve[u] - v) for u, v in curve.items()):.2e}")
    else:
        print("SKIP  phase1_curve_matches_tfevents\n      Phase-1 event files "
              "absent (expected in a Phase-4-only checkout); the artefact is "
              "still checked against its anchors below")

    # 2. the true anchors: 56.55 at 2,000 (NOT 40.3), best 20.826 at 35,005
    check("phase1_u2000_is_56pt55", abs(curve[2000] - 56.5503) < 1e-3,
          f"update 2,000 -> {curve[2000]} (the corrupted curve said 40.3)")
    best_u = min(curve, key=lambda u: curve[u])
    check("phase1_best_is_20pt826_at_35000",
          best_u == 35000 and abs(curve[best_u] - 20.8259) < 1e-3,
          f"best {curve[best_u]} at update {best_u}")

    # 3. the corrupted series IS the improvement envelope, and the true curve is not it
    best = 1e9; env = []
    for u in sorted(curve):
        if curve[u] < best:
            best = curve[u]; env.append(round(curve[u], 1))
    check("corrupted_curve_reproduced_as_improvement_envelope",
          env[1:] == BAD_ENVELOPE,
          f"{len(env) - 1} improvement events after the pre-resume point")
    check("true_curve_is_not_the_envelope",
          [round(curve[u], 1) for u in sorted(curve)][:19] != BAD_ENVELOPE,
          "per-validation series differs from the envelope")

    # 4. no builder or tool can re-emit the defect
    offenders = []
    for f in sorted((REPO / "experiments/p4").glob("make_council_*.py")) + \
             sorted((REPO / "tools/p4").glob("*.py")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for ln in txt.split("\n"):
            if "New best score" in ln and "re.findall" in ln:
                offenders.append(f"{f.name}: {ln.strip()[:60]}")
    check("no_builder_rebuilds_the_curve_from_lightning_logs",
          not offenders, "; ".join(offenders) or "no re.findall over 'New best score'")

    # 5. the registered thresholds do NOT come from the curve
    base = P4.load_baseline()
    if base is None:
        check("thresholds_derive_from_measured_baseline_not_the_curve", None, "no baseline_p1.json")
    else:
        check("thresholds_derive_from_measured_baseline_not_the_curve",
              abs(P4.GATE_N40_D_ABS_PP - (base["D"]["wer_frozen"] * 100 + 1.0)) < 1e-9
              and abs(P4.GATE_N40_S_ABS_PP - (base["S_pp"] + 1.0)) < 1e-9,
              f"D {P4.GATE_N40_D_ABS_PP} = {base['D']['wer_frozen'] * 100:.3f}+1.0, "
              f"S {P4.GATE_N40_S_ABS_PP} = {base['S_pp']}+1.0")


# --------------------------------------------------------------------------
# (m) SPEC §8w.9 (council R): decisions run on the EXACT metric, not the 5-dp
#     display value, and the registered panel/source set is required in full.
# --------------------------------------------------------------------------
DEN = {"D": 55191, "M": 16669, "R": 14434, "K": 23995}


def _panels_from_edits(edits):
    return {k: {"counts": {"w_edits": edits[k], "w_ref": DEN[k]}} for k in DEN}


def _recorded_S(edits):
    """S the way the pre-8w.9 path computed it: panel WERs rounded to 5 dp first."""
    return P4.selection_score({k: round(edits[k] / DEN[k], 5) for k in DEN}, True)[0]


def test_exact_decisions():
    # council R's verified counterexample, on the real denominators
    a = {"D": 9986, "M": 3009, "R": 4000, "K": 3300}    # "100,000"
    b = {"D": 9950, "M": 3000, "R": 4000, "K": 3300}    # "102,000"
    ax, _ = P4.selection_score_exact(_panels_from_edits(a), True)
    bx, _ = P4.selection_score_exact(_panels_from_edits(b), True)
    ar, br = _recorded_S(a), _recorded_S(b)
    check("exact_S_matches_council_R_to_ten_places",
          abs(ax - 18.6129335071) < 1e-9 and abs(bx - 18.5629982031) < 1e-9,
          f"exact S {ax:.10f} / {bx:.10f}")
    check("the_rounded_metric_splits_a_tie_the_exact_one_keeps",
          (ax - bx) < P4.TIE_BAND_PP and (ar - br) > P4.TIE_BAND_PP,
          f"exact ΔS {ax - bx:.9f} < {P4.TIE_BAND_PP} (tied, earlier wins); "
          f"recorded ΔS {ar - br:.9f} > {P4.TIE_BAND_PP} (splits, later wins)")
    # and the consequence: select_best on exact keeps the EARLIER checkpoint
    hist_exact = [(ax, 100000, True), (bx, 102000, True)]
    hist_round = [(ar, 100000, True), (br, 102000, True)]
    check("select_best_on_exact_S_keeps_the_earlier_checkpoint",
          P4.select_best(hist_exact)[1] == 100000 and P4.select_best(hist_round)[1] == 102000,
          f"exact selects {P4.select_best(hist_exact)[1]:,}; "
          f"rounded selects {P4.select_best(hist_round)[1]:,} - and retain would then "
          f"delete every copy of 100,000, which no preserved hypothesis can rebuild")

    # qualification flip
    q = {"D": 10000, "M": 3010, "R": 8908, "K": 3300}
    qx, _ = P4.selection_score_exact(_panels_from_edits(q), True)
    qr = _recorded_S(q)
    tol = P4.GATE20K_REPR_TOL_PP
    check("qualification_flips_on_the_discarded_precision",
          qx > P4.GATE_N40_S_ABS_PP + tol and qr <= P4.GATE_N40_S_ABS_PP + tol,
          f"exact S {qx:.12f} FAILS vs {P4.GATE_N40_S_ABS_PP}; recorded {qr} PASSES")

    # records with no counts fall back to the recorded value - past decisions stand
    legacy = {k: {"wer_frozen": round(v / DEN[k], 5)} for k, v in a.items()}
    ls, lm = P4.selection_score_exact(legacy, True)
    check("records_without_counts_fall_back_and_do_not_raise",
          ls is not None and lm["from_counts"] is False and abs(ls - ar) < 1e-9,
          f"legacy record -> S {ls} (== the recorded {ar}), no exception")

    # a registered source missing from the evaluation is a FAILED guard
    base = {"per_source": {x: 0.10 for x in "abcde"},
            "conversational": {"wer_frozen": 0.20}}
    cur = {"per_source": {"a": 0.10}, "conversational": {"wer_frozen": 0.20},
           "empty_rate": 0.0}
    g = P4.guards(cur, base)
    missing = [c["check"] for c in g["checks"] if c.get("evaluable") is False]
    check("dropping_registered_sources_no_longer_passes_the_guards",
          g["pass"] is False and sorted(missing) == ["D/b", "D/c", "D/d", "D/e"],
          f"four of five D sources removed -> pass={g['pass']}, non-evaluable {sorted(missing)}")

    # zero total reference words blocks authoritative evaluation, never scores 0
    zero = {"counts": {"w_edits": 0, "w_ref": 0, "empties": 0, "n": 10},
            "per_source_counts": {"a": [0, 0]}, "conversational": {"wer_frozen": 0.2},
            "empty_rate": 0.0}
    raised = False
    try:
        P4.exact_wer(zero)
    except P4.PanelIntegrityError:
        raised = True
    gz = P4.guards(zero, base)
    check("a_panel_with_zero_reference_words_cannot_score_a_pass",
          raised and gz["pass"] is False and gz["evaluable"] is False,
          "exact_wer raises PanelIntegrityError; guards() reports it as non-evaluable, not a pass")


# --------------------------------------------------------------------------
def main():
    if "--case" in sys.argv:                     # child entry point (unused
        return 0                                 # by default; fork is cheaper)
    for fn in (test_gate20k, test_comparator, test_eligibility, test_plateau,
               test_retention, test_resume, test_live_checkpoint,
               test_supervisor, test_supervisor_startup_guard,
               test_resume_replay, test_ending_class, test_init_n,
               test_phase1_curve, test_exact_decisions):
        print(f"\n=== {fn.__name__} " + "=" * (60 - len(fn.__name__)))
        fn()
    n_pass = sum(1 for r in RESULTS if r["passed"] is True)
    n_fail = sum(1 for r in RESULTS if r["passed"] is False)
    n_skip = sum(1 for r in RESULTS if r["passed"] is None)
    out = HERE / "results.json"
    out.write_text(json.dumps(
        {"checks": RESULTS, "passed": n_pass, "failed": n_fail,
         "skipped": n_skip,
         "code_sha256": {f: P4.sha256_file(REPO / "tools/p4" / f)
                         for f in ("p4_train.py", "p4_common.py")},
         "supervisor_sha256": P4.sha256_file(SUP)}, indent=1),
        encoding="utf-8")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped -> {out}")
    for r in RESULTS:
        if r["passed"] is False:
            print("FAILED: " + r["name"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
