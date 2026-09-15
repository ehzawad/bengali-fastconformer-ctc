#!/usr/bin/env python
"""Shared surface for the this recipe warm-start trainer (experiments/SPEC.md §4, §5).

Everything the this recipe trainer, its launcher and its watchdog need that is a
REGISTERED number lives here exactly once, and every constant carries the SPEC
section it is transcribed from. ``configure(cfg)`` asserts that
``configs/ctc.yaml`` still agrees with these values, so an edit to the YAML
can never move a registered number without also moving this file - which is a
code review, which is the point (same discipline as bnasr/scoring.py).

Nothing is re-implemented here that the repo already froze:

  * the SCORING surface is ``bnasr.scoring.frozen_norm`` / ``wer_counts`` /
    ``cer_counts`` / ``nemo_wer`` - the byte-for-byte benchmark scorer the
    published the baseline (20.35 % frozen / 20.83 % NeMo-internal on panel D),
    Phase-2 (17.03 %) and hishab (22.30 %) numbers were measured on. A
    this recipe number scored on anything else is not comparable with them.
  * the TRAINING-TARGET surface is ``bnasr.normalize.metric_surface(NFC(t),
    False)`` (SPEC §1 RULING), applied by ``bnasr/build_corpus.py`` when
    the arm manifests are written. This module only re-applies it when it
    builds the smoke manifests, so there is one implementation, not two.
  * the GPU pin, the process anonymisation, the sha256 helper and the hourly
    logger are ``bnasr.scoring``'s.

What is NOT reused is ``scoring.dev_eval``: it decodes an RNNT/TDT model and
reaches into ``model.joint`` (fused loss/WER flags) which an
``EncDecCTCModelBPE`` does not have. ``ctc_panel_eval`` below is the CTC
equivalent and it calls the SAME scorer helpers, so the surface is shared even
though the decode path cannot be.
"""
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bnasr.scoring import (  # noqa: E402,F401  (re-exported on purpose)
    A5000_UUID, HourlyLogger, anonymize_process, cer_counts, frozen_norm,
    load_manifest, nemo_wer, pin_gpu, sha256_file, wer_counts)
from bnasr.normalize import metric_surface  # noqa: E402

GIB = float(1 << 30)

# ---------------------------------------------------------------------------
# Registered constants. Every entry cites the SPEC section it comes from; a
# number that is not in SPEC is not registered and is marked as such.
# ---------------------------------------------------------------------------
INIT_NEMO = REPO / "experiments/fastconformer_ctc_bn_1kh/final.nemo"
INIT_SHA256_PREFIX = "61362a36"        # SPEC §4 ("sha256 61362a36…")
TOKENIZER_DIR = REPO / "data/tokenizer_v1024_corpus"
TOKENIZER_SHA256_PREFIX = "6288cf8e"   # SPEC §3 G0.5

# SPEC §4: "effective batch 252 = 12 x 21".
MICROBATCH = 12
ACCUM = 21
EFFECTIVE_BATCH = 252
assert MICROBATCH * ACCUM == EFFECTIVE_BATCH, "SPEC §4: 252 = 12 x 21"

# SPEC §4: AdamW (0.9, 0.98) wd 1e-3, clip 1.0, bf16.
OPTIM = {
    "betas": (0.9, 0.98),
    "eps": 1.0e-8,          # NOT in SPEC; torch's AdamW default, stated here
                            # so the run record is unambiguous.
    "weight_decay": 1.0e-3,  # 0 for biases and every norm scale/shift
    "grad_clip": 1.0,
}

# SPEC §4: "warmup 1,000 steps 1e-5 -> peak 2e-4 ... cosine to 1e-5 at
# H = 150,000 steps". `horizon` is the denominator BOTH the cosine leg and the
# ETA use; it is a runtime value only so that a pilot (--steps 8000) decays on
# the SAME 150k curve rather than on its own shorter one - the pilot's 8k
# checkpoint has to resume "to H without a second warmup" (SPEC §5), which is
# only true if its LRs were on the H-curve all along.
SCHED = {
    "warmup_steps": 1000,
    "lr_start": 1.0e-5,
    "lr_peak": 2.0e-4,
    "lr_final": 1.0e-5,
    "horizon": 150000,
}
HORIZON = 150000

# SPEC §8q/§8s (councils I, J, K; owner 2026-09-08 "use NVIDIA weights first"):
# init N = nvidia/stt_en_fastconformer_ctc_large (English) + a FRESH 1,025-way
# Bengali CTC head, trained with the baseline's own cold-start recipe on the SAME
# 150,000 horizon. Init A (the baseline warm start) is unchanged.
INITS = ("A", "N")
SCHED_N = {
    "warmup_steps": 2000,
    "lr_start": 1.0e-5,
    "lr_peak": 5.0e-4,
    "lr_final": 1.0e-5,
    "horizon": 120000,      # council N / SPEC §8v (2026-09-08): 27.9 epochs of 1,084,696 rows; was 150,000
}
HORIZON_N = 120000
MILESTONES_N = (40000, 60000, 80000, 120000)
INIT_N_SHA256_PREFIX = "1f2c819d"     # the HF blob name IS its sha256
INIT_N_HEAD_SEED = 20260905            # torch seed before change_vocabulary()
# Init N's own progress gates (council I §8q, absolute clauses; the A-relative
# clause needs A's LITERAL 40k checkpoint and is applied only when supplied).
GATE_N40_UPDATE = 40000
GATE_N40_S_ABS_PP = 22.0296            # S(P1) + 1.00
GATE_N40_D_ABS_PP = 21.348             # D(P1) + 1.00
# (the A-relative +2.00 clauses of council I were DELETED by council K, SPEC §8s)
GATE_N60_UPDATE = 60000                # same absolute numbers as A's 20k gate
# SPEC §8w (council P): the update-40,000 screen's POLICY, not its numbers.
#   "terminal"   - the registered §8s/§8t screen: a miss ends this recipe (default;
#                  this is what stands until the owner records an adoption).
#   "diagnostic" - council P's amendment: the same conditions are measured and
#                  recorded as `former_N40_satisfied`, a miss stops nothing, and
#                  the sole terminal qualification moves to the checkpoint the
#                  registered selector picks after the run completes H.
# The THRESHOLDS are identical either way; only the consequence of a miss moves.
N40_POLICIES = ("terminal", "diagnostic")
N40_POLICY_DEFAULT = "terminal"

# The "diagnostic" policy removes the only performance-based stop init N has:
# the §5 pilot aborts are init-A only (`if d is not None and init == "A"`), so
# without this the run would have NO performance termination between update
# 6,000 and update 120,000 - 92-98 GPU-h with nothing watching. Council P chose
# that; I am not willing to ship it.
#
# This floor is deliberately NOT a quality bar and NOT derived from the baseline. It
# asks one question of the run's OWN measurements, on its own panel and surface,
# with no cross-run conversion and no chosen constant: after 20,000 more updates
# and another 17 % of the anneal, is it still learning at all? A run that at
# 40,000 is no better than its own best by 20,000 will not be rescued by more of
# the same. Any healthy trajectory clears it by a wide margin; the earlier 27 %
# proposal was rejected by council P for resting on an unverified frozen
# conversion of the baseline's curve, and this rests on neither.
N40_FLOOR_REF_UPDATE = 20000      # compare against the run's own best D by here
N40_FLOOR_EMPTY_RATE = 0.01       # > 1 % empties at 40k is a broken decoder


# ---- policy GENERATION (council R, SPEC 8w.9) -------------------------------
# A bare name cannot identify semantics: "diagnostic" meant two different things
# within one evening (with the 8w.2a floor, then without it). A generation
# string pins the meaning, a frozen legacy map resolves an old bare name to the
# generation it ACTUALLY had - never to "whatever is current" - and only listed
# transitions are authorised.
N40_POLICY_GENERATION = {
    "terminal":   "n40/2026-09-08/terminal-screen",
    "diagnostic": "n40/2026-09-09/diagnostic-no-floor",
}
# What a bare name found in an OLD checkpoint meant when it was written. Frozen:
# extend it when a generation is added, never repoint an existing entry.
N40_LEGACY_NAME_GENERATION = {
    "terminal":   "n40/2026-09-08/terminal-screen",
    "diagnostic": "n40/2026-09-09/diagnostic-no-floor",
}
# (from_generation, to_generation) -> the SPEC section that authorised it.
N40_ALLOWED_MIGRATIONS = {
    ("n40/2026-09-08/terminal-screen", "n40/2026-09-09/diagnostic-no-floor"):
        "SPEC 8w.2 (councils O+P), adopted 2026-09-09 at update 7,212",
}


def n40_policy_generation(name):
    """The generation string for a policy name, or None if unknown."""
    return N40_POLICY_GENERATION.get(str(name))


def n40_migration_authority(from_gen, to_gen):
    """The SPEC section authorising from_gen -> to_gen, or None if unauthorised."""
    if not from_gen or from_gen == to_gen:
        return "same generation"
    return N40_ALLOWED_MIGRATIONS.get((from_gen, to_gen))


def n40_policy_of(cfg):
    """The registered N40 policy for this config, validated."""
    if cfg is not None and "n40_policy" not in cfg:
        # Council R: a MISSING key must never silently restore "terminal" - that
        # is a policy change by omission. The trainer turns this into abort().
        raise KeyError("n40_policy is mandatory: the config must state the "
                       "registered policy explicitly (SPEC 8w.2)")
    pol = str((cfg or {}).get("n40_policy", N40_POLICY_DEFAULT))
    if pol not in N40_POLICIES:
        raise ValueError(f"n40_policy must be one of {N40_POLICIES}, got {pol!r}")
    return pol


GATE_N60_S_PP = 19.5296
GATE_N60_D_PP = 19.35


def sched_for(init):
    assert init in INITS, init
    return SCHED_N if init == "N" else SCHED

# SPEC §4 (the baseline values): SpecAugment 2 x 15 freq / 2 x 0.05 time. These are
# ASSERTION targets against the module already inside final.nemo - this recipe never
# rebuilds the augmentation, it checks that init A carries it.
SPECAUG = {"freq_masks": 2, "freq_width": 15,
           "time_masks": 2, "time_width": 0.05}

# SPEC §4: validation every 2,000 steps; checkpoints last 2 resumable, best 3
# by S, immutable milestones (pilot end, 20k, 40k, 80k, 120k, H).
VAL_EVERY = 2000
PILOT_STEPS = 8000                          # SPEC §5, "8,000 steps each"
KEEP_LAST = 2
KEEP_BEST = 3
MILESTONES = (PILOT_STEPS, 20000, 40000, 80000, 120000, HORIZON)

# SPEC §3 G0.7 / §4 / §8i: profile target 16.5 GiB, watchdog 13.0 GiB (share-a-card).
PROFILE_TARGET_GIB = 16.5
WATCHDOG_GIB = 13.0        # owner 2026-09-07 share-a-card ruling (was 17.0): measured peak reserved is 9.23 GiB

# SPEC §2 selection score and regression guards.
S_WEIGHTS_WITH_MADASR = {"D": 0.60, "M": 0.20, "R": 0.10, "K": 0.10}
S_WEIGHTS_NO_MADASR = {"D": 0.60, "M": 0.30, "K": 0.10}
assert abs(sum(S_WEIGHTS_WITH_MADASR.values()) - 1.0) < 1e-12
assert abs(sum(S_WEIGHTS_NO_MADASR.values()) - 1.0) < 1e-12
GUARD_SOURCE_PP = 1.0        # no D source worse than P1 by more than 1.0 pp
GUARD_CONVERSATIONAL_PP = 0.5
GUARD_EMPTY_RATE = 0.001     # empty-output rate <= 0.1 %

# SPEC §5 pilot abort rules and §4 run gates.
PILOT_ABORT_EMPTY_RATE = 0.01     # > 1 % empties stops an arm
PILOT_ABORT_D_PP = 5.0            # D worse than P1 by > 5 pp stops an arm

# SPEC §4: "Progress gate at 20,000: S <= S(P1) - 1.5 pp and D <= D(P1) - 1.0 pp
# and guards pass; else STOP." SPEC §8k Q8 restates the SAME gate in absolute
# points - "S <= 19.530 AND D <= 19.35 AND all main-run guards, else this recipe
# TERMINATES" - and calls it "the decisive cheap check". Both forms are kept
# here and reconciled by the assertions below, because the relative form is
# what the code evaluates (S(P1) is a measured number, not a registered one)
# and the absolute form is what council F ruled on; if the measured baseline
# ever drifted away from the ruling, the run must not quietly re-aim the gate.
GATE20K_UPDATE = 20000
GATE20K_S_PP = 1.5                # at 20,000: S <= S(P1) - 1.5 pp
GATE20K_D_PP = 1.0                # and D <= D(P1) - 1.0 pp
GATE20K_S_ABS_PP = 19.530         # SPEC §8k Q8, absolute form
GATE20K_D_ABS_PP = 19.35          # SPEC §8k Q8, absolute form
# 0.001 pp: §8j records S(P1) as 21.030 (3 dp) while baseline_p1.json measured
# 21.0296, so the two forms differ by 0.0004 pp and by nothing else.
GATE20K_ABS_TOL_PP = 0.001
# Float REPRESENTATION tolerance for the two gate comparisons, NOT gate slack
# (council H Q5: "S = 19.5300 fails; do not round the observed S or add 0.001").
# D(P1)*100 - 1.0 evaluates to 19.349999999999998 in binary, so a measured D of
# exactly 19.35 - which the registered rule "D <= 19.35" admits - would fail on
# representation alone. 1e-9 pp is five orders of magnitude below the D panel's
# WER quantum (one word in ~30,000 reference words ~= 0.003 pp) and cannot let
# 19.5297 pass against 19.5296.
GATE20K_REPR_TOL_PP = 1e-9
assert abs((21.030 - GATE20K_S_PP) - GATE20K_S_ABS_PP) <= GATE20K_ABS_TOL_PP, (
    "SPEC §8k Q8: S(P1) 21.030 - 1.5 pp must be the quoted 19.530")

# SPEC §4 plateau, as RE-RULED by council F Q4 (SPEC §8k): "anchor 50k,
# >= 0.20 pp cumulative reset, 10 non-reset validations, earliest stop 70k".
# This SUPERSEDES the earlier "5 validations from 30,000" rule that this file
# carried until 2026-09-07; the two differ in every one of their three numbers
# and in what the improvement is measured AGAINST (a moving best-so-far versus
# a sticky anchor). The anchor is what makes the 0.20 pp CUMULATIVE: a run
# that improves 0.15 pp per validation forever never resets a best-so-far
# comparison, because the best moves with it, but it does cross a fixed anchor.
PLATEAU_ANCHOR_AT = 50000         # anchor initialised here to best eligible S
PLATEAU_NONRESET = 10             # stop after 10 consecutive non-reset vals
PLATEAU_MIN_PP = 0.20             # cumulative improvement of the ANCHOR
PLATEAU_EARLIEST_STOP = 70000     # SPEC §4 "earliest stop 70k"
assert PLATEAU_ANCHOR_AT + PLATEAU_NONRESET * VAL_EVERY == PLATEAU_EARLIEST_STOP, (
    "SPEC §4/§8k Q4: 50,000 + 10 validations x 2,000 steps = 70,000, the "
    "earliest step at which the plateau rule may stop the run")

# SPEC §4: "Final checkpoint = lowest eligible S; ties within 0.05 pp ->
# earlier." SPEC §8k Q1 re-affirms the tie band with S kept frozen.
TIE_BAND_PP = 0.05

# SPEC §0/§2 published baselines on the locked dev panel D, frozen surface.
# Only D's is a registered NUMBER; the per-source and conversational-slice
# baselines are MEASURED by --restore-check (they are evidence, not registered
# constants) and land in experiments/baseline_p1.json.
P1_DEV_FROZEN = 0.2035            # SPEC §2 ("P1 20.35")
P1_DEV_NEMO_INTERNAL = 0.2083     # task brief / the baseline record, reported only
RESTORATION_TOL = 0.0005          # SPEC §4: "equals ... 20.35 % (frozen) ±0.05"
BASELINE_JSON = REPO / "data/baselines/baseline.json"
PHASE1_CURVE_JSON = REPO / "data/baselines/baseline_val_curve.json"

# Panels (SPEC §2). M is written by bnasr/build_corpus.py and may not
# exist yet; every consumer here treats a missing panel as absent, never as
# zero. FLEURS is deliberately NOT in this table: SPEC §2 seals it for a single
# decode in §6 and this trainer must never be able to touch it.
PANELS = {
    "D": REPO / "data2/val_corpus.json",
    "M": REPO / "experiments/corpus/panel_M.jsonl",
    "K": REPO / "data4/kathbath/manifest_valid.jsonl",
    "R": REPO / "data4/respin/manifest_madasr_dev.jsonl",
}

# The conversational slice of D (SPEC §2 "conversational slice of D", §5's
# guard). MEASURED, not assumed: data2/val_corpus.json has no `style` key. Its
# 2,276 IndicVoices rows carry `scenario` in {Extempore 1282, Conversation 774,
# Read 220}; the other 3,109 rows (openslr53/kathbath/commonvoice/subakko) have
# no scenario at all. So "IndicVoices rows whose style is Conversation" is
# source == "indicvoices" and scenario == "Conversation" - 774 rows.
CONVERSATIONAL = {"source": "indicvoices", "scenario": "Conversation"}


def is_conversational(row):
    return (row.get("source") == CONVERSATIONAL["source"]
            and row.get("scenario") == CONVERSATIONAL["scenario"])


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def lr_at(step, horizon=None, sched=None):
    """LR for the update about to be applied, 0-based (SPEC §4).

    step 0        -> 1e-5   (the first update of the run)
    step 1,000    -> 2e-4   (end of the linear warmup)
    step H        -> 1e-5   (end of the cosine)

    `horizon` is an explicit argument for the same reason it is in
    scoring.lr_at: the cosine DENOMINATOR is the one number whose silent
    change would leave every LR after warmup wrong while the run still looks
    healthy. A pilot stopping at 8,000 must still decay on the 150,000
    denominator, because SPEC §5 resumes that pilot "to H without a second
    warmup" - a pilot that had decayed to 1e-5 by step 8,000 could not.
    """
    sc = SCHED if sched is None else sched
    h = int(sc["horizon"] if horizon is None else horizon)
    w = int(sc["warmup_steps"])
    s0, s1, s2 = sc["lr_start"], sc["lr_peak"], sc["lr_final"]
    if step <= 0:
        return s0
    if step <= w:
        return s0 + (s1 - s0) * (step / w)
    if step >= h:
        return s2
    t = (step - w) / (h - w)
    return s2 + (s1 - s2) * 0.5 * (1.0 + math.cos(math.pi * t))


def make_param_groups(model):
    """Two AdamW groups: decayed weights, and bias/norm parameters at wd 0.

    SPEC §4 registers "wd 1e-3 (no decay on bias/norm)". Decaying a LayerNorm
    gain or a BatchNorm scale pulls the normalisation towards zero-gain; on a
    WARM START that is worse than on a fresh model, because those buffers and
    scales are exactly the part of init A that this recipe is trying to preserve
    (council B item 17: "preserve 100 % of normalisation buffers on load").

    The split is by shape and name, not by module walk, so a parameter that a
    future NeMo version renames still lands in the right group as long as it is
    1-D (every bias, every norm scale/shift in this model is).
    """
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            nodecay.append(p)
        else:
            decay.append(p)
    assert decay and nodecay, "parameter split produced an empty group"
    return [
        {"name": "decay", "params": decay,
         "weight_decay": OPTIM["weight_decay"], "lr": lr_at(0)},
        {"name": "nodecay", "params": nodecay, "weight_decay": 0.0,
         "lr": lr_at(0)},
    ]


# ---------------------------------------------------------------------------
# Config guard
# ---------------------------------------------------------------------------
def configure(cfg):
    """Assert configs/ctc.yaml still transcribes SPEC's numbers.

    Called once at start-up, before any GPU work. The YAML is where a reader
    looks for the recipe; this module is where the trainer reads it from. If
    they can drift apart, the run record describes something the run did not
    do (the exact defect scoring.configure_data() was written for after a
    10-bucket sampler shipped against a 15-bucket audit).
    """
    b = cfg["batch"]
    assert int(b["microbatch"]) == MICROBATCH, "SPEC §4 microbatch 12"
    assert int(b["accumulate"]) == ACCUM, "SPEC §4 accumulate 21"
    assert int(b["effective"]) == EFFECTIVE_BATCH, "SPEC §4 effective 252"
    assert int(b["microbatch"]) * int(b["accumulate"]) == int(b["effective"]), (
        f"effective batch {b['effective']} != {b['microbatch']} x "
        f"{b['accumulate']} (SPEC §4 registers 252 = 12 x 21)")

    o = cfg["optim"]
    assert tuple(float(x) for x in o["betas"]) == OPTIM["betas"], "SPEC §4"
    for k in ("weight_decay", "grad_clip", "eps"):
        assert float(o[k]) == OPTIM[k], f"optim.{k} != registered (SPEC §4)"

    init = str(cfg.get("init", "A"))
    assert init in INITS, f"init must be one of {INITS}, got {init!r}"
    cfg["init"] = init
    s = cfg["sched"]
    registered = sched_for(init)
    for k, v in registered.items():
        got = s[k]
        assert abs(float(got) - float(v)) <= 1e-15 * max(1.0, abs(float(v))), (
            f"sched.{k}: config {got} != registered {v} for init {init} "
            f"(SPEC §4 for A, §8q/§8s for N)")
    assert int(s["horizon"]) == int(registered["horizon"]), (
        f"horizon {s['horizon']} != registered {registered['horizon']} for init {init} (A: SPEC §4 150,000; N: §8v 120,000)")
    if init == "N":
        assert "init_N" in cfg["paths"], "init N needs paths.init_N (the NVIDIA .nemo)"
        assert str(cfg["paths"].get("init_N_sha256_prefix", "")) == INIT_N_SHA256_PREFIX, (
            "paths.init_N_sha256_prefix must be the registered NVIDIA blob sha (SPEC §8s)")

    sa = cfg["specaug"]
    for k, v in SPECAUG.items():
        assert abs(float(sa[k]) - float(v)) < 1e-12, f"specaug.{k} (SPEC §4)"

    d = cfg["data"]
    assert d["concurrent_bucketing"] is False, (
        "data.concurrent_bucketing must be false: NeMo 2.7.3 defaults it True "
        "and Lhotse documents concurrent bucket filling as non-deterministic, "
        "which would make the sampler cursor in a checkpoint a lie (SPEC §4 "
        "resume semantics)")
    assert d["use_bucketing"] is True, "SPEC §4: bucketing on duration"
    assert int(d["shard_seed"]) == int(d["seed"]), (
        "shard_seed must be the fixed seed, not Lhotse's 'trng' default "
        "(SPEC §4 'shard_seed fixed')")

    v = cfg["val"]
    assert int(v["every"]) == VAL_EVERY, "SPEC §4 validation every 2,000 steps"

    c = cfg["ckpt"]
    assert int(c["keep_last"]) == KEEP_LAST and int(c["keep_best"]) == KEEP_BEST
    want_ms = MILESTONES_N if init == "N" else MILESTONES
    assert tuple(int(x) for x in c["milestones"]) == want_ms, (
        "SPEC §4 milestones: pilot end 8,000, 20k, 40k, 80k, 120k, H")

    lim = cfg["limits"]
    assert float(lim["watchdog_reserved_gib"]) == WATCHDOG_GIB, f"SPEC §8i watchdog must be {WATCHDOG_GIB} GiB"
    assert float(lim["profile_target_gib"]) == PROFILE_TARGET_GIB, "§3 G0.7"
    return cfg


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def train_surface(text):
    """SPEC §1 RULING's training-target surface, imported not re-implemented.

    metric_surface(NFC(text), score_punctuation=False) + whitespace collapse.
    Used ONLY when this module writes a smoke manifest; the real arm manifests
    get the same treatment from bnasr/build_corpus.py.
    """
    return " ".join(
        metric_surface(unicodedata.normalize("NFC", text), False).split())


def _resolve(path):
    p = Path(path)
    return str(p if p.is_absolute() else (REPO / p))


def refs_of(rows):
    return [r["text"] for r in rows]


def score_pairs(rows, hyps, per_source=True, conversational=False):
    """Score hypotheses against a panel's rows, carrying the INTEGER counts.

    Council R (SPEC 8w.9) proved with the real functions that keeping only the
    5-dp rounded WER changes registered outcomes: exact S differences of
    0.049935 pp and 0.050400 pp fall on OPPOSITE sides of the 0.05 pp tie band,
    so selection picks a different checkpoint and `retain` then deletes every
    copy of the one the rule would have chosen - and preserved hypotheses cannot
    reconstruct deleted weights. Qualification flips too: exact 22.029658047728
    FAILS where the recorded 22.0296 PASSES, an excess 5.8e-4 times the 1e-9
    representation tolerance.

    So every decision reads `counts`; the rounded fields are kept, byte for
    byte, for display and for every record already written.
    """
    assert len(hyps) == len(rows), f"{len(hyps)} hyps for {len(rows)} rows"
    we = wn = ce = cn = 0
    empties = 0
    src = {}
    conv = [0, 0]
    for row, ref, hyp in zip(rows, refs_of(rows), hyps):
        nr, nh = frozen_norm(ref), frozen_norm(hyp)
        if not nh.strip():
            empties += 1
        e, n = wer_counts(nr, nh)
        c, m = cer_counts(nr, nh)
        we, wn, ce, cn = we + e, wn + n, ce + c, cn + m
        if per_source:
            acc = src.setdefault(row.get("source", "?"), [0, 0])
            acc[0] += e
            acc[1] += n
        if conversational and is_conversational(row):
            conv[0] += e
            conv[1] += n
    res = {
        "n": len(rows),
        "wer_frozen": round(we / max(wn, 1), 5),
        "cer_frozen": round(ce / max(cn, 1), 5),
        "empty_rate": round(empties / max(len(rows), 1), 5),
        "empties": empties,
        "counts": {"w_edits": int(we), "w_ref": int(wn),
                   "c_edits": int(ce), "c_ref": int(cn),
                   "empties": int(empties), "n": int(len(rows))},
    }
    if per_source:
        res["per_source"] = {k: round(a[0] / max(a[1], 1), 5)
                             for k, a in sorted(src.items())}
        res["per_source_counts"] = {k: [int(a[0]), int(a[1])]
                                    for k, a in sorted(src.items())}
    if conversational:
        res["conversational"] = ({"n_ref_words": conv[1], "w_edits": int(conv[0]),
                                  "wer_frozen": round(conv[0] / max(conv[1], 1), 5)}
                                 if conv[1] else None)
    return res


class PanelIntegrityError(ValueError):
    """A registered panel or source cannot be scored authoritatively.

    Council R: a panel or source with ZERO total reference words must block
    authoritative evaluation, not silently become a 0 % WER pass. This is
    distinct from scoring.wer_counts' deliberate per-ROW tolerance, which
    stays: one empty reference row can never take down a 120k-update run.
    """


def _counts(panel):
    c = (panel or {}).get("counts")
    return c if isinstance(c, dict) else None


def has_exact_counts(panel):
    """True when this record carries the integers a decision needs.

    False for records written before SPEC 8w.9 - the four this run wrote at
    updates 2,000-8,000 - which is not an error: the accessors fall back to the
    recorded value, so those records decide exactly as they always did. Their
    exactness was verified independently (experiments/unrounded_recon.json:
    largest S discrepancy 1.39e-04 pp).
    """
    c = _counts(panel)
    return bool(c and "w_edits" in c and "w_ref" in c)


def exact_wer(panel):
    """Frozen-surface WER as a FRACTION for DECISIONS - never for display."""
    if panel is None:
        return None
    c = _counts(panel)
    if c and "w_edits" in c and "w_ref" in c:
        e, n = c["w_edits"], c["w_ref"]
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise PanelIntegrityError(
                f"panel has {n!r} total reference words: it cannot be scored")
        return float(e) / float(n)
    w = panel.get("wer_frozen")
    return None if w is None else float(w)


def exact_per_source(panel):
    """{source: WER fraction} for DECISIONS, exact where counts exist."""
    if panel is None:
        return {}
    c = (panel or {}).get("per_source_counts")
    if isinstance(c, dict) and c:
        out = {}
        for k, v in c.items():
            e, n = (list(v) + [0, 0])[:2]
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise PanelIntegrityError(
                    f"source {k!r} has {n!r} total reference words: it cannot be scored")
            out[k] = float(e) / float(n)
        return out
    return {k: float(v) for k, v in (panel.get("per_source") or {}).items()}


def exact_conversational(panel):
    """Conversational-slice WER as a fraction for DECISIONS, or None."""
    conv = (panel or {}).get("conversational")
    if not conv:
        return None
    e, n = conv.get("w_edits"), conv.get("n_ref_words")
    if isinstance(e, int) and isinstance(n, int) and not isinstance(n, bool):
        if n <= 0:
            raise PanelIntegrityError(
                "conversational slice has 0 reference words: it cannot be scored")
        return float(e) / float(n)
    w = conv.get("wer_frozen")
    return None if w is None else float(w)


def exact_empty_rate(panel):
    """Empty-output rate as a fraction for DECISIONS."""
    if panel is None:
        return None
    c = _counts(panel)
    if c and "empties" in c and "n" in c and c.get("n"):
        return float(c["empties"]) / float(c["n"])
    er = panel.get("empty_rate")
    return None if er is None else float(er)


def selection_score_exact(panels, madasr_admitted_flag=None):
    """S in pp for DECISIONS, from the integer counts where they exist.

    Same weights and same missing-panel semantics as selection_score; the only
    difference is that each panel's WER comes through exact_wer().
    """
    adm = madasr_admitted() if madasr_admitted_flag is None else madasr_admitted_flag
    w = S_WEIGHTS_WITH_MADASR if adm else S_WEIGHTS_NO_MADASR
    wers = {}
    for k in w:
        wers[k] = exact_wer((panels or {}).get(k))
    missing = [k for k in w if wers.get(k) is None]
    if missing:
        return None, {"weights": w, "missing": missing, "exact": True}
    s = sum(w[k] * wers[k] * 100.0 for k in w)
    return s, {"weights": w, "missing": [], "exact": True,
               "from_counts": all(has_exact_counts((panels or {}).get(k)) for k in w)}


def ctc_panel_eval(model, manifest_path, batch_size=32, limit=None,
                   per_source=True, conversational=False):
    """Greedy CTC decode of one panel, scored on BOTH registered surfaces.

    Returns {n, wer_frozen, cer_frozen, wer_nemo, empty_rate, per_source,
    conversational, wall_s, hyps}.

    Both surfaces, always (SPEC §4 "frozen surface + NeMo-internal"):
      * wer_frozen - bnasr.scoring.frozen_norm + wer_counts, the surface the
        20.35 / 17.03 / 22.30 baselines live on. Selection uses this one.
      * wer_nemo   - NeMo's own word_error_rate on RAW text, the surface the
        the baseline val_wer curve (20.83 on D) was logged on.
    Reporting one without the other is how a comparison quietly changes its
    yardstick mid-phase, so neither is optional here.

    `empty_rate` is a registered guard (SPEC §2, <= 0.1 %; SPEC §5 aborts a
    pilot arm above 1 %), and it is counted on the FROZEN surface: a hypothesis
    of nothing but punctuation is an empty output for scoring purposes.

    The model is restored to whatever training/eval mode it arrived in - the
    trainer calls this mid-run, and a model left in eval() would silently
    disable dropout and SpecAugment for the next 2,000 updates.
    """
    import torch
    rows = load_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    paths = [_resolve(r["audio_filepath"]) for r in rows]

    was_training = model.training
    t0 = time.time()
    try:
        model.eval()
        with torch.inference_mode():
            out = model.transcribe(paths, batch_size=batch_size,
                                   num_workers=0, verbose=False)
    finally:
        if was_training:
            model.train()
    wall = time.time() - t0
    if isinstance(out, tuple):
        out = out[0]
    hyps = [h.text if hasattr(h, "text") else str(h) for h in out]
    assert len(hyps) == len(rows), f"{len(hyps)} hyps for {len(rows)} rows"

    res = score_pairs(rows, hyps, per_source=per_source,
                      conversational=conversational)
    res["wer_nemo"] = round(nemo_wer(refs_of(rows), hyps), 5)
    res["wall_s"] = round(wall, 1)
    res["hyps"] = hyps
    return res


def madasr_admitted(cfg=None):
    """Was MADASR admitted, per the REGISTERED decision - not per what is on disk.

    Council Q #6: inferring this from `"R" in res` let a vanished panel file
    silently switch S's weight vector and manufacture an improvement. SPEC §8m
    records the decision (madasr ADMITTED 2026-09-08, delta -0.994, sim -0.702),
    so it is a registered constant here; a config may state it explicitly but
    may not contradict the registered value.
    """
    registered = True
    if cfg is not None and "madasr_admitted" in cfg:
        stated = bool(cfg["madasr_admitted"])
        if stated != registered:
            raise ValueError(
                f"config states madasr_admitted={stated}, but the registered "
                f"decision (SPEC §8m) is {registered}; amend the contract, not the config")
    return registered


def selection_score(panel_wers, madasr_admitted):
    """S = 0.60 D + 0.20 M + 0.10 R + 0.10 K, or 0.60 D + 0.30 M + 0.10 K.

    SPEC §2 defines S on frozen-surface CORPUS WER in percentage points, and
    which of the two weightings is in force depends on whether MADASR was
    admitted (panel R exists). Returns (S_in_pp, weights_used) or (None, ...)
    when a weighted panel is missing - a missing panel makes S UNDEFINED, and
    silently renormalising the weights over the panels that happen to exist
    would produce a number that is not the registered S and cannot be compared
    against S(P1) or across arms.
    """
    w = S_WEIGHTS_WITH_MADASR if madasr_admitted else S_WEIGHTS_NO_MADASR
    missing = [k for k in w if panel_wers.get(k) is None]
    if missing:
        return None, {"weights": w, "missing": missing}
    s = sum(w[k] * float(panel_wers[k]) * 100.0 for k in w)
    return round(s, 4), {"weights": w, "missing": []}


def guards(current, baseline):
    """SPEC §2 regression guards, evaluated against MEASURED the baselines.

    current/baseline are the dicts ctc_panel_eval() returns for panel D.
    Returns {pass, checks:[...]} with every check in percentage points, and
    `pass` False if any registered guard is breached. A missing baseline is
    reported as `evaluable: false` rather than silently passing: SPEC §2's
    guards are comparisons, and a comparison with nothing is not a pass.
    """
    try:
        return _guards_inner(current, baseline)
    except PanelIntegrityError as exc:
        # A registered panel or source with zero reference words cannot be
        # scored; that is a failed, non-evaluable guard - never a silent pass.
        return {"evaluable": False, "pass": False, "reason": str(exc), "checks": []}


def _guards_inner(current, baseline):
    out = {"evaluable": baseline is not None, "checks": []}
    if baseline is None:
        out["pass"] = False
        out["reason"] = ("no the baseline (run train.py "
                         "--restore-check first; it writes baseline_p1.json)")
        return out
    ok = True
    # Council R (SPEC 8w.9): this used to iterate the sources PRESENT in
    # `current`, so a metrics dict that simply omitted four of the five D
    # sources passed the guards on the strength of the one that remained.
    # The REGISTERED set is the baseline's; a registered source missing from
    # the current record is a failure, not an absence.
    cur_src, base_src = exact_per_source(current), (baseline.get("per_source") or {})
    for s in sorted(set(base_src) | set(cur_src)):
        b, w = base_src.get(s), cur_src.get(s)
        if b is None:
            out["checks"].append({"check": f"D/{s}", "evaluable": False,
                                  "reason": "source not in the registered baseline"})
            ok = False
            continue
        if w is None:
            out["checks"].append({"check": f"D/{s}", "evaluable": False,
                                  "reason": "registered source missing from this evaluation"})
            ok = False
            continue
        delta = (float(w) - float(b)) * 100.0
        passed = delta <= GUARD_SOURCE_PP
        ok = ok and passed
        out["checks"].append({"check": f"D/{s}", "delta_pp": round(delta, 3),
                              "limit_pp": GUARD_SOURCE_PP, "pass": passed})
    cur_c = exact_conversational(current)
    base_c = (baseline.get("conversational") or {}).get("wer_frozen")
    if cur_c is None or base_c is None:
        out["checks"].append({"check": "D/conversational", "evaluable": False})
        ok = False
    else:
        delta = (float(cur_c) - float(base_c)) * 100.0
        passed = delta <= GUARD_CONVERSATIONAL_PP
        ok = ok and passed
        out["checks"].append({"check": "D/conversational",
                              "delta_pp": round(delta, 3),
                              "limit_pp": GUARD_CONVERSATIONAL_PP,
                              "pass": passed})
    er = exact_empty_rate(current)
    er = 1.0 if er is None else float(er)
    passed = er <= GUARD_EMPTY_RATE
    ok = ok and passed
    out["checks"].append({"check": "empty_rate", "value": er,
                          "limit": GUARD_EMPTY_RATE, "pass": passed})
    out["pass"] = ok
    return out


def load_baseline():
    """The measured the baseline panel baselines, or None if --restore-check has
    not been run yet (never a fabricated default)."""
    if not BASELINE_JSON.exists():
        return None
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


def phase1_val_curve():
    """the baseline's TRUE per-validation curve: {update: internal WER pct}.

    Read from experiments/phase1_val_curve.json, which bnasr/baseline_curve.py
    extracts from the TensorBoard event protobufs.

    NEVER rebuild this from lightning_logs.txt.  Those "New best score" lines are
    EarlyStopping improvement events, not a 1,000-update grid; enumerating them and
    labelling event i as update (i+1)*1000 yields the running-minimum envelope on a
    wrong x-axis, understating the true update by up to 16,005 (84%).  That defect
    reached councils I, J and O (SPEC 8w).  Callers that want the envelope must build
    it from this curve and label it honestly.
    """
    if not PHASE1_CURVE_JSON.exists():
        return None
    doc = json.loads(PHASE1_CURVE_JSON.read_text(encoding="utf-8"))
    return {r["update"]: r["val_wer_pct"] for r in doc["curve"]}


def assert_frozen_p1_baseline(baseline):
    """Refuse any baseline that is not the FROZEN the baseline record (council F Q2).

    SPEC §8k Q2 is explicit that continuation and final-selection guards are
    evaluated against the frozen baseline and NOT against O_ref:
    "Admission against O_ref does NOT certify continuation eligibility."
    O_ref and P1 are only 0.043 pp apart on S, so a file swapped by mistake
    would not look wrong in a log - it would just quietly move every guard by
    the amount of one arm's training noise, in the direction that makes a
    regression easier to pass. The one thing that distinguishes them is
    provenance: baseline_p1.json is measured at ZERO updates from init A, so
    it carries init A's sha256 (SPEC §4, "sha256 61362a36..."), and a control
    arm's evaluation record does not.

    Returns the baseline unchanged; raises ValueError with the offending value
    so the caller can turn it into a recorded ABORT rather than a traceback.
    """
    if baseline is None:
        raise ValueError("no the baseline: run train.py --restore-check "
                         "--write-baseline first (SPEC §4)")
    sha = str(baseline.get("init_sha256", ""))
    if not sha.startswith(INIT_SHA256_PREFIX):
        raise ValueError(
            f"{BASELINE_JSON.name} was not measured from init A: init_sha256 "
            f"{sha[:16]!r} does not start with {INIT_SHA256_PREFIX!r}. The "
            f"continuation guards must use the FROZEN the baseline, not a "
            f"control arm (SPEC §8k council F Q2)")
    d = baseline.get("D") or {}
    got = float(d.get("wer_frozen", -1.0))
    if abs(got - P1_DEV_FROZEN) > RESTORATION_TOL:
        raise ValueError(
            f"baseline D wer_frozen {got} is not the baseline's {P1_DEV_FROZEN} "
            f"+/- {RESTORATION_TOL} (SPEC §4 restoration check)")
    return baseline


def conversational_limit_pp(baseline):
    """The absolute conversational ceiling council F fixed at 22.075 % (§8k Q2).

    P1's conversational slice is 21.575 % (baseline_p1.json, exact) and the
    guard is +0.5 pp, so the limit is 22.075 %. It is computed rather than
    hard-coded - the guard has to move if the baseline is ever re-measured -
    and the caller asserts the computed value against the ruling.
    """
    c = ((baseline or {}).get("D") or {}).get("conversational") or {}
    w = c.get("wer_frozen")
    if w is None:
        return None
    return round(float(w) * 100.0 + GUARD_CONVERSATIONAL_PP, 4)


def is_eligible(val):
    """Is this validation's checkpoint ELIGIBLE for selection (council F Q2)?

    SPEC §8k Q2 makes checkpoint INELIGIBILITY the whole protection against a
    slice regressing mid-run ("no abort rule added ... checkpoint
    INELIGIBILITY is the protection"), on two conditions that this predicate
    exists to make enforceable: (i) ineligible checkpoints never reset plateau
    patience, and (ii) the selected eligible checkpoint is preserved
    independently of rotation.

    Eligible means every registered guard passed against the frozen P1
    baseline: every D source within +1.0 pp, the conversational slice within
    +0.5 pp, empties <= 0.1 %. A validation whose guards could not be
    evaluated (no baseline, a missing panel) is NOT eligible - `guards()`
    already reports that as pass False, and a checkpoint whose guards are
    unknown must not become the final model.
    """
    return bool((val or {}).get("guards", {}).get("pass"))


def select_best(history):
    """SPEC §4: "Final checkpoint = lowest eligible S; ties within 0.05 pp ->
    earlier."

    `history` is the list of (S_pp, update, eligible) triples the trainer
    accumulates, one per validation. Returns the winning (S_pp, update) or
    None when nothing is eligible yet.

    The tie band is not a sort key. Sorting by (S, update) picks the strictly
    lowest S, which for S 20.47 at 80,000 and S 20.50 at 60,000 returns the
    LATER checkpoint - the two are 0.03 pp apart, inside the registered 0.05 pp
    band, so SPEC says the earlier one wins. The band is applied as an explicit
    sweep from the best S: every checkpoint within TIE_BAND_PP of it is tied
    with it, and the earliest of those is the selection. That is also why the
    band is not transitive-chained: the comparison is always against the best
    S, never against the current candidate, so a long shallow chain of 0.04 pp
    steps cannot walk the selection arbitrarily far back.
    """
    elig = [(float(s), int(u)) for s, u, e in history if e and s is not None]
    if not elig:
        return None
    best_s = min(s for s, _ in elig)
    tied = [(s, u) for s, u in elig if s <= best_s + TIE_BAND_PP]
    u = min(u for _, u in tied)
    return next((s, uu) for s, uu in tied if uu == u)


def normalise_history(raw):
    """Read a `best_hist` out of a checkpoint written before eligibility existed.

    Entries used to be (S_pp, update) pairs, with no record of whether the
    guards passed at that validation. Council F Q2 forbids selecting a
    checkpoint whose guards are unknown, so a legacy pair is restored as
    NOT eligible rather than assumed good; the next validation after a resume
    writes a full triple anyway.
    """
    out, legacy = [], 0
    for e in raw or []:
        t = tuple(e)
        if len(t) == 3:
            out.append((float(t[0]), int(t[1]), bool(t[2])))
        elif len(t) == 2:
            out.append((float(t[0]), int(t[1]), False))
            legacy += 1
    return out, legacy


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def atomic_save(obj, path):
    """torch.save to <path>.tmp, fsync, then os.replace (SPEC §4 "atomic").

    A 1.8 GB checkpoint written in place is a 10-second window in which a
    watchdog kill, an OOM or a power event leaves a TRUNCATED file where the
    only resume point should be. os.replace() is atomic within a filesystem,
    so the reader either sees the previous checkpoint or the complete new one
    and never a half-written one.
    """
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
