#!/usr/bin/env python
"""Shared surface for Phase 3 (Parakeet-TDT-0.6B -> Bengali TDT).

Everything that more than one Phase-3 tool needs lives here so there is
exactly ONE copy of the scoring surface, the LR schedules, the GPU pin and
the process-anonymisation call. `experiments/SPEC.md` is the contract;
the constants below are transcriptions of it, and `configure_schedules()`
asserts that `configs/ctc.yaml` agrees with them, so a config edit can
never silently move a registered number.

Two scorers are carried side by side on purpose (SPEC section 4):

  * `frozen_norm` + `wer_counts` - the FROZEN benchmark surface, byte for
    byte the same as `bengali-asr-benchmark/bench_score.norm`. This is what
    the published the baseline/Phase-2 numbers and the final FLEURS claim use, so
    Phase 3 must be measured on it or the comparison is not a comparison.
  * `nemo_wer` - NeMo's own `word_error_rate` on RAW text, no normalisation.
    This is the surface the the baseline `val_wer` curve was logged on, and Gate 3
    compares Phase-3 dev WER against that curve (SPEC section 5), so it has
    to be reproduced exactly rather than approximated.

They disagree by roughly a point on Bengali (punctuation and the U+09DF /
U+09AF+U+09BC spelling split), which is precisely why both are logged.
"""
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The A5000. The A6000 is a co-tenant's card and is off-limits for the whole
# of Phase 3 (SPEC section 0); pin_gpu() turns that rule into an assertion.
A5000_UUID = "GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567"
A6000_UUID = "GPU-4754ca84-fd1d-907b-5c18-7c578d001bda"   # owner-permitted for this recipe pilots only (P4_ALLOW_A6000=1)

# --------------------------------------------------------------------------
# Registered LR schedules (SPEC section 3). Held here rather than only in the
# YAML because lr_at() is the thing the optimiser actually reads; the YAML is
# checked against these by configure_schedules().
# --------------------------------------------------------------------------
# Registered optimiser constants (SPEC section 3).
OPTIM = {
    "betas": (0.9, 0.98),
    "eps": 1.0e-8,
    "weight_decay": 1.0e-3,      # 0 for biases and norm scales/shifts
    "grad_clip": 1.0,
    "accum_min_utterances": 16,
}

# HORIZON. SPEC section 3 registers H = 60,000 updates ("owner-authorised
# increase from council's 40,000, registered before any run starts"); SPEC
# section 5's "continue winner to 40,000" is the stale pre-amendment sentence
# and section 3 is the later registered value, so 60,000 is what is coded.
# This matters far beyond the stopping point: BOTH cosine legs decay TO the
# horizon, so running a 60k schedule against a 40,000 denominator would leave
# the last 20k updates pinned flat at the LR floor instead of on the
# registered cosine. `--max-updates` moves the stopping point and NOT the
# denominator, which is exactly how that would have gone unnoticed.
#
# SPEC section 3 also requires H to be FIXED AT GATE 3 from the measured rate
# (H = min(60,000, the largest horizon that fits the remaining GPU-h cap)) and
# never extended afterwards. That is why the horizon is a run-time value
# (`SCHED["horizon"]`, set from `sched.horizon_updates` by
# configure_schedules()) rather than a module constant: it can be lowered once
# at Gate 3, is recorded in run_meta.json and in every checkpoint, and a resume
# whose recorded horizon disagrees with the current config is refused.
HORIZON_MAX = 60000

SCHED = {
    "total_updates": HORIZON_MAX,
    # The horizon in force for THIS process; configure_schedules() overwrites
    # it from the config and every caller of lr_at() reads it through there.
    "horizon": HORIZON_MAX,
    # HEAD (decoder.* + joint.*): linear 3e-5 -> 3e-4 over updates 0-500,
    # then cosine 3e-4 -> 3e-5 over 500-H.
    "head": {"lr_start": 3.0e-5, "lr_peak": 3.0e-4, "lr_final": 3.0e-5,
             "warmup_start": 0, "warmup_end": 500},
    # ENCODER: 0 over updates 0-1,000 (it is frozen there anyway), then
    # linear 1e-6 -> 1e-5 over 1,000-2,000, then cosine 1e-5 -> 1e-6 to H.
    "encoder": {"lr_start": 1.0e-6, "lr_peak": 1.0e-4, "lr_final": 1.0e-6,
                "warmup_start": 0, "warmup_end": 1000},
}

# Registered Lhotse sampler plan (SPEC section 2). Held here, not only in the
# YAML, because a sampler that is not this one trains on a different microbatch
# stream from the one the Gate-1 audit certified - and the 10-bucket the baseline
# variant that SPEC explicitly rejects is one careless copy-paste away.
DATA = {
    "batch_duration": 100.0,
    "batch_size": 8,
    "quadratic_duration": 20.0,
    "use_bucketing": True,
    "num_buckets": 15,
    "bucket_duration_bins": [2.27, 2.9, 3.5, 4.2, 5.18, 6.23, 7.65, 9.69,
                             12.58, 15, 18, 21, 24, 27],
    "bucket_buffer_size": 20000,
    "shuffle_buffer_size": 10000,
    "min_duration": 0.1,
    "max_duration": 30.0,
    "max_tps": 12.0,
    "drop_last": False,
    "skip_missing_manifest_entries": False,
    # Not in SPEC's key list and it must be: NeMo 2.7.3 defaults this to True,
    # which Lhotse documents as non-deterministic sampling.
    "concurrent_bucketing": False,
}

# The ONE registered init set (SPEC section 1.3: v2 and v3 must start from a
# byte-identical randomly-initialised decoder+joint, so that the arms differ
# only in the encoder). Any other copy on disk is unregistered by definition.
INITS_REGISTRY = REPO / "experiments/inits/INITS.json"

# Frozen scoring surface. Copied verbatim from bench_score.py: punctuation
# becomes a word boundary, whitespace collapses, then NFC. NFC is load-bearing
# for Bengali - য়/ড়/ঢ় have a precomposed and a decomposed spelling that are
# canonically equivalent but different byte strings, and without folding them
# a model is charged a substitution for an encoding convention.
_PUNCT = re.compile(r"[\"“”‘’'।,.!?;:()\[\]{}—–\-]")


def frozen_norm(text):
    """The frozen benchmark scoring surface. Identical to bench_score.norm."""
    return unicodedata.normalize(
        "NFC", re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip())


def load_manifest(path):
    """NeMo JSON-lines manifest -> list of dicts, in file order.

    Order matters: dev_eval() zips hypotheses back onto rows positionally,
    and the locked dev panel's row order is part of what makes the the baseline /
    Phase-2 / Phase-3 numbers comparable.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wer_counts(ref, hyp):
    """(word errors, reference word count) for one pair, jiwer on whitespace.

    An empty hypothesis is scored as the single token "*" exactly as
    bench_score does: jiwer refuses an empty hypothesis, and silently
    dropping the row would flatter a model that has collapsed to silence.
    An empty REFERENCE cannot occur in these manifests, but is handled
    (every hypothesis word is an insertion, denominator 0) rather than
    raising, so one bad row can never take down a 40k-update run.
    """
    import jiwer
    if not ref.strip():
        return len(hyp.split()), 0
    o = jiwer.process_words([ref], [hyp if hyp.strip() else "*"])
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def cer_counts(ref, hyp):
    """(character errors, reference character count), same conventions."""
    import jiwer
    if not ref.strip():
        return len(hyp), 0
    o = jiwer.process_characters([ref], [hyp if hyp.strip() else "*"])
    return o.substitutions + o.deletions + o.insertions, len(ref)


def nemo_wer(refs, hyps):
    """NeMo's own corpus WER on RAW text - no normalisation, on purpose.

    Parity with the the baseline `val_wer` curve (p3_probe/phase1_val_curve.json),
    which Gate 3 compares against. Normalising here would make the Phase-3
    AUC incomparable with the registered the baseline reference AUC of 44.70.
    """
    from nemo.collections.asr.metrics.wer import word_error_rate
    return float(word_error_rate(hypotheses=list(hyps), references=list(refs)))


def _resolve(path):
    """Manifest audio paths are repo-relative; make them absolute."""
    p = Path(path)
    return str(p if p.is_absolute() else (REPO / p))


def dev_eval(model, manifest_path, batch_size=32, limit=None):
    """Greedy-batch decode of a manifest, scored on BOTH surfaces.

    Returns {n, wer_frozen, cer_frozen, wer_nemo, per_source, wall_s, hyps}.

    The model's train/eval mode and its fused-loss flag are saved and put
    back: the trainer calls this mid-run, and leaving the joint in fused
    mode or the model in eval mode afterwards would corrupt the next update
    in a way that only shows up thousands of steps later.
    """
    import torch
    rows = load_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    paths = [_resolve(r["audio_filepath"]) for r in rows]

    was_training = model.training
    joint = model.joint
    was_fused = joint.fuse_loss_wer
    fused_bs = joint.fused_batch_size
    # SPEC section 1 pops train_ds/validation_ds/test_ds when the init is
    # built, but NeMo's transcribe() reaches into cfg.validation_ds for a
    # single flag (`use_start_end_token`) while assembling its temporary
    # dataloader, and an OmegaConf struct raises rather than returning a
    # default. Put a minimal node in for the duration of the decode and take
    # it straight back out, so nothing that gets checkpointed or saved
    # afterwards carries a dataset section the SPEC says must not be there.
    from omegaconf import OmegaConf, open_dict
    had_val_ds = "validation_ds" in model.cfg
    if not had_val_ds:
        with open_dict(model.cfg):
            model.cfg.validation_ds = OmegaConf.create(
                {"use_start_end_token": False})
    t0 = time.time()
    try:
        # Greedy decoding calls joint.joint() directly, but the fused path
        # changes joint.forward()'s contract; switch it off so nothing in
        # transcribe() can trip over a missing `transcripts` argument.
        joint.set_fuse_loss_wer(False)
        model.eval()
        with torch.inference_mode():
            out = model.transcribe(paths, batch_size=batch_size,
                                   num_workers=0, verbose=False)
    finally:
        joint.set_fuse_loss_wer(was_fused, model.loss, model.wer)
        joint.set_fused_batch_size(fused_bs)
        if not had_val_ds:
            with open_dict(model.cfg):
                model.cfg.pop("validation_ds", None)
        if was_training:
            model.train()
    wall = time.time() - t0

    # NeMo 2.x returns Hypothesis objects (and, for some configs, a
    # (best, all_beams) tuple); reduce whatever comes back to plain text.
    if isinstance(out, tuple):
        out = out[0]
    hyps = [h.text if hasattr(h, "text") else str(h) for h in out]
    assert len(hyps) == len(rows), f"{len(hyps)} hyps for {len(rows)} rows"

    refs = [r["text"] for r in rows]
    we = wn = ce = cn = 0
    per_src = {}
    for row, ref, hyp in zip(rows, refs, hyps):
        nr, nh = frozen_norm(ref), frozen_norm(hyp)
        e, n = wer_counts(nr, nh)
        c, m = cer_counts(nr, nh)
        we, wn, ce, cn = we + e, wn + n, ce + c, cn + m
        s = row.get("source", "?")
        acc = per_src.setdefault(s, [0, 0])
        acc[0] += e
        acc[1] += n
    return {
        "n": len(rows),
        "wer_frozen": round(we / max(wn, 1), 5),
        "cer_frozen": round(ce / max(cn, 1), 5),
        "wer_nemo": round(nemo_wer(refs, hyps), 5),
        "per_source": {s: round(a[0] / max(a[1], 1), 5)
                       for s, a in sorted(per_src.items())},
        "wall_s": round(wall, 1),
        "hyps": hyps,
    }


def sha256_file(path, chunk=1 << 20):
    """Streamed SHA-256; the .nemo inits are 2.4 GB each."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def pin_gpu():
    """Hard-assert that we are on the A5000 and nothing else.

    The box is shared: the A6000 belongs to a co-tenant and Phase 3 is
    budgeted in A5000 hours (SPEC section 0). A misconfigured
    CUDA_VISIBLE_DEVICES must stop the run at second zero rather than
    quietly spend somebody else's card for 28 hours.
    """
    got = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # this recipe amendment (owner, 2026-09-06): pilot/ablation and Gate-0 work MAY use the A6000 under a 20 GB cap.
    # The exception must be requested explicitly per process (P4_ALLOW_A6000=1) so a stray launch still fails at second zero.
    allowed = {A5000_UUID}
    if os.environ.get("P4_ALLOW_A6000") == "1":
        allowed.add(A6000_UUID)
    assert got in allowed, (
        f"CUDA_VISIBLE_DEVICES must be one of {sorted(allowed)!r}, got {got!r}")
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() == 1, (
        f"expected exactly 1 visible GPU, saw {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    ok_names = ("A5000",) + (("A6000",) if os.environ.get("P4_ALLOW_A6000") == "1" else ())
    assert any(n in name for n in ok_names), f"pinned device is {name!r}, not one of {ok_names}"
    return name


def anonymize_process(title=None):
    """Retitle the process to noise so `ps`/`nvidia-smi` leak nothing.

    Same mechanism and same string as the venv's sitecustomize.py and
    train.py: on a shared box /proc/<pid>/cmdline is world-readable and
    would otherwise advertise the repo path and the script name. Duplicated
    here (rather than relying on sitecustomize alone) so that a venv rebuild
    cannot silently de-anonymise a Phase-3 run. Fails silent by design.
    """
    try:
        import setproctitle
        setproctitle.setproctitle(title or os.environ.get("PROC_TITLE",
                                                          "6mn lmu"))
        return True
    except Exception:
        return False


class HourlyLogger:
    """Append-only hourly summary at <run_dir>/hourly.jsonl (SPEC section 4).

    `due()` is separate from `log()` so the caller decides what to summarise;
    the file is opened and closed per line and fsync-free, which is what makes
    it safe to `tail -f` from another shell while a 100-hour run is going.
    """

    def __init__(self, run_dir, interval_s=3600.0):
        self.path = Path(run_dir) / "hourly.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.last = time.time()

    def due(self, now=None):
        return (now or time.time()) - self.last >= self.interval_s

    def log(self, **fields):
        fields.setdefault("wall_utc", time.strftime("%Y-%m-%dT%H:%M:%S",
                                                    time.gmtime()))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self.last = time.time()
        return fields


def _no_decay(name, param):
    """Biases and every norm scale/shift get weight_decay 0.

    Decaying a LayerNorm gain or a BatchNorm scale pulls the normalisation
    towards zero-gain and is a well-known way to make a transducer's encoder
    quietly worse; SPEC section 3 registers wd=0 for them.
    """
    return param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()


def make_param_groups(model):
    """Two logical groups - ENCODER and HEAD - split by weight-decay policy.

    SPEC section 3 registers exactly two LR groups (encoder.* and
    decoder.*+joint.*) AND weight_decay 0 for biases/norms. A torch param
    group carries a single `weight_decay`, so the two rules force four
    dicts. Each keeps `group` ('encoder' | 'head') - which is what lr_at()
    reads - alongside a unique `name`, so the two-group schedule and the
    per-tensor decay policy stay independent and both are inspectable in
    the step log.
    """
    buckets = {("encoder", True): [], ("encoder", False): [],
               ("head", True): [], ("head", False): []}
    unknown = []
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            g = "encoder"
        elif name.startswith("decoder.") or name.startswith("joint."):
            g = "head"
        else:
            unknown.append(name)
            continue
        buckets[(g, _no_decay(name, p))].append(p)
    # The preprocessor is a fixed mel filterbank with no parameters; anything
    # else here means the model is not the one SPEC section 1 built.
    assert not unknown, f"unexpected parameter groups: {unknown[:5]}"

    groups = []
    for (g, nodecay), params in buckets.items():
        if not params:
            continue
        groups.append({
            "name": g if not nodecay else f"{g}_nodecay",
            "group": g,
            "params": params,
            "weight_decay": 0.0 if nodecay else OPTIM["weight_decay"],
            "lr": lr_at(0, g),
        })
    groups.sort(key=lambda d: (d["group"] != "encoder", d["name"]))
    return groups


def _cosine(step, s0, s1, hi, lo):
    """Half-cosine from `hi` at s0 down to `lo` at s1, clamped past s1."""
    import math
    if step <= s0:
        return hi
    if step >= s1:
        return lo
    t = (step - s0) / (s1 - s0)
    return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * t))


def lr_at(step, group, horizon=None):
    """LR for `group` at update `step`, exactly as SPEC section 3 registers it.

    `step` is the 0-based index of the update about to be applied, so
    update 1 of the run is step 0. That convention makes the freeze phase
    "updates 1..1,000" the steps 0..999, and it is the convention the
    registered checkpoints (0, 500, 1,000, 2,000, H) are quoted at.

    HEAD    : linear 3e-5 -> 3e-4 over 0..500, cosine 3e-4 -> 3e-5 to H.
    ENCODER : 0 over 0..1,000, linear 1e-6 -> 1e-5 over 1,000..2,000,
              cosine 1e-5 -> 1e-6 to H. It is frozen while the LR is 0, so the
              two mechanisms agree instead of merely coexisting.

    `horizon` is H. It is an explicit argument rather than a module constant
    read behind the caller's back, because it is the one registered number in
    this file that SPEC section 3 allows to be FIXED at Gate 3 rather than at
    authoring time - and a schedule whose denominator can move silently is the
    exact defect this signature exists to prevent. Omitted, it falls back to
    the horizon configure_schedules() put in force for this process.
    """
    name = group["group"] if isinstance(group, dict) else str(group)
    name = "encoder" if name.startswith("encoder") else "head"
    s = SCHED[name]
    h = int(SCHED["horizon"] if horizon is None else horizon)
    w0, w1 = s["warmup_start"], s["warmup_end"]
    if step < w0:                       # encoder only: the frozen phase
        return 0.0
    if step <= w1:                      # linear ramp
        frac = (step - w0) / (w1 - w0)
        return s["lr_start"] + (s["lr_peak"] - s["lr_start"]) * frac
    return _cosine(step, w1, h, s["lr_peak"], s["lr_final"])


def configure_schedules(cfg_sched):
    """Assert configs/ctc.yaml still matches the registered schedules.

    The YAML is the place a reader looks for the numbers; SCHED is the place
    the optimiser reads them from. Rather than let those drift, the trainer
    calls this once at start-up and dies if they disagree - so editing the
    YAML cannot change the schedule without also changing this file, which
    is a code review, which is the point.
    """
    assert int(cfg_sched["total_updates"]) == SCHED["total_updates"], (
        f"total_updates: config {cfg_sched['total_updates']} != "
        f"registered {SCHED['total_updates']} (SPEC section 3, H = 60,000)")
    # SPEC section 3: H may be LOWERED once, at Gate 3, to whatever fits the
    # remaining 168 GPU-h cap - never raised, and never moved after Gate 3.
    horizon = int(cfg_sched.get("horizon_updates", SCHED["total_updates"]))
    assert 0 < horizon <= SCHED["total_updates"], (
        f"sched.horizon_updates {horizon} must be in (0, "
        f"{SCHED['total_updates']}]")
    SCHED["horizon"] = horizon
    for g in ("head", "encoder"):
        for k, v in SCHED[g].items():
            got = cfg_sched[g][k]
            assert abs(float(got) - float(v)) <= 1e-12 * max(1.0, abs(v)), (
                f"sched.{g}.{k}: config {got} != registered {v}")
    return SCHED


def configure_optim(cfg_optim):
    """Same guard for the AdamW constants (SPEC section 3)."""
    assert tuple(float(x) for x in cfg_optim["betas"]) == OPTIM["betas"]
    for k in ("eps", "weight_decay", "grad_clip"):
        assert float(cfg_optim[k]) == OPTIM[k], (
            f"optim.{k}: config {cfg_optim[k]} != registered {OPTIM[k]}")
    assert int(cfg_optim["accum_min_utterances"]) == \
        OPTIM["accum_min_utterances"]
    return OPTIM


def configure_data(cfg_train_ds):
    """Assert the sampler plan is the REGISTERED one before any data is drawn.

    This is not belt-and-braces. The 10-bucket / 9-edge list from the baseline was
    shipped in this config once already, and SPEC section 2 rejects it by name:
    its top bucket spans 12.6-30 s and Lhotse pads every member of a microbatch
    up to the longest, so the audited plan contained a 4-clip microbatch of
    114.7 PADDED audio-seconds against a 100 s budget. A run launched from that
    config trains on a different, memory-unsafe sampler from the one Gate 1
    certified, which invalidates both the envelope evidence and the v2/v3 arm
    comparison. `concurrent_bucketing` is asserted for the same reason: NeMo
    defaults it True, and a silent default is what caused the divergence.
    """
    missing = object()
    for key, want in DATA.items():
        got = cfg_train_ds.get(key, missing)
        if got is missing:
            raise AssertionError(
                f"train_ds.{key} is missing; it must be set explicitly to "
                f"{want!r} (SPEC section 2). A silent framework default is "
                f"exactly the failure this guard exists to catch.")
        if isinstance(want, list):
            ok = ([float(x) for x in got] == [float(x) for x in want]
                  if isinstance(got, (list, tuple)) else False)
        elif isinstance(want, bool):
            ok = got is want or int(got) == int(want)
        elif isinstance(want, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - want) < 1e-9
        else:
            ok = got == want
        assert ok, (f"train_ds.{key}: config {got!r} != registered {want!r} "
                    f"(SPEC section 2)")
    return DATA


def load_inits_registry(path=None):
    """The registered init set (experiments/inits/INITS.json)."""
    with open(path or INITS_REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)


def assert_registered_init(init_path, registry=None):
    """Refuse to spend GPU time on an init that is not the registered one.

    SPEC section 1.3 requires v2 and v3 to be seeded from ONE shared randomly
    initialised decoder+joint so the two arms differ only in the encoder. Two
    byte-different init sets existed on this box at once (an earlier
    p3_probe/build_inits.py build and the registered
    tools/p3_build_inits.py one), and the smoke runs used the unregistered
    copy. If v2 and v3 are ever pilotted from different heads the Gate-3 AUC
    comparison measures the head, not the encoder, and nothing downstream can
    detect it. So the sha256 is checked against the registry, and the shared
    head digest is checked to be common to both reinit arms, BEFORE the model
    is restored.

    Returns (arm_name, sha256).
    """
    reg = registry if registry is not None else load_inits_registry()
    got = sha256_file(init_path)
    arms = reg["arms"]
    match = [name for name, a in arms.items() if a.get("sha256") == got]
    assert match, (
        f"init {init_path} sha256 {got} is not in {INITS_REGISTRY}. "
        f"Registered: "
        + ", ".join(f"{n}={a.get('sha256', '')[:16]}" for n, a in arms.items()))
    shared = reg["shared_head"]["digest"]
    for name in ("init_v2_reinit", "init_v3_reinit"):
        assert arms[name]["head_digest"] == shared, (
            f"{name} head_digest != registered shared_head digest; SPEC "
            f"section 1.3's byte-identical head is not evidenced")
    return match[0], got
