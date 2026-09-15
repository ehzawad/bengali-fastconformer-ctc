#!/usr/bin/env python
"""this recipe warm-start trainer: FastConformer-CTC large, init A, A5000 only.

Implements experiments/SPEC.md §4 and §5 on a NeMo 2.7.3
``EncDecCTCModelBPE`` restored from the the baseline checkpoint
(``experiments/fastconformer_ctc_bn_1kh/final.nemo``, sha256 61362a36…):
ALL weights including BatchNorm buffers, tokenizer unchanged, FRESH AdamW and
FRESH scheduler.

Why a hand-written loop rather than the baseline's ``Trainer.fit`` (src/training.py)
-----------------------------------------------------------------------------
the baseline could delegate to Lightning because nothing about its bookkeeping was
load-bearing for a later comparison. this recipe's is:

  * SPEC §5 makes each pilot arm exactly ``8,000 x 252 = 2.016 M`` presentations
    and compares arms at that fixed exposure. Lightning's accumulation counter
    is derived from the dataloader and silently re-phases across epochs (Phase
    1 hit exactly this: an int ``val_check_interval`` restarted every epoch, so
    validation stopped happening between step 1,000 and ~2,450 - see
    config.launch.yaml). Counting updates and utterances here makes the
    exposure claim a measured number rather than an inference.
  * SPEC §4's resume semantics ("restores optimizer, scheduler, sampler cursor
    and RNG so an interrupted run replays the SAME sample ids and LRs") cannot
    be asserted from inside a framework that owns the sampler. The sampler
    cursor correction and the resume handshake here are Phase 3's
    (bnasr/train_core.py, imported not copied): the DataLoader prefetches
    ``num_workers x prefetch_factor`` microbatches out of the in-process Lhotse
    sampler, so the sampler's own counter runs AHEAD of the training loop and
    a naive ``load_state_dict`` skips rows on resume. That defect shipped once
    already in Phase 3 and was caught only by a cut-id level drill.
  * SPEC §4's watchdog (13.0 GiB since the 2026-09-07 share-a-card ruling,
    SPEC §8i) has to stop THIS process cleanly (the box is
    shared and a co-tenant owns the A6000); an OOM inside ``fit()`` is a
    traceback and a zombie holding several GiB.
  * The selection score S, its panels, the regression guards, the 20k progress
    gate and the plateau rule are all registered stopping conditions. They are
    computed here from both scoring surfaces, so a run cannot continue past a
    STOP it has already earned.

Modes (all of them GPU-cheap except training itself)
----------------------------------------------------
    --restore-check            zero-update restoration: load init A through the
                               EXACT training init path, decode panel D with
                               the frozen scorer, require 20.35 % ± 0.05, and
                               write experiments/baseline_p1.json (the
                               MEASURED per-source and conversational-slice
                               baselines the §2 guards need).
    --profile-longest          peak reserved memory at microbatch 12 on the 21
                               longest admitted clips (SPEC §3 G0.7, ≤16.5 GiB).
    --build-smoke-manifest OUT 2,000 rows of the old corpus, targets
                               re-normalised through the SPEC §1 surface.
    --build-longest-manifest   the N longest old-corpus clips (profiling input).
    (default)                  train; --steps 8000 is SPEC §5's pilot.

Usage
-----
    CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv/bin/python bnasr/train.py --arm O --run-name pilot_O --steps 8000

Outputs under ``experiments/runs/<run_name>/``: ``steps.jsonl`` (one line per
optimizer update, including the LR and the cut ids that made it), ``val.jsonl``,
``hourly.jsonl``, ``run_meta.json``, ``ckpt/`` (last 2), ``best/`` (best 3 by S),
``milestones/`` (immutable), and ``STOP``/``ABORT`` records.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bnasr import common as P4  # noqa: E402
from bnasr.common import (GIB, REPO, HourlyLogger,  # noqa: E402
                                anonymize_process, atomic_save, ctc_panel_eval,
                                guards, load_baseline, load_manifest, lr_at,
                                make_param_groups, pin_gpu, selection_score,
                                sha256_file, train_surface)
# Phase 3's sampler-cursor machinery, imported rather than re-implemented: the
# prefetch-lead correction and the epoch loop are the parts that were proven
# wrong once and fixed with evidence (bnasr/train_core.py docstrings).
from bnasr.train_core import (BatchStream, hard_exit,  # noqa: E402
                            sampler_load_state_dict, sampler_state_dict)

anonymize_process()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ABORT_EXIT = 3          # unrecoverable: no in-process retry
STOP_EXIT = 4           # a REGISTERED stop (watchdog, gate, plateau, pilot rule)
_RUN_DIR = [None]
_RUN_LOCK = [None]


def _record(kind, reason, run_dir=None, **ctx):
    payload = {kind.lower(): reason,
               "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
               **ctx}
    line = json.dumps(payload, ensure_ascii=False, default=str)
    print(f"{kind} " + line, flush=True)
    if run_dir is not None:
        with open(Path(run_dir) / kind, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return payload


def abort(reason, run_dir=None, **ctx):
    """Stop now, loudly, with the reason on disk and exit code 3.

    An abort means the run's own premises were violated (non-finite loss, a
    dropped row, a CUDA fault). Retrying in-process would produce a run whose
    record no longer describes what happened.
    """
    _record("ABORT", reason, run_dir, **ctx)
    hard_exit(ABORT_EXIT)


def stop(reason, run_dir=None, ending="terminal", **ctx):
    """A REGISTERED stop: watchdog, 20k gate, plateau, or a §5 pilot rule.

    Distinct from abort() by exit code, because these are protocol outcomes -
    the checkpoint on disk is valid and the run ended the way SPEC says it
    should. A watchdog that exited 3 would be indistinguishable from a crash.

    `ending` (SPEC §8p, owner-delegated ruling 2026-09-07) classifies the
    outcome for the POST-RUN procedure, not for the supervisor: "normal" means
    training is over and the run proceeds to §4 checkpoint selection and the
    §6 headline (the plateau stop, and reaching H); "terminal" means no
    headline follows without an owner ruling (the 20k gate, pilot aborts, the
    watchdog). The supervisor treats both as non-restartable, which is right
    for both - training does not resume after either.
    """
    assert ending in ("normal", "terminal"), ending
    _record("STOP", reason, run_dir, ending=ending, **ctx)
    hard_exit(STOP_EXIT)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _make_dataset_class():
    """Lhotse dataset that also returns cut ids, durations and a skip count.

    NeMo's ``LhotseSpeechToTextBpeDataset`` loads audio with
    ``AudioSamples(fault_tolerant=True)``: an unreadable file is DROPPED and
    nothing is said. A silently short microbatch would break both the 252-per-
    update exposure claim and the resume handshake, so the drop is made
    observable and any occurrence aborts the run.
    """
    from nemo.collections.asr.data.audio_to_text_lhotse import \
        LhotseSpeechToTextBpeDataset

    class _P4(LhotseSpeechToTextBpeDataset):
        def __init__(self, tokenizer):
            super().__init__(tokenizer, return_cuts=True)

        def __getitem__(self, cuts):
            n_in = len(cuts)
            audio, audio_lens, tokens, token_lens, out = super().__getitem__(cuts)
            return {"audio": audio, "audio_lens": audio_lens,
                    "tokens": tokens, "token_lens": token_lens,
                    "ids": [c.id for c in out],
                    "durations": [float(c.duration) for c in out],
                    "n_skipped": n_in - len(out)}
    return _P4


P4LhotseDataset = None      # bound in main(), after NeMo is importable


def build_loader(cfg, tokenizer, manifest):
    """Lhotse train loader with SPEC §4's sampler keys, verbatim.

    Fixed-size microbatches: ``batch_size`` is set and ``batch_duration`` is
    NOT, so Lhotse's TimeConstraint degenerates to ``max_cuts`` and every
    microbatch is exactly 12 cuts (with ``drop_last: true``). That is what makes
    "252 presentations per update" and "2.016 M per pilot arm" (SPEC §5) exact
    rather than approximate, and it also bounds memory analytically at
    12 x longest-clip - the quantity --profile-longest measures.
    """
    from nemo.collections.common.data.lhotse import \
        get_lhotse_dataloader_from_config
    d = cfg["data"]
    ds_cfg = {
        "use_lhotse": True,
        "manifest_filepath": str(manifest),
        "sample_rate": int(d["sample_rate"]),
        "text_field": d["text_field"],
        "batch_size": int(cfg["batch"]["microbatch"]),
        "use_bucketing": bool(d["use_bucketing"]),
        "num_buckets": int(d["num_buckets"]),
        "bucket_duration_bins": [float(x) for x in d["bucket_duration_bins"]],
        "bucket_buffer_size": int(d["bucket_buffer_size"]),
        "shuffle_buffer_size": int(d["shuffle_buffer_size"]),
        "shuffle": bool(d["shuffle"]),
        "seed": int(d["seed"]),
        "shard_seed": int(d["shard_seed"]),
        "concurrent_bucketing": bool(d["concurrent_bucketing"]),
        "num_workers": int(d["num_workers"]),
        "pin_memory": bool(d["pin_memory"]),
        "min_duration": float(d["min_duration"]),
        "max_duration": float(d["max_duration"]),
        "drop_last": bool(d["drop_last"]),
        "skip_missing_manifest_entries": bool(d["skip_missing_manifest_entries"]),
        # Throughput knob (2026-09-06): NeMo tokenises in the MAIN process when pretokenize=True (its own warning);
        # the from-cache drills showed the main process at 63 % CPU and the GPU at 28 %. False moves tokenisation
        # into the dataloader workers. It does not change the sampled cut order (no max_tps filter is used), which
        # the cross-card/replay checks verify; the value used is recorded in run_meta.json.
        "pretokenize": bool(d.get("pretokenize", True)),
    }
    assert ds_cfg["concurrent_bucketing"] is False, (
        "concurrent_bucketing must be false (SPEC §4); NeMo defaults it True "
        "and Lhotse then samples non-deterministically")
    assert "max_tps" not in ds_cfg, (
        "a runtime tokens-per-second filter would silently drop rows; SPEC §1 "
        "registers CTC feasibility at corpus-build time instead")
    loader = get_lhotse_dataloader_from_config(
        ds_cfg, global_rank=0, world_size=1,
        dataset=P4LhotseDataset(tokenizer), tokenizer=tokenizer)
    return loader, loader.sampler


# --------------------------------------------------------------------------
# Model contract (SPEC §4 init A)
# --------------------------------------------------------------------------
def bn_buffer_digest(model):
    """sha256 over every BatchNorm running buffer, in a fixed name order.

    SPEC §4 requires init A to carry "BN buffers preserved" and council B item
    17 asks that evaluation be shown to change zero buffers. A digest makes
    both checkable in one number rather than by eyeballing tensors: it is taken
    right after the restore, again after the restoration decode, and recorded in
    run_meta.json.
    """
    h = hashlib.sha256()
    for name, buf in sorted(model.named_buffers()):
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            h.update(name.encode())
            h.update(buf.detach().to("cpu", torch.float64).numpy().tobytes())
    return h.hexdigest()


def load_init_N(cfg, device, verify_sha=True):
    """Init N (SPEC §8q/§8s): the NVIDIA English FastConformer-CTC large with a
    FRESH 1,025-way Bengali head - the baseline's own cold-start construction.

    What is kept from NVIDIA: encoder weights and BatchNorm buffers, the
    preprocessor. What is replaced: the entire CTC head including blank
    (``change_vocabulary`` builds a new ConvASRDecoder for the registered
    1,024-piece Bengali tokenizer), and SpecAugment, which NVIDIA ships as
    2 x 27 / 10 x 0.05 and SPEC §4 registers as 2 x 15 / 2 x 0.05. Not one
    the baseline tensor is imported. The head initialisation is seeded and hashed so
    the run record can prove which head it started from.

    Returns (model, sha256 of the NVIDIA archive); provenance in
    ``model.p4_init_meta``.
    """
    from nemo.collections.asr.models import EncDecCTCModelBPE
    from nemo.collections.asr.modules import SpectrogramAugmentation
    from omegaconf import OmegaConf, open_dict
    init = Path(cfg["paths"]["init_N"])
    sha = sha256_file(init) if verify_sha else None
    if verify_sha:
        want = str(cfg["paths"]["init_N_sha256_prefix"])
        assert sha.startswith(want), (
            f"init N {init} sha256 {sha[:16]} does not start with the registered "
            f"{want} (SPEC §8s)")
    model = EncDecCTCModelBPE.restore_from(str(init), map_location="cpu")
    # Same architecture as A, asserted before anything is touched.
    e = model.cfg.encoder
    assert int(e.n_layers) == 18 and int(e.d_model) == 512, "encoder shape"
    assert str(e.subsampling) == "dw_striding" and int(e.subsampling_factor) == 8
    assert str(e.conv_norm_type) == "batch_norm" and int(e.conv_kernel_size) == 9
    pp = model.cfg.preprocessor
    assert int(pp.features) == 80 and int(pp.sample_rate) == 16000, "preprocessor"
    enc_digest_before = _tensor_digest(model, prefix="encoder.")
    bn_digest = bn_buffer_digest(model)
    # The head: replaced in full, from a recorded seed.
    tok_dir = REPO / cfg["paths"]["tokenizer_dir"]
    tok_sha = sha256_file(tok_dir / "tokenizer.model")
    assert tok_sha.startswith(P4.TOKENIZER_SHA256_PREFIX), "tokenizer (SPEC §3 G0.5)"
    seed = int(cfg.get("init_N", {}).get("head_seed", P4.INIT_N_HEAD_SEED))
    torch.manual_seed(seed)
    model.change_vocabulary(new_tokenizer_dir=str(tok_dir), new_tokenizer_type="bpe")
    assert model.tokenizer.vocab_size == 1024, "tokenizer vocab 1024"
    assert model.decoder.num_classes_with_blank == 1025, "fresh head 1,024 + blank"
    head_digest = _tensor_digest(model, prefix="decoder.")
    enc_digest_after = _tensor_digest(model, prefix="encoder.")
    assert enc_digest_after == enc_digest_before, "change_vocabulary must not touch the encoder"
    # SpecAugment: SET to the registered values (NVIDIA ships different ones).
    want = cfg["specaug"]
    model.spec_augmentation = SpectrogramAugmentation(
        freq_masks=int(want["freq_masks"]), time_masks=int(want["time_masks"]),
        freq_width=int(want["freq_width"]), time_width=float(want["time_width"]))
    with open_dict(model.cfg):
        model.cfg.spec_augment = OmegaConf.create({
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": int(want["freq_masks"]), "time_masks": int(want["time_masks"]),
            "freq_width": int(want["freq_width"]), "time_width": float(want["time_width"])})
    sa = model.spec_augmentation.spec_augment
    assert int(sa.freq_masks) == int(want["freq_masks"]) and int(sa.time_masks) == int(want["time_masks"])
    assert int(sa.freq_width) == int(want["freq_width"]) and abs(float(sa.time_width) - float(want["time_width"])) < 1e-9
    strategy = str(model.cfg.decoding.strategy)
    assert strategy in ("greedy", "greedy_batch"), f"decoding {strategy!r} not greedy"
    model.p4_init_meta = {"init": "N", "init_N_sha256": sha, "head_seed": seed,
                          "head_digest": head_digest, "encoder_digest": enc_digest_after,
                          "bn_buffer_digest_at_load": bn_digest,
                          "specaug_replaced_from": "2x27/10x0.05 (NVIDIA) -> registered"}
    model = model.to(device)
    return model, sha


def _tensor_digest(model, prefix):
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        if name.startswith(prefix):
            h.update(name.encode()); h.update(t.detach().to("cpu", torch.float32).numpy().tobytes())
    return h.hexdigest()


def load_init(cfg, device, verify_sha=True):
    """Init A: restore ALL of final.nemo, assert it is the registered file.
    (Init N is dispatched to load_init_N; SPEC §8q/§8s.)

    ``restore_from`` brings weights, BatchNorm buffers, the SentencePiece
    tokenizer and the decoding config out of the archive together - which is
    exactly what SPEC §4 registers, and the reason nothing here rebuilds the
    model from a config (Phase 3's memory note: NeMo's ``change_vocabulary``
    silently swaps the loss; the CTC analogue would silently reinitialise the
    1,024+blank head this phase is trying to keep).

    Returns (model, sha256).
    """
    if str(cfg.get("init", "A")) == "N":
        return load_init_N(cfg, device, verify_sha)
    from nemo.collections.asr.models import EncDecCTCModelBPE
    init = REPO / cfg["paths"]["init"]
    sha = sha256_file(init) if verify_sha else None
    if verify_sha:
        want = cfg["paths"]["init_sha256_prefix"]
        assert sha.startswith(want), (
            f"init {init} sha256 {sha[:16]} does not start with the registered "
            f"{want} (SPEC §4)")
    model = EncDecCTCModelBPE.restore_from(str(init), map_location="cpu")

    # Tokenizer unchanged (SPEC §4, §3 G0.5). The archive carries its own copy;
    # assert it is byte-identical to the registered the baseline tokenizer rather
    # than trusting that the .nemo was built from it.
    tok_model = REPO / cfg["paths"]["tokenizer_dir"] / "tokenizer.model"
    tok_sha = sha256_file(tok_model)
    assert tok_sha.startswith(P4.TOKENIZER_SHA256_PREFIX), (
        f"tokenizer {tok_model} sha256 {tok_sha[:16]} != registered "
        f"{P4.TOKENIZER_SHA256_PREFIX} (SPEC §3 G0.5)")
    assert model.tokenizer.vocab_size == 1024, (
        f"tokenizer vocab {model.tokenizer.vocab_size} != 1024 (SPEC §4)")
    assert model.decoder.num_classes_with_blank == 1025, (
        "CTC head must be 1,024 + blank (SPEC §4)")

    # Architecture, asserted not assumed (SPEC §4: 18 layers, d 512,
    # dw-striding 8x, conv BatchNorm, kernel 9).
    e = model.cfg.encoder
    assert int(e.n_layers) == 18 and int(e.d_model) == 512, "encoder shape"
    assert str(e.subsampling) == "dw_striding", "subsampling"
    assert int(e.subsampling_factor) == 8, "subsampling factor 8x"
    assert str(e.conv_norm_type) == "batch_norm", "conv BatchNorm"
    assert int(e.conv_kernel_size) == 9, "conv kernel 9"

    # SpecAugment 2 x 15 / 2 x 0.05 (SPEC §4, the baseline values) - checked on the
    # live module, because that is what actually augments the batch.
    sa = model.spec_augmentation
    assert sa is not None, "init A has no SpectrogramAugmentation"
    want = cfg["specaug"]
    assert int(sa.spec_augment.freq_masks) == int(want["freq_masks"])
    assert int(sa.spec_augment.time_masks) == int(want["time_masks"])
    assert int(sa.spec_augment.freq_width) == int(want["freq_width"])
    assert abs(float(sa.spec_augment.time_width) - float(want["time_width"])) < 1e-9

    # Greedy CTC decoding only: SPEC §4 forbids a decoder search anywhere in
    # the phase, and the selection score would otherwise measure the search.
    strategy = str(model.cfg.decoding.strategy)
    assert strategy in ("greedy", "greedy_batch"), (
        f"decoding strategy {strategy!r} is not greedy (SPEC §4)")

    model = model.to(device)
    return model, sha


# --------------------------------------------------------------------------
# One optimizer update
# --------------------------------------------------------------------------
def forward_microbatch(model, batch, device):
    """encoder -> CTC head -> CTC loss for one microbatch, under bf16 autocast.

    NeMo builds this model's CTCLoss with ``zero_infinity=True``
    (ctc_models.py), so a row whose targets are longer than its frames yields
    loss 0 and gradient 0 rather than +inf. That matters here because SPEC §1's
    feasibility filter runs at corpus-build time and a single infeasible row
    that slipped through would otherwise NaN the whole 252-utterance update and
    AdamW's moments with it.
    """
    sig = batch["audio"].to(device, non_blocking=True)
    sig_len = batch["audio_lens"].to(device, non_blocking=True)
    tok = batch["tokens"].to(device, non_blocking=True)
    tok_len = batch["token_lens"].to(device, non_blocking=True)
    log_probs, enc_len, _ = model.forward(input_signal=sig,
                                          input_signal_length=sig_len)
    loss = model.loss(log_probs=log_probs, targets=tok,
                      input_lengths=enc_len, target_lengths=tok_len)
    return loss, enc_len.detach(), tok_len.detach()


def gather_microbatches(data_iter, accum, run_dir, update):
    """Pull exactly `accum` Lhotse microbatches for one optimizer update.

    Held as raw CPU waveform (a few MB each) before any forward pass, so the
    update's utterance count is known before the first backward and the weights
    below sum to 1 exactly. A skipped row aborts: see _make_dataset_class().
    """
    micros, n_utt, epoch = [], 0, None
    for _ in range(accum):
        epoch, batch = next(data_iter)
        if batch["n_skipped"]:
            abort("skipped row(s) in a microbatch", run_dir, update=update,
                  n_skipped=int(batch["n_skipped"]), ids=batch["ids"][:4])
        micros.append(batch)
        n_utt += len(batch["ids"])
    return micros, n_utt, epoch


def run_update(model, opt, micros, n_utt, cfg, device, run_dir, update):
    """One optimizer update over `accum` microbatches (SPEC §4).

    Weighting: each microbatch loss is scaled by n_i / N_accum before backward.
    With ``drop_last: true`` every microbatch holds exactly 12 cuts, so this is
    the uniform 1/21 that the baseline's Lightning accumulation applied - the same
    gradient the warm-started checkpoint was trained with. The n_i form is kept
    (rather than a hard-coded 1/21) so that a short final microbatch, if the
    sampler ever produced one, would change the average instead of silently
    over-weighting its rows.

    The pre-clip gradient norm is what gets logged: clipping hides a blow-up,
    and the hourly report's grad-clip frequency is the diagnostic SPEC §4 asks
    for.
    """
    loss_value, n_tokens, sum_seconds, max_seconds = 0.0, 0, 0.0, 0.0
    b_min = 10 ** 9
    for batch in micros:
        n_i = len(batch["ids"])
        b_min = min(b_min, n_i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, enc_len, tok_len = forward_microbatch(model, batch, device)
        # Out of autocast on purpose: scaling and backward are fp32, and the
        # master weights plus Adam moments stay fp32 (bf16 is the compute dtype
        # inside the forward only - SPEC §4 "bf16").
        loss_f = loss.float()
        if not torch.isfinite(loss_f):
            abort("non-finite microbatch loss", run_dir, update=update,
                  loss=float(loss_f), ids=batch["ids"][:4])
        (loss_f * (n_i / n_utt)).backward()
        loss_value += float(loss_f) * (n_i / n_utt)
        n_tokens += int(tok_len.sum())
        sum_seconds += float(sum(batch["durations"]))
        max_seconds = max(max_seconds, float(max(batch["durations"])))
        del loss, loss_f

    params = [p for p in model.parameters() if p.grad is not None]
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        params, float(cfg["optim"]["grad_clip"])))
    if not math.isfinite(gnorm):
        abort("non-finite gradient norm", run_dir, update=update,
              grad_norm=gnorm)

    lr = lr_at(update, sched=P4.sched_for(str(cfg.get("init", "A"))))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"loss": loss_value, "grad_norm": gnorm, "lr": lr,
            "n_tokens": n_tokens, "sum_seconds": sum_seconds,
            "max_seconds": max_seconds, "min_microbatch": b_min,
            "clipped": gnorm > float(cfg["optim"]["grad_clip"])}


# --------------------------------------------------------------------------
# Checkpoints (SPEC §4: atomic, last 2 + best 3 by S + immutable milestones)
# --------------------------------------------------------------------------
def save_checkpoint(path, model, opt, sampler, update, epoch, cfg, consumed,
                    last_micro_ids, extra=None):
    """Full resume point: weights, optimizer, sampler cursor, RNG, counters.

    The scheduler has no separate state: lr_at() is a pure function of the
    update index and the horizon, both of which are stored, so there is no
    scheduler object whose state could drift out of sync with the optimizer's.
    The HORIZON is recorded because a resume under a different H is a different
    LR curve on the same weights, and is refused on resume.
    """
    return atomic_save({
        "update": update,
        "epoch": epoch,
        "consumed_batches": int(consumed),
        "last_micro_ids": list(last_micro_ids or []),
        "horizon": int(cfg["sched"]["horizon"]),
        "effective_batch": int(cfg["batch"]["effective"]),
        "init": str(cfg.get("init", "A")),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "sampler": sampler_state_dict(sampler, consumed),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "numpy": np.random.get_state()},
        "cfg": cfg,
        "extra": extra or {},
    }, path)


def _link_or_copy(src, dst):
    """Hard-link a checkpoint into best/ or milestones/ when the filesystem
    allows it: a 1.8 GB copy per protected checkpoint would cost tens of GB
    over a 150k-step run, and a hard link survives the rolling unlink of the
    original (that is the whole point of a link count)."""
    import shutil
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def decision_s(val):
    """The S a DECISION must use: exact when the record has it, else recorded.

    SPEC §8w.9. Records written before 2026-09-09 carry only the 4-dp `S_pp`;
    their exactness was verified independently against the preserved
    predictions (experiments/unrounded_recon.json, largest discrepancy
    1.39e-04 pp), so falling back to them is sound and changes no past decision.
    """
    if not val:
        return None
    v = val.get("S_exact")
    return float(v) if v is not None else (
        None if val.get("S_pp") is None else float(val["S_pp"]))


def retain(run_dir, update, s_score, cfg, history, eligible=False):
    """Apply SPEC §4's retention rule after a checkpoint is written.

    last 2 resumable  - rolling ckpt/step_*.pt, oldest unlinked
    best 3 by S       - best/ (hard links), lowest S wins
    selection         - the current best ELIGIBLE checkpoint and everything
                        that could still overtake it under the 0.05 pp tie
                        band, kept in best/ no matter where it ranks by S
    milestones        - milestones/ (hard links), never rotated

    Why the selection needs its own protection (council F Q2, SPEC §8k)
    ------------------------------------------------------------------
    Council F declined to add a mid-run abort rule and made checkpoint
    INELIGIBILITY the protection instead, "provided ... the selected eligible
    checkpoint is preserved independently of rotation". Best-3-by-S alone does
    not preserve it: S ignores the guards, so three INELIGIBLE checkpoints
    with lower S evict the best eligible one from best/, and because ckpt/
    keeps only the last 2, unlinking the best/ hard link deletes the only copy
    of the model the protocol says must be selected. The run would then finish
    with a "final checkpoint = lowest eligible S" rule and no such file.

    The protected set is closed under the future as well as the present: an
    entry is kept while it is eligible and within TIE_BAND_PP of the best
    eligible S. The best eligible S can only fall, so that band only moves
    down - an entry outside it can never re-enter, and any entry that could
    ever win the earliest-of-the-tied sweep is inside it now. That is what
    makes "never unlink the current best eligible" true for every LATER
    selection too, not just the one visible at this validation.

    `eligible` is the guard verdict for THIS validation (common.is_eligible
    of the val record). A checkpoint written outside a validation - a
    milestone, a periodic robustness checkpoint - has no S and no guard
    verdict and takes part in neither ranking.
    """
    ck = run_dir / "ckpt" / f"step_{update}.pt"
    if update in tuple(int(x) for x in cfg["ckpt"]["milestones"]):
        _link_or_copy(ck, run_dir / "milestones" / f"step_{update}.pt")
    if s_score is not None:
        history.append((float(s_score), int(update), bool(eligible)))
        ranked = sorted(history, key=lambda t: (t[0], t[1]))
        keep_u = [u for _, u, _ in ranked[:int(cfg["ckpt"]["keep_best"])]]
        elig = [(s, u) for s, u, e in history if e]
        if elig:
            best_e = min(s for s, _ in elig)
            keep_u += [u for s, u in elig if s <= best_e + P4.TIE_BAND_PP]
        keep_u = sorted(set(keep_u))
        for u in keep_u:
            src = run_dir / "ckpt" / f"step_{u}.pt"
            if src.exists():
                _link_or_copy(src, run_dir / "best" / f"step_{u}.pt")
        keep_names = {f"step_{u}.pt" for u in keep_u}
        for old in (run_dir / "best").glob("step_*.pt"):
            if old.name not in keep_names:
                old.unlink()
    files = sorted((run_dir / "ckpt").glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in files[:-int(cfg["ckpt"]["keep_last"])]:
        old.unlink()
    # Post-condition, checked rather than asserted in a comment: after the
    # rolling unlink the SELECTED checkpoint is still on disk. A hard link in
    # best/ or milestones/ is what keeps it there, so this fires only if a
    # link was never made (an _link_or_copy fallback that failed, a best/
    # pruned by hand). Losing it is unrecoverable - the model would have to be
    # retrained - so the run stops here rather than at the end of the horizon.
    sel = P4.select_best(history)
    if sel is not None:
        u = sel[1]
        alive = any((run_dir / d / f"step_{u}.pt").exists()
                    for d in ("best", "milestones", "ckpt"))
        if not alive:
            abort("retention deleted the selected eligible checkpoint "
                  "(SPEC §4 final selection, §8k council F Q2)", run_dir,
                  update=update, selected_update=u, selected_S_pp=sel[0])
    return history


# --------------------------------------------------------------------------
# Validation (SPEC §2 panels, §4 both surfaces)
# --------------------------------------------------------------------------
def should_validate(update, val_every, stop_at, val_at, init="A"):
    """Is `update` a validation step (SPEC §4, and §8k Q8's gate)?

    The registered cadence is every 2,000 updates, plus the run's own step
    target, plus anything --val-at adds, plus - unconditionally - update
    20,000 whenever the run is heading that far.

    That last clause is the one with teeth. The 20,000 progress gate is a
    REGISTERED stop and it can only be evaluated at a validation, but
    `--val-every` is a CLI flag with no config guard (unlike `val.every`,
    which common.configure asserts): `--val-every 0` turns validation off
    entirely, which is exactly what the resume-equivalence drill needs and
    exactly what must not silently disable the check council F called "the
    decisive cheap check ... cost to reach the decision ~15-19 GPU-h; the
    failure it prevents ~157-210 GPU-h" (SPEC §8k Q8). A pilot that stops at
    8,000 never reaches 20,000, so the clause costs a pilot nothing.
    """
    val_at = val_at or ()
    if init == "N":
        if update in (P4.GATE_N40_UPDATE, P4.GATE_N60_UPDATE) and stop_at >= update:
            return True
    elif update == P4.GATE20K_UPDATE and stop_at >= P4.GATE20K_UPDATE:
        return True
    if update in val_at:
        return True
    if not val_every:
        return False
    return bool(update % val_every == 0 or update == stop_at)


def check_resume_state(state, cfg, run_dir=None):
    """Refuse a resume whose checkpoint does not describe THIS run (SPEC §4).

    Called before anything is loaded out of `state`. Two premises are checked
    because breaking either produces a run whose record is a lie rather than a
    crash:

      * HORIZON. lr_at() has no scheduler object; the LR is a pure function of
        (update, horizon), so resuming a 150,000-horizon checkpoint under any
        other H silently puts every remaining update on a different cosine leg
        while the loss curve still looks healthy. SPEC §4 registers H = 150,000
        and §8k Q4 re-affirms it ("KEEP H=150,000 ... no second warmup, no LR
        reset"), so a mismatch is refused, not adapted to.
      * EFFECTIVE BATCH. SPEC §5 compares arms at a fixed number of
        PRESENTATIONS (8,000 x 252); a resume at a different effective batch
        would keep counting updates and quietly change what an update means.

    This is a recorded ABORT and not an `assert`: `python -O` strips asserts,
    and an AssertionError out of main() would be reported by the supervisor as
    a crash - which is a RESUMABLE outcome (run_supervised.sh re-queues
    crashes), so the refusal would turn into an infinite relaunch loop against
    the same bad checkpoint.
    """
    got_h, want_h = int(state.get("horizon", -1)), int(cfg["sched"]["horizon"])
    if got_h != want_h:
        abort("resume refused: checkpoint horizon does not match the config; "
              "a resume under a different H is a different LR curve on the "
              "same weights (SPEC §4, §8k Q4)", run_dir,
              checkpoint_horizon=got_h, config_horizon=want_h,
              update=int(state.get("update", -1)))
    got_i, want_i = str(state.get("init", "A")), str(cfg.get("init", "A"))
    if got_i != want_i:
        abort("resume refused: checkpoint init does not match the config "
              "(SPEC §8s: an A checkpoint must never continue as N, nor N as A)",
              run_dir, checkpoint_init=got_i, config_init=want_i,
              update=int(state.get("update", -1)))
    got_b = int(state.get("effective_batch", -1))
    want_b = int(cfg["batch"]["effective"])
    if got_b != want_b:
        abort("resume refused: checkpoint effective batch does not match the "
              "config; SPEC §5 fixes the exposure at 252 per update", run_dir,
              checkpoint_effective_batch=got_b, config_effective_batch=want_b,
              update=int(state.get("update", -1)))
    return True


def validate(model, cfg, update, run_dir, args, baseline):
    """Decode every available panel, score S, evaluate the registered guards.

    Panels come from SPEC §2: D (locked dev, per-source breakdown and the
    conversational slice), M (MaCro dev slice, only once build_corpus has
    written it), K (Kathbath official valid) and R (RESPIN official dev, and its
    presence is what selects the MADASR weighting of S).

    ``torch.cuda.empty_cache()`` runs ONCE after the panels (SPEC §4). This is
    not the in-loop cache-flush that would hide training-step fragmentation: it
    releases the decode's allocator blocks, which in Phase 3 were measured
    growing reserved memory 16.74 -> 17.56 GiB across one validation and never
    coming back - enough to walk this run into its own reserved-memory
    watchdog (13.0 GiB, SPEC §8i).
    """
    # Council Q #6: this loop used to SILENTLY DROP a registered panel whose file
    # was absent, and `madasr_admitted` was then inferred from whichever panels
    # happened to survive (`"R" in res`). A vanished R therefore switched the
    # weight vector from 0.60/0.20/0.10/0.10 to 0.60/0.30/0.10 and produced a
    # LOWER S with no model change and no error - a fictitious improvement of
    # ~1.5 pp on plausible values, in the metric that selects the shipped
    # checkpoint. The registered panel set and the weight vector are part of the
    # contract, so a missing panel is an integrity failure, not a re-weighting.
    panels, res, missing_panels = {}, {}, []
    for name, rel in cfg["val"]["panels"].items():
        p = Path(rel)
        p = p if p.is_absolute() else (REPO / p)
        if p.exists():
            panels[name] = p
        else:
            missing_panels.append(f"{name}={rel}")
    if missing_panels:
        abort("registered validation panel(s) missing: " + ", ".join(missing_panels)
              + ". S is defined on the registered panel set (SPEC §2); dropping a "
                "panel would silently change the weight vector and the selection "
                "metric. Restore the file or amend the contract.",
              run_dir, update=int(update), missing_panels=missing_panels)
    for name, path in panels.items():
        r = ctc_panel_eval(model, path, batch_size=args.val_batch_size,
                           limit=args.val_limit, per_source=(name == "D"),
                           conversational=(name == "D"))
        hyps = r.pop("hyps")
        (run_dir / "val_preds").mkdir(exist_ok=True)
        (run_dir / "val_preds" / f"{name}_step_{update}.json").write_text(
            json.dumps(hyps, ensure_ascii=False), encoding="utf-8")
        res[name] = r
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved() / GIB
    torch.cuda.empty_cache()

    wers = {k: v["wer_frozen"] for k, v in res.items()}
    # The weighting is decided by the REGISTERED admission (§8m: MADASR admitted
    # 2026-09-08), not by which panel files exist at this instant.
    #
    # SPEC §8w.9 / council R: S_exact is what every DECISION reads - selection,
    # the 0.05 pp tie band, retention, eligibility and qualification. S_pp stays
    # the rounded display value so every record and reader already written keeps
    # working. Council R's verified counterexample: exact ΔS 0.049935 pp keeps
    # two checkpoints TIED (so the earlier is selected) while the rounded
    # ΔS 0.050400 pp splits them, selects the later, and lets `retain` delete
    # every copy of the earlier - which no preserved hypothesis can rebuild.
    s, s_meta = selection_score(wers, madasr_admitted=P4.madasr_admitted(cfg))
    s_exact, s_exact_meta = P4.selection_score_exact(res, P4.madasr_admitted(cfg))
    s_meta["exact"] = s_exact_meta
    out = {"update": update, "panels": res, "S_pp": s, "S_exact": s_exact,
           "S_meta": s_meta,
           "guards": guards(res.get("D", {}), (baseline or {}).get("D"),),
           "reserved_gib_before_release": round(before, 3),
           "reserved_gib_after_release": round(
               torch.cuda.memory_reserved() / GIB, 3),
           "val_limit": args.val_limit,
           "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
    with open(run_dir / "val.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"VAL step={update} S={s} (exact {s_exact}) "
          + " ".join(f"{k}={v['wer_frozen']}/{v['wer_nemo']}"
                     for k, v in sorted(res.items()))
          + f" empty={res.get('D', {}).get('empty_rate')}", flush=True)
    return out


GATE20K_FILE = "GATE20K"


def existing_val_record(run_dir, update, ckpt_sha=None):
    """The one val.jsonl record for `update`, or None; more than one is an ABORT.

    A resume must never score a checkpoint against somebody else's record. Two
    records for one update mean two processes validated the same state; the
    run's own history is then ambiguous and a human has to look.

    Council R (SPEC §8w.9): matching on the update NUMBER alone let an
    unrelated checkpoint's record be reused - it reproduced exactly that. When
    `ckpt_sha` is supplied and the record carries one, they must agree; a record
    stamped with a DIFFERENT checkpoint is refused rather than silently reused.
    A record with no stamp predates this binding and is accepted, once, on its
    update (the four this run wrote at 2,000-8,000).
    """
    p = Path(run_dir) / "val.jsonl"
    if not p.exists():
        return None
    hits = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("update") == int(update)]
    if len(hits) > 1:
        abort(f"resume refused: {len(hits)} val.jsonl records for update "
              f"{update}; the run's validation history is ambiguous",
              run_dir, update=update)
    if hits and ckpt_sha:
        got = hits[0].get("ckpt_sha256")
        if got and got != ckpt_sha:
            abort(f"resume refused: the val.jsonl record for update {update} was "
                  f"produced by a DIFFERENT checkpoint (recorded {got[:12]}…, "
                  f"current {ckpt_sha[:12]}…); it cannot be reused as this "
                  f"checkpoint's evaluation (SPEC §8w.9)",
                  run_dir, update=update, recorded_sha=got, current_sha=ckpt_sha)
    return hits[0] if hits else None


def gate20k_recorded(run_dir, sel):
    """Is there a durable PASS decision for the 20,000 gate for this run?"""
    p = Path(run_dir) / GATE20K_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("result") == "pass"
        except Exception:  # noqa: BLE001
            return False
    return (sel or {}).get("gate20k") == "pass"


def _median_d_frozen(run_dir, updates, live_value):
    """Median frozen panel-D WER (pp) over `updates`, or None if any is absent.

    `live_value` supplies the current update's own reading, which is not in
    val.jsonl yet when the milestone is evaluated. Equal numbers of observations
    on both sides keep the comparison cadence-matched (council Q Q2).
    """
    have = _d_frozen_history(run_dir)
    vals = []
    for u in updates:
        v = live_value if (live_value is not None and u == max(updates)) else have.get(u)
        if v is None:
            return None
        vals.append(float(v))
    vals.sort()
    return vals[len(vals) // 2]


def _d_frozen_history(run_dir):
    """{update: frozen panel-D WER in pp} from this run's val.jsonl.

    Council Q #3: records with a missing, non-integer or negative update, or a
    non-finite WER, are REFUSED rather than skipped - a synthetic `update: -1,
    D: 0` record was able to poison the withdrawn floor's minimum to zero.
    """
    out = {}
    path = Path(run_dir) / "val.jsonl" if run_dir is not None else None
    if path is None or not path.exists():
        return out
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        u = rec.get("update")
        if not isinstance(u, int) or isinstance(u, bool) or u < 0:
            raise ValueError(f"{path}:{lineno} has a missing or invalid update: {u!r}")
        w = (rec.get("panels", {}).get("D") or {}).get("wer_frozen")
        if w is None:
            continue
        pp = float(w) * 100.0
        if not math.isfinite(pp):
            raise ValueError(f"{path}:{lineno} has a non-finite D WER: {w!r}")
        if u in out and abs(out[u] - pp) > 1e-12:
            raise ValueError(f"{path} has conflicting records for update {u}")
        out[u] = pp
    return out


def _best_d_frozen_by(run_dir, ref_update):
    """The run's own best frozen panel-D WER (pp) at or before `ref_update`.

    Read from this run's val.jsonl - the same panel, the same surface and the
    same decoder as the value it will be compared against, so no cross-run or
    cross-surface conversion enters the comparison. None if no record exists.
    """
    best = None
    path = Path(run_dir) / "val.jsonl" if run_dir is not None else None
    if path is None or not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if int(rec.get("update", -1)) > int(ref_update):
            continue
        w = (rec.get("panels", {}).get("D") or {}).get("wer_frozen")
        if w is None:
            continue
        pp = float(w) * 100.0
        best = pp if best is None else min(best, pp)
    return best


def _n_gate_recorded(run_dir, sel, fname, key, accept=("pass",), expect_update=None):
    """Is the milestone's decision durably on disk (or carried in `sel`)?

    `accept` is the set of results that count as "the milestone completed".
    Under the terminal N40 policy only a "pass" lets the run continue past
    40,000. Under SPEC §8w's diagnostic policy a "missed" record is equally
    valid evidence that the milestone fired - council P: resume past 40k
    requires milestone-COMPLETION evidence, not a successful outcome.
    """
    p = Path(run_dir) / fname
    if not p.exists():
        # Council R (SPEC §8w.9): a bare in-memory flag used to satisfy the
        # invariant with no durable record at all. Milestone evidence must be
        # ON DISK, in this run's directory; `sel` alone is not evidence.
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    # Council Q #3: this used to accept ANY file whose "result" string matched -
    # a record carrying update 17 from an unrelated run was accepted as evidence
    # that the 40,000 milestone had fired. Bind it to the update it decides,
    # and (council R) to this run and this policy generation.
    if expect_update is not None and int(rec.get("update", -1)) != int(expect_update):
        return False
    want_run = (sel or {}).get("run_uid")
    if want_run and rec.get("run_uid") and rec["run_uid"] != want_run:
        return False
    want_pol = (sel or {}).get("n40_policy_generation")
    if want_pol and rec.get("n40_policy_generation") and rec["n40_policy_generation"] != want_pol:
        return False
    return rec.get("result") in accept


def check_gate_invariant_on_resume(update, stop_at, run_dir, sel):
    """Council H Q11: before executing any update past 20,000, exactly one PASS
    decision for the 20,000 checkpoint must exist. An equality-triggered gate
    that is skipped by a crash between the 20,000 checkpoint write and the
    gate decision would otherwise never fire again. This is the belt; the
    replay in `replay_owed_validation` is the braces.
    """
    init = str((sel or {}).get("init", "A"))
    if init == "N":
        # Council K: N at 20,500 without GATE20K MUST be accepted; N past its
        # own gates without ITS records must be refused. Never cross-apply.
        # SPEC §8t.3: only the 40k screen is a gate; 60k is a reported milestone
        # and is never required to resume past it (council L Q4/Q10).
        policy = str((sel or {}).get("n40_policy", P4.N40_POLICY_DEFAULT))
        accept = ("met", "missed") if policy == "diagnostic" else ("pass",)
        want = "record" if policy == "diagnostic" else "PASS"
        for upd, fname, key, label in ((P4.GATE_N40_UPDATE, GATE_N40_FILE, "gate_n40", "40,000 screen"),):
            if stop_at >= upd and update > upd and not _n_gate_recorded(
                    run_dir, sel, fname, key, accept, expect_update=upd):
                abort(f"resume refused: init-N update > {upd:,} with no recorded {label} {want} "
                      f"(SPEC §8q/§8s/§8w); the milestone cannot be assumed to have fired",
                      run_dir, update=update, stop_at=stop_at)
        return
    if stop_at >= P4.GATE20K_UPDATE and update > P4.GATE20K_UPDATE:
        if not gate20k_recorded(run_dir, sel):
            abort("resume refused: update > 20,000 with no recorded 20,000 "
                  "progress-gate PASS (SPEC §4, §8k Q8, council H Q11); the "
                  "gate cannot be assumed to have fired",
                  run_dir, update=update, stop_at=stop_at)


def replay_owed_validation(update, run_dir, sel, best_hist, cfg, args,
                           stop_at, val_at, decode_fn):
    """Council H Q11. A checkpoint at a validation step is written BEFORE that
    step's validation, gate/plateau decision and retention (see the loop), so
    a resume FROM it always owes those three. A checkpoint written later - at
    update U+500 - was written by a process that had completed them, and its
    `sel` and `best_hist` carry their effect, so nothing is owed then.

    Decode only if no val.jsonl record for `update` exists; an existing record
    IS the evaluation of this saved state and is reused. The decision and the
    retention are always replayed, because the resumed `sel`/`best_hist`
    predate them by construction - this is idempotent on disk (`_link_or_copy`
    is a no-op on an existing link) and correct in state.

    Returns (val, sel, best_hist), or (None, sel, best_hist) when nothing is
    owed. A registered STOP raised by the decision exits the process here,
    exactly as it would have in the interrupted run.
    """
    if not should_validate(update, args.val_every, stop_at, val_at, init=(sel or {}).get("init", "A")):
        return None, sel, best_hist
    val = existing_val_record(run_dir, update)
    source = "existing val.jsonl record"
    if val is None:
        val = decode_fn(update)
        source = "fresh decode of the resumed state"
    print(f"RESUME-REPLAY update={update}: the checkpoint owed this step's "
          f"validation decision; replaying it from the {source}", flush=True)
    sel = check_registered_stops(val, update, args, run_dir, sel)
    best_hist = retain(run_dir, update, decision_s(val), cfg, best_hist,
                       eligible=P4.is_eligible(val))
    return val, sel, best_hist


GATE_N40_FILE, GATE_N60_FILE = "GATE_N40", "GATE_N60"


def check_init_n_gates(val, update, run_dir, sel, s, d, eligible):
    """Init N's own progress gates (SPEC §8q/§8s, councils I/K), durable like GATE20K.

    40,000 (spending screen): S <= 22.0296 and D <= 21.348 and empties <= 0.1 %.
        Absolute thresholds only (council K Q3, SPEC §8s): the A-relative
        clauses were deleted prospectively. Full P1 eligibility is NOT required.
    60,000 (DIAGNOSTIC milestone, SPEC §8t.3): S <= 19.5296, D <= 19.35 and every
    P1 guard are measured and recorded; a miss is reported, never terminal.
    Failure of the 40k screen terminates this recipe (no fallback, SPEC §8t.4). No
    plateau rule applies to N. Never writes GATE20K; never reads it.
    """
    tol = P4.GATE20K_REPR_TOL_PP
    policy = str((sel or {}).get("n40_policy", P4.N40_POLICY_DEFAULT))
    d_pp = (d["wer_frozen"] * 100.0) if d is not None else None
    if update == P4.GATE_N40_UPDATE:
        if s is None or d_pp is None:
            # Missing S or D is an INTEGRITY failure, not a performance one, so
            # it is terminal under either policy (council P §8w.2).
            stop("init-N 40k screen not evaluable (missing S or D)", run_dir, update=update)
        # Council K Q3 / SPEC §8s: absolute thresholds ONLY. The A-relative
        # clauses (S <= S_A40 + 2.00, D <= D_A40 + 2.00) are deleted; A40 is
        # neither a prerequisite nor a comparator and is never applied later.
        s_lim, d_lim = P4.GATE_N40_S_ABS_PP, P4.GATE_N40_D_ABS_PP
        ref = None
        ok = (s <= s_lim + tol and d_pp <= d_lim + tol
              and d["empty_rate"] <= P4.GUARD_EMPTY_RATE)
        rec = {"result": "pass" if ok else "fail", "update": int(update), "S_pp": s,
               "S_limit_pp": round(s_lim, 4), "D_pp": round(d_pp, 4), "D_limit_pp": round(d_lim, 4),
               "empty_rate": d["empty_rate"],
               "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec["n40_policy"] = policy
        if policy == "diagnostic":
            # SPEC §8w (council P Q4/Q6): identical conditions, recorded as a
            # milestone. A miss creates no STOP, no terminal ending and no
            # exclusion; the sole terminal qualification is on the selected
            # checkpoint after H. Resume past 40k needs this RECORD to exist -
            # evidence that the milestone completed - not a passing outcome.
            rec["result"] = "met" if ok else "missed"
            rec["former_N40_satisfied"] = bool(ok)
            rec["diagnostic_only"] = True
            # SPEC §8w.2a (the learning floor) was WITHDRAWN on council Q's
            # counterexample before it ever ran: a trajectory reaching 21.000 by
            # 18k, 19.400 by 38k and 21.050 at 40k would have been terminated
            # although a QUALIFYING eligible checkpoint already existed at 38k,
            # and exact equality would terminate a model stable at 18.000. The
            # rule rejected demonstrated intervening improvement, so it did not
            # test what it claimed to. There is no terminal early-performance
            # comparison for init N; the 40k warning below is diagnostic only.
            # Council Q's replacement for the withdrawn floor: a cadence-matched
            # median comparison over equal numbers of observations. It creates no
            # STOP - it flags a trajectory worth investigating (optimizer actually
            # stepping, gradients, train/eval mode, sampler exposure, decoder).
            d40 = _median_d_frozen(run_dir, (36000, 38000, 40000), d_pp)
            d20 = _median_d_frozen(run_dir, (16000, 18000, 20000), None)
            rec["delta_40"] = (round(d40 - d20, 6) if (d40 is not None and d20 is not None) else None)
            rec["delta_40_flag"] = bool(rec["delta_40"] is not None and rec["delta_40"] >= -1e-9)
            rec["delta_40_note"] = ("median(D36k,D38k,D40k) - median(D16k,D18k,D20k); "
                                    ">= 0 flags engineering review, never a stop")
            if run_dir is not None:
                (Path(run_dir) / GATE_N40_FILE).write_text(json.dumps(rec) + "\n", encoding="utf-8")
            if rec["delta_40_flag"]:
                print(f"WARNING delta_40 = {rec['delta_40']} pp >= 0: the 36-40k median is no better "
                      f"than the 16-20k median. Investigate execution (optimizer step, gradients, "
                      f"train/eval mode, sampler exposure, decoder) before reading it as quality. "
                      f"This is NOT a stop (SPEC §8w.2a WITHDRAWN).", flush=True)
            sel["gate_n40"] = rec["result"]
            print(f"MILESTONE_N40 {rec['result'].upper()} (diagnostic, not terminal): "
                  f"S={s} vs {round(s_lim, 4)}; D={round(d_pp, 4)} vs {round(d_lim, 4)}; "
                  f"empties={d['empty_rate']}", flush=True)
        else:
            if not ok:
                stop("init-N 40,000 spending screen failed (SPEC §8q/§8s); this run only",
                     run_dir, ending="terminal", **{k: v for k, v in rec.items() if k != "result"})
            if run_dir is not None:
                (Path(run_dir) / GATE_N40_FILE).write_text(json.dumps(rec) + "\n", encoding="utf-8")
            sel["gate_n40"] = "pass"
            print(f"GATE_N40 PASS S={s} <= {round(s_lim, 4)}; D={round(d_pp, 4)} <= {round(d_lim, 4)}", flush=True)
    if update == P4.GATE_N60_UPDATE:
        # SPEC §8t.3 (council L): a DIAGNOSTIC milestone, not a gate. Every
        # former qualification condition is measured and recorded durably;
        # a miss neither stops N nor excludes it from selection, and no
        # GATE_N60 PASS is ever required to resume past 60,000.
        conds = {"S_le_19.5296": (s is not None and s <= P4.GATE_N60_S_PP + tol),
                 "D_le_19.35": (d_pp is not None and d_pp <= P4.GATE_N60_D_PP + tol),
                 "all_P1_guards": bool(eligible)}
        rec = {"result": "met" if all(conds.values()) else "missed", "update": int(update), "S_pp": s,
               "S_limit_pp": P4.GATE_N60_S_PP, "D_pp": (round(d_pp, 4) if d_pp is not None else None),
               "D_limit_pp": P4.GATE_N60_D_PP, "conditions": conds, "guards": val.get("guards"),
               "diagnostic_only": True, "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if run_dir is not None:
            (Path(run_dir) / GATE_N60_FILE).write_text(json.dumps(rec) + "\n", encoding="utf-8")
        sel["gate_n60"] = rec["result"]
        print(f"MILESTONE_N60 {rec['result'].upper()} (diagnostic, not terminal): S={s} vs 19.5296, "
              f"D={rec['D_pp']} vs 19.35, guards={'pass' if eligible else 'fail'}", flush=True)
    return sel


def check_registered_stops(val, update, args, run_dir, sel):
    """The registered stop rules that key off a validation (SPEC §4, §5, §8k).

    §5 pilot rules (any arm, from the first validation): > 1 % empty outputs,
    or panel D worse than the baseline by more than 5 pp.
    §4/§8k Q8 progress gate at exactly 20,000: S <= S(P1) - 1.5 pp (= 19.530)
      AND D <= D(P1) - 1.0 pp (= 19.35) AND all main-run guards pass, else a
      REGISTERED STOP - "else this recipe TERMINATES".
    §4/§8k Q4 plateau: an anchor initialised at 50,000 to the best ELIGIBLE S
      so far, reset on a >= 0.20 pp CUMULATIVE improvement of that anchor by
      an eligible checkpoint, stop after 10 consecutive non-reset validations,
      never before 70,000.

    `sel` carries the selection/plateau state across validations and is
    check-pointed with the run: {"best_eligible_s", "anchor", "nonreset"}.

    Two things this function refuses to do quietly
    ----------------------------------------------
    1. Compare against anything but the FROZEN the baseline. Council F Q2
       (SPEC §8k) draws the line explicitly - O_ref is the ADMISSION
       comparator and "admission against O_ref does NOT certify continuation
       eligibility" - and the two baselines are only 0.043 pp apart on S, so a
       swapped file would move every guard by one arm's worth of noise without
       looking wrong in any log. `assert_frozen_p1_baseline` checks the
       provenance (init A's sha256, D = 20.35 % at zero updates) instead.
    2. Re-aim the gate. The relative form (S(P1) - 1.5) is evaluated, and the
       result is checked against council F's absolute ruling (19.530 / 19.35)
       before it is applied; a baseline that had drifted would otherwise move
       the gate silently in whichever direction the drift went.
    """
    init = str((sel or {}).get("init", "A"))
    d = val["panels"].get("D")
    # §5 pilot aborts apply to init A only: a fresh Bengali head is EXPECTED to
    # emit empties and a poor D for a while (councils I/J, SPEC §8q/§8s).
    if d is not None and init == "A":
        if d["empty_rate"] > P4.PILOT_ABORT_EMPTY_RATE:
            stop("empty-output rate over 1 % (SPEC §5 pilot abort)", run_dir,
                 update=update, empty_rate=d["empty_rate"])
        delta = (d["wer_frozen"] - P4.P1_DEV_FROZEN) * 100.0
        if delta > P4.PILOT_ABORT_D_PP:
            stop("panel D worse than the baseline by > 5 pp (SPEC §5 pilot abort)",
                 run_dir, update=update, delta_pp=round(delta, 3))
    s = decision_s(val)
    eligible = P4.is_eligible(val)
    base = load_baseline()
    if init == "N":
        sel = check_init_n_gates(val, update, run_dir, sel, s, d, eligible)
    if init == "A" and update == P4.GATE20K_UPDATE:
        try:
            P4.assert_frozen_p1_baseline(base)
        except ValueError as exc:
            stop(f"20k progress gate comparator rejected: {exc}", run_dir,
                 update=update)
        s_p1 = (base or {}).get("S_pp")
        if s is None or s_p1 is None or d is None:
            stop("20k progress gate not evaluable (missing S or baseline); "
                 "SPEC §4 requires the gate, so the run does not continue",
                 run_dir, update=update, S=s, S_p1=s_p1)
        s_limit = s_p1 - P4.GATE20K_S_PP
        d_limit = (P4.P1_DEV_FROZEN * 100.0) - P4.GATE20K_D_PP
        if (abs(s_limit - P4.GATE20K_S_ABS_PP) > P4.GATE20K_ABS_TOL_PP
                or abs(d_limit - P4.GATE20K_D_ABS_PP) > P4.GATE20K_ABS_TOL_PP):
            stop("20k progress gate thresholds do not reproduce council F Q8 "
                 "(SPEC §8k: S <= 19.530, D <= 19.35); the baseline has moved",
                 run_dir, update=update, s_limit=round(s_limit, 4),
                 d_limit=round(d_limit, 4))
        d_pp = d["wer_frozen"] * 100.0
        tol = P4.GATE20K_REPR_TOL_PP        # representation only; see common
        ok = (s <= s_limit + tol and d_pp <= d_limit + tol and eligible)
        if not ok:
            stop("20,000-step progress gate failed (SPEC §4, §8k Q8)", run_dir,
                 update=update, S=s, S_limit_pp=round(s_limit, 4),
                 D_pp=round(d_pp, 4), D_limit_pp=round(d_limit, 4),
                 guards_pass=eligible, guards=val["guards"])
        print(f"GATE20K PASS S={s} <= {round(s_limit, 4)}; "
              f"D={round(d_pp, 4)} <= {round(d_limit, 4)}; guards pass",
              flush=True)
        # Council H Q11: the PASS must be durable, not a print. It is written
        # both as a file and into `sel` (which rides in every later checkpoint)
        # so a resume from beyond 20,000 can prove the gate fired.
        rec = {"result": "pass", "update": int(update), "S_pp": s,
               "S_limit_pp": round(s_limit, 4), "D_pp": round(d_pp, 4),
               "D_limit_pp": round(d_limit, 4), "guards_pass": True,
               "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if run_dir is not None:
            (Path(run_dir) / GATE20K_FILE).write_text(json.dumps(rec) + "\n",
                                                      encoding="utf-8")
        sel["gate20k"] = "pass"

    # ---- selection state and the plateau rule (SPEC §4, §8k Q4) ------------
    # An INELIGIBLE checkpoint never touches the anchor and never resets the
    # counter (council F Q2 condition (i)); it still counts as one non-reset
    # validation, so a run that stops producing eligible checkpoints at all
    # still terminates instead of training to H on guard failures.
    if eligible and s is not None:
        if sel["best_eligible_s"] is None or s < sel["best_eligible_s"]:
            sel["best_eligible_s"] = s
    if init == "A" and update >= P4.PLATEAU_ANCHOR_AT:
        if sel["anchor"] is None:
            if sel["best_eligible_s"] is not None:
                sel["anchor"] = sel["best_eligible_s"]
                sel["nonreset"] = 0
                print(f"PLATEAU anchor initialised at update {update} to "
                      f"S={sel['anchor']} (SPEC §4)", flush=True)
            else:
                sel["nonreset"] += 1
        elif eligible and s is not None and s <= sel["anchor"] - P4.PLATEAU_MIN_PP:
            sel["anchor"] = s
            sel["nonreset"] = 0
        else:
            sel["nonreset"] += 1
        # The earliest-stop floor is enforced on the UPDATE, not inferred from
        # the counter: --val-at can add validations between the registered
        # 2,000-step ones, and ten of those could otherwise reach the count
        # before 70,000 (SPEC §4 "earliest stop 70k").
        if (sel["nonreset"] >= P4.PLATEAU_NONRESET
                and update >= P4.PLATEAU_EARLIEST_STOP):
            stop(f"plateau: {P4.PLATEAU_NONRESET} consecutive validations "
                 f"without a {P4.PLATEAU_MIN_PP} pp cumulative improvement of "
                 f"the anchor (SPEC §4, §8k council F Q4)", run_dir,
                 ending="normal",           # SPEC §8p: convergence, not failure
                 update=update, anchor_S=sel["anchor"], S=s,
                 best_eligible_S=sel["best_eligible_s"],
                 nonreset_validations=sel["nonreset"])
    return sel


# --------------------------------------------------------------------------
# Non-training modes
# --------------------------------------------------------------------------
def build_smoke_manifest(out, rows, source_manifest=None, longest=False):
    """A temp manifest from the old corpus with SPEC §1 targets (CPU only).

    Two shapes, one code path:
      * ``rows`` random-but-deterministic rows (seed 20260905) - the arm_O
        subset the 30-step smoke trains on;
      * ``longest`` - the N longest clips, which is the SPEC §3 G0.7 profiling
        input ("the 21 longest admitted clips at microbatch 12").
    Targets go through train_surface(), i.e. the frozen
    metric_surface(NFC(t), False) the real arm manifests use, so the smoke
    exercises the shipped tokenisation rather than the baseline text.
    """
    src = Path(source_manifest or (REPO / "data2/train_corpus_ready.json"))
    all_rows = load_manifest(src)
    if longest:
        sel = sorted(all_rows, key=lambda r: -float(r["duration"]))[:rows]
    else:
        rng = random.Random(20260905)
        sel = rng.sample(all_rows, min(rows, len(all_rows)))
    kept, dropped = [], 0
    for r in sel:
        t = train_surface(r["text"])
        if not t:
            dropped += 1
            continue
        kept.append({**r, "text": t})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "rows": len(kept),
                      "dropped_empty_target": dropped,
                      "hours": round(sum(r["duration"] for r in kept) / 3600, 3),
                      "max_duration": round(max(r["duration"] for r in kept), 3),
                      "source": str(src)}, indent=1))
    return out


def restore_check(cfg, args, device):
    """SPEC §4 zero-update restoration check, and the baseline the guards need.

    Loads init A through the SAME code path training uses (load_init), decodes
    panel D with the frozen scorer and requires 20.35 % ± 0.05 pp. This is the
    acceptance test for the whole init path: a warm start that silently lost
    BatchNorm buffers, swapped the tokenizer or reinitialised the head would
    fail here at a cost of ~3 GPU-minutes instead of after 150,000 steps.

    Side effect by design: it also records the MEASURED per-source and
    conversational-slice D baselines (and, when those panels exist, M/K/R and
    S(P1)) into experiments/baseline_p1.json. SPEC §2's regression guards are
    comparisons against the baseline, and those per-source numbers are nowhere in
    SPEC - measuring them once, here, is the only honest way to have them.
    """
    model, sha = load_init(cfg, device)
    bn0 = bn_buffer_digest(model)
    out = {"init_sha256": sha, "bn_digest_after_load": bn0,
           "registered_D_frozen": P4.P1_DEV_FROZEN,
           "registered_D_nemo_internal": P4.P1_DEV_NEMO_INTERNAL,
           "tolerance_pp": P4.RESTORATION_TOL * 100,
           "val_limit": args.val_limit, "panels": {}}
    for name, rel in cfg["val"]["panels"].items():
        if args.panels and name not in args.panels.split(","):
            continue
        p = Path(rel)
        p = p if p.is_absolute() else (REPO / p)
        if not p.exists():
            out["panels"][name] = {"missing": str(p)}
            continue
        r = ctc_panel_eval(model, p, batch_size=args.val_batch_size,
                           limit=args.val_limit, per_source=(name == "D"),
                           conversational=(name == "D"))
        r.pop("hyps")
        out["panels"][name] = r
    out["bn_digest_after_eval"] = bn_buffer_digest(model)
    out["bn_buffers_unchanged_by_eval"] = (out["bn_digest_after_eval"] == bn0)

    d = out["panels"].get("D", {})
    if "wer_frozen" in d and not args.val_limit:
        delta = abs(d["wer_frozen"] - P4.P1_DEV_FROZEN)
        out["restoration_delta_pp"] = round(delta * 100, 4)
        out["restoration_pass"] = bool(delta <= P4.RESTORATION_TOL)
    else:
        out["restoration_pass"] = None
        out["restoration_note"] = ("not a gate on a --val-limit subset: the "
                                   "20.35 % figure is the full 5,385-row panel")
    wers = {k: v.get("wer_frozen") for k, v in out["panels"].items()
            if "wer_frozen" in v}
    s, s_meta = selection_score(wers, madasr_admitted=("R" in wers))
    out["S_pp"], out["S_meta"] = s, s_meta
    if args.write_baseline:
        P4.BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
        P4.BASELINE_JSON.write_text(json.dumps(
            {"measured_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
             "init_sha256": sha, "S_pp": s, **out["panels"],
             "D": out["panels"].get("D")}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        out["baseline_written"] = str(P4.BASELINE_JSON)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def profile_longest(cfg, args, device):
    """SPEC §3 G0.7: peak reserved memory at microbatch 12 on the longest clips.

    Builds true worst-case microbatches by hand (12 of the longest clips,
    padded to the longest) and runs a full forward+backward+step for each, so
    the number reported is the one a training update would actually reach -
    including the optimizer state, which only materialises after the first
    step() and is a third of the footprint.

    Reported: peak reserved (the quantity the 16.5 GiB target and the
    watchdog are both stated in) and peak allocated, plus the clip durations, so
    the measurement can be reproduced.
    """
    rows = sorted(load_manifest(args.manifest),
                  key=lambda r: -float(r["duration"]))
    micro = int(cfg["batch"]["microbatch"])
    rows = rows[:max(micro * args.profile_batches, micro)]
    model, _ = load_init(cfg, device)
    model.train()
    groups = make_param_groups(model)
    opt = torch.optim.AdamW(groups, lr=lr_at(0, sched=P4.sched_for(str(cfg.get("init", "A")))),
                            betas=tuple(cfg["optim"]["betas"]),
                            eps=float(cfg["optim"]["eps"]))
    import soundfile as sf
    torch.cuda.reset_peak_memory_stats()
    peaks = []
    # Windows of exactly `micro` clips covering EVERY row, with the last window
    # clamped to the tail rather than dropped: SPEC §3 G0.7 profiles "the 21
    # longest admitted clips at microbatch 12", and 21 is not a multiple of 12.
    # The windows overlap (0-11 and 9-20 for 21 rows) instead of running a
    # 9-clip microbatch, because a short microbatch would understate the peak
    # and the number here is the one the 16.5 GiB target is compared against.
    starts = list(range(0, max(len(rows) - micro, 0) + 1, micro))
    if len(rows) >= micro and starts[-1] != len(rows) - micro:
        starts.append(len(rows) - micro)
    for b in starts:
        chunk = rows[b:b + micro]
        waves = []
        for r in chunk:
            p = Path(r["audio_filepath"])
            w, sr = sf.read(str(p if p.is_absolute() else REPO / p),
                            dtype="float32")
            assert sr == int(cfg["data"]["sample_rate"]), f"{p} at {sr} Hz"
            waves.append(torch.from_numpy(w))
        n = max(len(w) for w in waves)
        sig = torch.zeros(len(waves), n)
        for i, w in enumerate(waves):
            sig[i, :len(w)] = w
        lens = torch.tensor([len(w) for w in waves], dtype=torch.long)
        toks = [model.tokenizer.text_to_ids(r["text"]) for r in chunk]
        tl = torch.tensor([len(t) for t in toks], dtype=torch.long)
        tt = torch.zeros(len(toks), int(tl.max()), dtype=torch.long)
        for i, t in enumerate(toks):
            tt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        batch = {"audio": sig, "audio_lens": lens, "tokens": tt,
                 "token_lens": tl, "ids": [r["audio_filepath"] for r in chunk],
                 "durations": [float(r["duration"]) for r in chunk]}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = forward_microbatch(model, batch, device)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None],
            float(cfg["optim"]["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peaks.append({"window_start_row": b,
                      "durations": [round(float(r["duration"]), 2)
                                    for r in chunk],
                      "peak_reserved_gib": round(
                          torch.cuda.max_memory_reserved() / GIB, 3),
                      "peak_alloc_gib": round(
                          torch.cuda.max_memory_allocated() / GIB, 3)})
    out = {"rows": len(rows), "microbatch": micro,
           "max_duration_s": round(float(rows[0]["duration"]), 3),
           "peak_reserved_gib": max(p["peak_reserved_gib"] for p in peaks),
           "peak_alloc_gib": max(p["peak_alloc_gib"] for p in peaks),
           "target_gib": P4.PROFILE_TARGET_GIB,
           "watchdog_gib": P4.WATCHDOG_GIB, "detail": peaks}
    out["pass"] = out["peak_reserved_gib"] <= P4.PROFILE_TARGET_GIB
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/ctc.yaml"))
    ap.add_argument("--arm",
                    help="pilot arm; selects paths.arms[<arm>] (SPEC §5)")
    ap.add_argument("--manifest", help="override the training manifest (smoke)")
    ap.add_argument("--run-name")
    ap.add_argument("--steps", type=int, default=None,
                    help="stop at exactly this update; 8000 is SPEC §5's pilot")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'auto' for the newest in the run")
    ap.add_argument("--val-every", type=int, default=None,
                    help="default: the registered 2,000 (SPEC §4)")
    ap.add_argument("--val-at", default="",
                    help="extra comma-separated update numbers to validate at")
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None,
                    help="rows per panel; smoke only, never a gate")
    ap.add_argument("--panels", default=None,
                    help="restrict --restore-check to these panels, e.g. 'D'")
    ap.add_argument("--hourly-seconds", type=float, default=None)
    ap.add_argument("--restore-check", action="store_true")
    ap.add_argument("--write-baseline", action="store_true",
                    help="with --restore-check: write baseline_p1.json")
    ap.add_argument("--profile-longest", action="store_true")
    ap.add_argument("--profile-batches", type=int, default=1)
    ap.add_argument("--build-smoke-manifest", default=None)
    ap.add_argument("--build-longest-manifest", default=None)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--pretokenize", type=int, choices=[0, 1], default=None,
                    help="override data.pretokenize (throughput test only; the registered value lives in the config)")
    ap.add_argument("--no-sha-check", action="store_true",
                    help="skip the 463 MB init sha256 (smoke iterations only)")
    return ap.parse_args()


def _policy_or_abort(cfg, run_dir):
    """The registered N40 policy, or a recorded ABORT - never an exception.

    Council R (SPEC 8w.9): raising here escapes main() uncaught, so the
    supervisor classifies it as a crash and re-queues the same bad
    configuration every ~11 minutes forever. abort() writes a terminal record
    the supervisor stops on and a human can see.
    """
    try:
        return P4.n40_policy_of(cfg)
    except Exception as exc:  # noqa: BLE001
        abort(f"N40 policy refused ({type(exc).__name__}): {exc}", run_dir,
              n40_policy=str((cfg or {}).get("n40_policy", "<absent>")))
        raise


def _policy_migration_or_abort(ck_pol, cur_pol, run_dir, update):
    """Authorise (or refuse) a policy change across a resume.

    Council R: printing a migration line is not authorisation, and a bare name
    from an old checkpoint must resolve through the FROZEN legacy map, never
    through whatever generation is current.
    """
    if ck_pol is None:
        return {"from": None, "to": P4.n40_policy_generation(cur_pol),
                "authority": "checkpoint predates policy generations"}
    frm = P4.N40_LEGACY_NAME_GENERATION.get(str(ck_pol))
    to = P4.n40_policy_generation(cur_pol)
    if frm is None:
        abort(f"resume refused: checkpoint carries the unknown N40 policy {ck_pol!r}; "
              f"it cannot be resolved to a registered generation (SPEC 8w.9)",
              run_dir, update=update, checkpoint_policy=str(ck_pol))
    auth = P4.n40_migration_authority(frm, to)
    if auth is None:
        abort(f"resume refused: unauthorised N40 policy migration {frm} -> {to}. "
              f"Only listed transitions are permitted (SPEC 8w.9); amend the "
              f"contract, do not change the config.",
              run_dir, update=update, from_generation=frm, to_generation=to)
    return {"from": frm, "to": to, "authority": auth}


def acquire_run_lock(run_dir):
    """Exclusive, process-lifetime lock on the RUN DIRECTORY.

    Council R (SPEC §8w.9): the per-card supervisor guard is not a lock on the
    run. Two launchers could both pass a fresh-directory check, or resume the
    same run on different cards, interleaving checkpoints and val.jsonl. The
    lock is held for the life of the process and released by the kernel on
    exit, so a crash never strands it.
    """
    import fcntl
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = open(run_dir / ".run.lock", "a+")          # noqa: SIM115 - held for the process
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        holder = fh.read(400).strip() or "(unknown)"
        fh.close()
        return None, holder
    fh.seek(0)
    fh.truncate()
    fh.write(f'{{"pid": {os.getpid()}, "since": "{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}"}}\n')
    fh.flush()
    return fh, None


def fresh_start_refusal(run_dir, resume, env=None):
    """Why a fresh start over `run_dir` must be refused, or None if it is safe.

    Starting at update 0 over a directory that already holds training state
    silently retrains on top of its own checkpoints, steps.jsonl and val.jsonl -
    the worst possible failure for a sole production run with no fallback, and it
    leaves no error to notice. It was reachable whenever a unit's ExecStart was
    written before the first checkpoint existed: run_arm.sh decided --resume at
    unit-WRITE time until 2026-09-08, so a unit created at update 0 carried an
    empty --resume forever and a later restart would have wiped the run.

    Fail-closed. An operator who genuinely wants a new run gives it a new
    --run-name; P4_ALLOW_FRESH_OVER_EXISTING=1 is the deliberate override.
    """
    env = os.environ if env is None else env
    run_dir = Path(run_dir)
    # Council R (SPEC §8w.9): supplying --resume used to skip this function
    # entirely, so a run the protocol had TERMINATED could be resumed straight
    # past its own record. A terminal STOP/ABORT blocks BOTH paths; only the
    # registered recovery (removing the record deliberately) reopens it.
    for name in ("STOP", "ABORT"):
        f = run_dir / name
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rec = {"unparsable": True}
        if name == "STOP" and rec.get("reason") == "graceful_stop":
            continue          # a graceful stop is the designed resume path
        if env.get("P4_OVERRIDE_TERMINAL_STOP") == "1":
            continue
        return {"reason": (f"refusing to start: this run has a terminal {name} record "
                           f"({json.dumps(rec)[:200]}). Training past a terminal outcome "
                           f"needs the registered recovery, not a restart. Set "
                           f"P4_OVERRIDE_TERMINAL_STOP=1 only under that authority."),
                "checkpoints": 0, "newest_checkpoint": "(none)",
                "steps_log": False, "other_evidence": [name], "terminal": name}
    if resume:
        return None
    if env.get("P4_ALLOW_FRESH_OVER_EXISTING") == "1":
        return None
    existing = sorted((run_dir / "ckpt").glob("step_*.pt"))
    steps_log = run_dir / "steps.jsonl"
    has_log = steps_log.exists() and steps_log.stat().st_size > 0
    # Council Q #5: checking only ckpt/ and steps.jsonl let a directory holding
    # validation records, milestone decisions, retained checkpoints or a terminal
    # STOP/ABORT pass as "fresh". Any authoritative evidence that this run name
    # has already been executed is enough to refuse.
    other = [n for n in ("val.jsonl", "GATE20K", "GATE_N40", "GATE_N60",
                         "STOP", "ABORT", "run_meta.json", "IDENTITY.json",
                         "startups.jsonl")
             if (run_dir / n).exists()]
    for d in ("milestones", "best", "preserved"):     # council R: best/ was missed
        if (run_dir / d).is_dir() and any((run_dir / d).iterdir()):
            other.append(f"{d}/")
    if not existing and not has_log and not other:
        return None
    newest = existing[-1].name if existing else "(none)"
    return {"reason": ("refusing to start from update 0 over an existing run: "
                       f"{len(existing)} checkpoint(s) (newest {newest}), steps.jsonl "
                       f"{'present' if has_log else 'absent'}"
                       + (f", plus {', '.join(other)}" if other else "")
                       + f" in {run_dir}. Pass --resume auto to continue it, or use a new "
                         "--run-name to start a different run. Set "
                         "P4_ALLOW_FRESH_OVER_EXISTING=1 only to deliberately overwrite."),
            "checkpoints": len(existing), "newest_checkpoint": newest,
            "steps_log": has_log, "other_evidence": other}


def main():
    args = parse_args()
    anonymize_process()
    # Council R (SPEC §8w.9): configure() used to run BEFORE _RUN_DIR was set,
    # so a configuration refusal was reported by the outer handler with nowhere
    # to write a run-local ABORT - and the supervisor, seeing no record, would
    # re-queue the same bad configuration indefinitely. Establish the error
    # destination from the raw YAML first, then validate inside it.
    _raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.run_name:
        try:
            _rd = REPO / (_raw.get("paths", {}) or {}).get("runs_dir", "experiments/runs") / args.run_name
            _rd.mkdir(parents=True, exist_ok=True)
            _RUN_DIR[0] = _rd
        except Exception:  # noqa: BLE001 - never let the error destination be fatal
            pass
    try:
        cfg = P4.configure(_raw)
    except Exception as exc:  # noqa: BLE001
        abort(f"configuration refused ({type(exc).__name__}): {exc}",
              _RUN_DIR[0], config=str(args.config))
        raise
    if args.pretokenize is not None:
        cfg["data"]["pretokenize"] = bool(args.pretokenize)
    args.val_every = args.val_every if args.val_every is not None \
        else int(cfg["val"]["every"])
    args.val_batch_size = args.val_batch_size or int(cfg["val"]["batch_size"])

    # CPU-only modes first: they must not touch the shared GPU at all.
    if args.build_smoke_manifest:
        build_smoke_manifest(args.build_smoke_manifest, args.rows)
        return 0
    if args.build_longest_manifest:
        build_smoke_manifest(args.build_longest_manifest, args.rows,
                             longest=True)
        return 0

    _allowed = {cfg["gpu_uuid"]}
    if os.environ.get("P4_ALLOW_A6000") == "1":       # owner amendment 2026-09-06: pilots may use the A6000 (20 GB cap)
        _allowed.add("GPU-4754ca84-fd1d-907b-5c18-7c578d001bda")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in _allowed, (
        "CUDA_VISIBLE_DEVICES must be the A5000 UUID from the config (SPEC §0), or the A6000 with P4_ALLOW_A6000=1")
    gpu_name = pin_gpu()
    device = torch.device("cuda")
    os.chdir(REPO)      # manifest audio paths are repo-relative

    if args.restore_check:
        r = restore_check(cfg, args, device)
        return 0 if r.get("restoration_pass") in (True, None) else 1
    if args.profile_longest:
        assert args.manifest, "--profile-longest needs --manifest"
        r = profile_longest(cfg, args, device)
        return 0 if r["pass"] else 1

    assert args.run_name, "--run-name is required for a training run"
    manifest = Path(args.manifest) if args.manifest else \
        (REPO / cfg["paths"]["arms"][args.arm])
    assert manifest.exists(), f"training manifest {manifest} does not exist"
    horizon = int(cfg["sched"]["horizon"])
    stop_at = int(args.steps) if args.steps else horizon
    run_dir = REPO / cfg["paths"]["runs_dir"] / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _RUN_DIR[0] = run_dir

    # A stable identity for this RUN (not its directory name): minted once,
    # reused forever, so milestone and validation evidence can be bound to it.
    _uid_f = run_dir / "RUN_UID"
    if _uid_f.exists():
        run_uid = _uid_f.read_text(encoding="utf-8").strip()
    else:
        run_uid = hashlib.sha256(
            f"{args.run_name}|{time.time_ns()}|{os.getpid()}".encode()).hexdigest()[:16]
        _uid_f.write_text(run_uid + "\n", encoding="utf-8")

    # Council Q #4: --steps was independent of the registered horizon, so a stale
    # unit asking for 150,000 against H=120,000 would have executed 30,000 extra
    # updates at terminal LR (~34.6 GPU-h), past the point where the registered
    # selection and qualification apply. The horizon is the contract.
    if stop_at > horizon:
        abort(f"--steps {stop_at:,} exceeds the registered horizon {horizon:,} "
              f"(SPEC §8v). Training past H is not authorized; amend the contract "
              f"or launch with --steps <= {horizon:,}.",
              run_dir, steps_requested=stop_at, horizon=horizon)

    # A fresh start over a run directory that already holds training state would
    # silently retrain from update 0 on top of its own checkpoints, steps.jsonl
    # and val.jsonl - the worst possible failure for a sole production run with
    # no fallback, and it leaves no error to notice. It is reachable whenever a
    # unit's ExecStart was written before the first checkpoint existed (run_arm.sh
    # decided --resume at unit-WRITE time until 2026-09-08) and the unit is later
    # restarted. Refuse, and say exactly how to proceed. Fail-closed: an operator
    # who really wants a new run gives it a new --run-name.
    # Council R (SPEC §8w.9): take the exclusive run lock BEFORE inspecting or
    # mutating any run state, and before the model is loaded.
    _lock_fh, _holder = acquire_run_lock(run_dir)
    if _lock_fh is None:
        abort(f"refusing to start: another process holds the run lock on {run_dir} "
              f"({_holder}). Two trainers on one run directory interleave "
              f"checkpoints and val.jsonl.", run_dir, lock_holder=_holder)
    _RUN_LOCK[0] = _lock_fh

    refusal = fresh_start_refusal(run_dir, args.resume)
    if refusal is not None:
        abort(refusal["reason"], run_dir, run_name=args.run_name,
              checkpoints=refusal["checkpoints"],
              newest_checkpoint=refusal["newest_checkpoint"],
              steps_log=refusal["steps_log"])

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    global P4LhotseDataset
    P4LhotseDataset = _make_dataset_class()
    model, init_sha = load_init(cfg, device, verify_sha=not args.no_sha_check)
    bn_digest = bn_buffer_digest(model)
    model.train()

    opt = torch.optim.AdamW(make_param_groups(model), lr=lr_at(0, sched=P4.sched_for(str(cfg.get("init", "A")))),
                            betas=tuple(cfg["optim"]["betas"]),
                            eps=float(cfg["optim"]["eps"]))
    loader, sampler = build_loader(cfg, model.tokenizer, manifest)

    update, epoch, resumed, consumed, replay_ids = 0, 0, False, 0, []
    # Selection + plateau state (SPEC §4, §8k Q4). `best_eligible_s` is the
    # best S among checkpoints whose guards PASSED - an ineligible checkpoint
    # can never be selected, so it must not be what "best" means here either.
    sel = {"best_eligible_s": None, "anchor": None, "nonreset": 0,
           "gate20k": None, "init": cfg["init"],
           "gate_n40": None, "gate_n60": None,
           "n40_policy": _policy_or_abort(cfg, run_dir),
           "n40_policy_generation": P4.n40_policy_generation(_policy_or_abort(cfg, run_dir)),
           "run_uid": run_uid}
    best_hist = []
    if args.resume:
        path = (max((run_dir / "ckpt").glob("step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))
                if args.resume == "auto" else Path(args.resume))
        state = torch.load(path, map_location=device, weights_only=False)
        check_resume_state(state, cfg, run_dir)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sampler_load_state_dict(sampler, state["sampler"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        update, epoch, resumed = int(state["update"]), int(state["epoch"]), True
        consumed = int(state.get("consumed_batches", 0))
        replay_ids = list(state.get("last_micro_ids", []))
        ex = state.get("extra", {})
        best_hist, legacy = P4.normalise_history(ex.get("best_hist", []))
        # SPEC §8w: the CONFIG is the registered policy of record, so the N40
        # policy is never inherited from an older checkpoint's `sel` - a
        # checkpoint written under one policy resumed under another must adopt
        # the config's, and the migration must be visible in the log. Every
        # other selection key is restored from the checkpoint as before.
        ck_sel = {k: v for k, v in (ex.get("sel") or {}).items() if k in sel}
        ck_pol = ck_sel.pop("n40_policy", None)
        sel.update(ck_sel)
        mig = _policy_migration_or_abort(ck_pol, sel["n40_policy"], run_dir, update)
        if ck_pol is not None and ck_pol != sel["n40_policy"]:
            print(f"N40 POLICY MIGRATION AUTHORISED: {mig['from']} -> {mig['to']} "
                  f"({mig['authority']}). No training state is changed by this.",
                  flush=True)
        sel["n40_policy_migration"] = mig
        if legacy:
            print(f"WARNING {legacy} checkpoint(s) in best_hist predate the "
                  f"eligibility record and are restored as NOT eligible; the "
                  f"next validation re-establishes the selection (SPEC §8k Q2)",
                  flush=True)
        print(f"resumed update={update} epoch={epoch} consumed={consumed} "
              f"sel={sel} from {path}", flush=True)

    baseline = load_baseline()

    # ---- experiment identity (council R, SPEC 8w.9) -------------------------
    # check_resume_state binds only horizon, init label and effective batch, so
    # a different checkpoint with compatible shapes passed as "this run". Bind
    # the things that actually define the experiment, comparing CONTENT (the
    # manifest is re-hashed, never trusted to a size/mtime cache a `cp -p`
    # satisfies). A checkpoint that predates this is admitted ONCE as legacy and
    # the admission is recorded - refusing it would strand the live run for a
    # binding it could not have carried.
    identity = {
        "run_uid": run_uid,
        "train_manifest": str(manifest),
        "train_manifest_sha256": sha256_file(manifest),
        "tokenizer_sha256": sha256_file(Path(cfg["paths"]["tokenizer_dir"]) / "tokenizer.model")
        if (Path(cfg["paths"]["tokenizer_dir"]) / "tokenizer.model").exists() else None,
        "init": str(cfg.get("init", "A")),
        "init_sha256": init_sha,
        "sched": P4.sched_for(str(cfg.get("init", "A"))),
        "effective_batch": int(cfg["batch"]["effective"]),
        "n40_policy_generation": sel.get("n40_policy_generation"),
        "panels": sorted((cfg.get("val", {}).get("panels") or {})),
    }
    _freeze = REPO / "experiments/corpus/FREEZE.sha256"
    if _freeze.exists():
        reg = {}
        for line in _freeze.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                reg[parts[1]] = parts[0]
        want = reg.get(Path(manifest).name)
        if want and want != identity["train_manifest_sha256"]:
            abort("resume refused: the training manifest's CONTENT does not match its "
                  "registered hash in FREEZE.sha256 (SPEC 8w.9)", run_dir,
                  manifest=str(manifest), registered=want,
                  measured=identity["train_manifest_sha256"])
        identity["train_manifest_registered"] = bool(want)

    _idf = run_dir / "IDENTITY.json"
    if _idf.exists():
        bound = json.loads(_idf.read_text(encoding="utf-8"))
        HARD = ("train_manifest_sha256", "tokenizer_sha256", "init", "init_sha256",
                "effective_batch", "n40_policy_generation")
        diff = [k for k in HARD
                if bound.get(k) is not None and identity.get(k) is not None
                and bound[k] != identity[k]]
        if diff:
            abort("resume refused: this run's bound experiment identity does not match "
                  "the current one on " + ", ".join(diff) + " (SPEC 8w.9). Amend the "
                  "contract or use a new --run-name; do not silently retrain on other data.",
                  run_dir, mismatched=diff,
                  bound={k: bound.get(k) for k in diff},
                  current={k: identity.get(k) for k in diff})
        identity_status = "matched"
    else:
        _idf.write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8")
        identity_status = ("bound-now (checkpoint predates identity binding; admitted as legacy)"
                           if resumed else "bound-now")
    print(f"IDENTITY {identity_status}: run_uid={run_uid} "
          f"manifest={identity['train_manifest_sha256'][:12]}… "
          f"policy={identity['n40_policy_generation']}", flush=True)

    # Append-only startup history: run_meta.json is overwritten every launch, so
    # it is not an execution record. This is (council R, SPEC 8w.9).
    with open(run_dir / "startups.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(), "run_uid": run_uid,
            "resumed_from_update": update, "resume_arg": str(args.resume),
            "identity_status": identity_status,
            "identity": identity,
            "n40_policy": sel.get("n40_policy"),
            "n40_policy_migration": sel.get("n40_policy_migration"),
            "code_sha256": {f: sha256_file(Path(__file__).parent / f)
                            for f in ("train.py", "common.py")},
            "config_sha256": sha256_file(args.config),
            "argv": sys.argv[1:],
        }) + "\n")

    (run_dir / "run_meta.json").write_text(json.dumps({
        "identity": identity, "identity_status": identity_status,
        "run_name": args.run_name, "arm": args.arm,
        "train_manifest": str(manifest),
        "init": str(cfg.get("init", "A")),
        "init_path": (cfg["paths"]["init_N"] if str(cfg.get("init", "A")) == "N" else cfg["paths"]["init"]),
        "init_sha256": init_sha,
        "init_meta": getattr(model, "p4_init_meta", None),
        "sched": P4.sched_for(str(cfg.get("init", "A"))),
        "bn_buffer_digest_at_load": bn_digest,
        "config": args.config, "config_sha256": sha256_file(args.config),
        "code_sha256": {f: sha256_file(Path(__file__).parent / f)
                        for f in ("train.py", "common.py")},
        "gpu": gpu_name, "effective_batch": int(cfg["batch"]["effective"]),
        "microbatch": int(cfg["batch"]["microbatch"]),
        "accumulate": int(cfg["batch"]["accumulate"]),
        "horizon": horizon, "stop_at": stop_at,
        "val_every": args.val_every, "val_limit": args.val_limit,
        "baseline_present": baseline is not None,
        "resumed_from_update": update,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }, indent=1), encoding="utf-8")

    data_iter = BatchStream(loader, sampler, epoch, resumed,
                            consumed=max(consumed - 1, 0) if resumed else 0)
    if resumed and replay_ids:
        _, first = next(data_iter)
        if list(first["ids"]) != list(replay_ids):
            abort("resume handshake failed: the replayed microbatch is not the "
                  "last one consumed before the checkpoint", run_dir,
                  update=update, expected_ids=replay_ids,
                  got_ids=list(first["ids"]))
        print(f"resume handshake OK on {len(replay_ids)} cut ids", flush=True)

    accum = int(cfg["batch"]["accumulate"])
    watchdog = float(cfg["limits"]["watchdog_reserved_gib"])
    hourly = HourlyLogger(run_dir, float(cfg["log"]["hourly_seconds"])
                          if args.hourly_seconds is None
                          else float(args.hourly_seconds))
    steps_fh = open(run_dir / "steps.jsonl", "a", encoding="utf-8")
    val_at = {int(x) for x in args.val_at.split(",") if x.strip()}
    _rv = None
    if resumed:
        # Council H Q11, both halves. First the invariant: a checkpoint past
        # the gate with no recorded PASS is refused outright. Then the replay:
        # a checkpoint AT a validation step still owes that step's decision.
        check_gate_invariant_on_resume(update, stop_at, run_dir, sel)
        _rv, sel, best_hist = replay_owed_validation(
            update, run_dir, sel, best_hist, cfg, args, stop_at, val_at,
            decode_fn=lambda u: validate(model, cfg, u, run_dir, args, baseline))
        if _rv is not None:
            model.train()
    t_run = time.time()
    hour = {"loss": [], "updates": 0, "utts": 0, "audio_s": 0.0, "clips": 0,
            "t0": time.time()}
    last_val, last_ids, stage = {}, list(replay_ids), "startup"
    if _rv is not None:
        last_val = _rv          # the replayed validation is the last one
    peak_res = 0.0

    def write_hourly(reason):
        """SPEC §4's hourly line: loss AND its denominator, LR, updates,
        utterances and audio-hours processed, throughput, grad-clip frequency,
        peak reserved GiB, cumulative GPU-h, ETA and the last validation.

        Flushed on the interval AND at exit, because a run shorter than the
        interval would otherwise produce no hourly.jsonl at all (a defect
        observed on Phase 3's first sub-hour gate leg)."""
        el = time.time() - hour["t0"]
        sph = hour["updates"] / max(el / 3600.0, 1e-9)
        return hourly.log(
            reason=reason, update=update, epoch=epoch,
            mean_loss=(round(float(np.mean(hour["loss"])), 5)
                       if hour["loss"] else None),
            loss_denominator_updates=len(hour["loss"]),
            lr=lr_at(update, sched=P4.sched_for(str(cfg.get("init", "A")))), updates_this_hour=hour["updates"],
            utterances_this_hour=hour["utts"],
            audio_hours_this_hour=round(hour["audio_s"] / 3600.0, 4),
            grad_clip_frequency=(round(hour["clips"] / hour["updates"], 4)
                                 if hour["updates"] else None),
            steps_per_hour=round(sph, 1),
            peak_reserved_gib=round(peak_res, 3),
            cumulative_gpu_hours=round((time.time() - t_run) / 3600.0, 4),
            eta_hours_to_target=round((stop_at - update) / max(sph, 1e-9), 2),
            steps_target=stop_at, horizon=horizon,
            last_validation={"update": last_val.get("update"),
                             "S_pp": last_val.get("S_pp"),
                             "D_frozen": (last_val.get("panels", {})
                                          .get("D", {}).get("wer_frozen")),
                             "D_nemo": (last_val.get("panels", {})
                                        .get("D", {}).get("wer_nemo")),
                             "guards_pass": (last_val.get("guards", {})
                                             .get("pass"))} if last_val else None)

    # Graceful stop (owner rule 2026-09-06 "if one card gets busy, continue on one card"): SIGTERM/SIGINT set a flag; the
    # loop finishes the current update, writes a resumable checkpoint and exits 4 (a SuccessExitStatus of the unit), so
    # `systemctl stop` costs at most one update instead of up to 2,000. The A6000 guard relies on this.
    import signal as _signal
    _stop_flag = {"set": False, "sig": None}
    def _on_stop(signum, frame):
        _stop_flag["set"] = True; _stop_flag["sig"] = signum
    for _sg in (_signal.SIGTERM, _signal.SIGINT):
        _signal.signal(_sg, _on_stop)
    try:
        while update < stop_at:
            if _stop_flag["set"]:
                stage = "graceful_stop"
                save_checkpoint(run_dir / "ckpt" / f"step_{update}.pt", model, opt, sampler, update, epoch, cfg,
                                consumed, last_ids, extra={"sel": sel, "best_hist": best_hist,
                                                           "graceful_stop_signal": int(_stop_flag["sig"] or 0)})
                (run_dir / "STOP").write_text(json.dumps({"reason": "graceful_stop", "signal": int(_stop_flag["sig"] or 0),
                                                          "update": update, "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}))
                print(f"graceful stop at update {update}: resumable checkpoint written, exiting 4", flush=True)
                sys.exit(4)
            stage = "gather"
            micros, n_utt, epoch = gather_microbatches(data_iter, accum,
                                                       run_dir, update)
            consumed = data_iter.consumed
            last_ids = list(micros[-1]["ids"])
            all_ids = [i for b in micros for i in b["ids"]]

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            stage = "update"
            try:
                st = run_update(model, opt, micros, n_utt, cfg, device,
                                run_dir, update)
            except torch.cuda.OutOfMemoryError as exc:
                abort("CUDA OOM", run_dir, stage=stage, update=update,
                      detail=str(exc)[:400])
            torch.cuda.synchronize()
            wall = time.time() - t0
            update += 1
            peak_res = torch.cuda.max_memory_reserved() / GIB

            rec = {"update": update, "epoch": epoch, "consumed_batches": consumed,
                   "lr": st["lr"], "loss": round(st["loss"], 6),
                   "grad_norm": round(st["grad_norm"], 6),
                   "clipped": st["clipped"], "n_utt": n_utt,
                   "n_micro": len(micros), "min_microbatch": st["min_microbatch"],
                   "n_tokens": st["n_tokens"],
                   "sum_seconds": round(st["sum_seconds"], 3),
                   "max_seconds": round(st["max_seconds"], 3),
                   "peak_reserved_gib": round(peak_res, 3),
                   "peak_alloc_gib": round(
                       torch.cuda.max_memory_allocated() / GIB, 3),
                   "wall_s": round(wall, 4), "ids": all_ids}
            if update % int(cfg["log"]["steps_every"]) == 0:
                steps_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                steps_fh.flush()
            hour["loss"].append(st["loss"])
            hour["updates"] += 1
            hour["utts"] += n_utt
            hour["audio_s"] += st["sum_seconds"]
            hour["clips"] += int(st["clipped"])

            # SPEC §4/§8i watchdog: stop THIS process cleanly at the
            # registered reserved-memory ceiling (13.0 GiB, from the config).
            # max_memory_reserved() is per-process, so nothing here can observe
            # or act on a co-tenant's allocation - the run stops itself and
            # never touches another PID.
            if peak_res > watchdog:
                stage = "watchdog"
                save_checkpoint(run_dir / "ckpt" / f"step_{update}.pt", model,
                                opt, sampler, update, epoch, cfg, consumed,
                                last_ids, extra={"watchdog": True,
                                                 "sel": sel,
                                                 "best_hist": best_hist})
                write_hourly("watchdog")
                stop(f"reserved memory over the {watchdog} GiB watchdog "
                     f"(SPEC §4, §8i); checkpoint written, exiting cleanly",
                     run_dir,
                     update=update, peak_reserved_gib=round(peak_res, 3))

            # A final validation at the step target is forced only when
            # periodic validation is on at all: `--val-every 0` means "no
            # decode in this process", which is what the resume-equivalence
            # drill needs (a validation between the two legs would change the
            # RNG stream and the drill would be testing the wrong thing).
            do_val = should_validate(update, args.val_every, stop_at, val_at, init=sel.get("init", "A"))
            # Robustness (owner 2026-09-06): a resumable checkpoint every ckpt.every updates (rotated by retain's keep_last)
            # so a crash, a killed terminal, or a card hand-over never loses more than ckpt.every updates.
            periodic = int(cfg["ckpt"].get("every", 0) or 0)
            if do_val or update in tuple(int(x) for x in cfg["ckpt"]["milestones"]) or (periodic and update % periodic == 0):
                stage = "checkpoint"
                save_checkpoint(run_dir / "ckpt" / f"step_{update}.pt", model,
                                opt, sampler, update, epoch, cfg, consumed,
                                last_ids, extra={"loss": st["loss"],
                                                 "sel": sel,
                                                 "best_hist": best_hist})
            if do_val:
                stage = "validate"
                last_val = validate(model, cfg, update, run_dir, args, baseline)
                model.train()
                sel = check_registered_stops(last_val, update, args,
                                             run_dir, sel)
                best_hist = retain(run_dir, update, decision_s(last_val), cfg,
                                   best_hist, eligible=P4.is_eligible(last_val))
            elif update in tuple(int(x) for x in cfg["ckpt"]["milestones"]) or (periodic and update % periodic == 0):
                best_hist = retain(run_dir, update, None, cfg, best_hist)

            if hourly.due():
                stage = "hourly"
                write_hourly("interval")
                hour = {"loss": [], "updates": 0, "utts": 0, "audio_s": 0.0,
                        "clips": 0, "t0": time.time()}

        stage = "final-checkpoint"
        save_checkpoint(run_dir / "ckpt" / f"step_{update}.pt", model, opt,
                        sampler, update, epoch, cfg, data_iter.consumed,
                        last_ids, extra={"final": True, "sel": sel,
                                         "best_hist": best_hist})
        # SPEC §5: the pilot end is an immutable milestone the main run resumes
        # from, so it is protected here even if it was not a validation step.
        best_hist = retain(run_dir, update, None, cfg, best_hist)
        if update == P4.PILOT_STEPS:
            _link_or_copy(run_dir / "ckpt" / f"step_{update}.pt",
                          run_dir / "milestones" / f"step_{update}.pt")
    except torch.cuda.OutOfMemoryError as exc:
        abort("CUDA OOM", run_dir, stage=stage, update=update,
              detail=str(exc)[:400])
    except SystemExit:                                # graceful stop (exit 4) / clean completion: not an abort
        raise
    except BaseException as exc:                     # noqa: BLE001
        import traceback as _tb
        abort(f"unhandled exception ({type(exc).__name__})", run_dir,
              stage=stage, update=update, detail=str(exc)[:400],
              traceback=_tb.format_exc()[-2000:])

    write_hourly("exit")
    steps_fh.close()
    del data_iter, loader
    print(f"TRAIN-EXIT clean update={update} "
          f"elapsed_h={round((time.time() - t_run) / 3600.0, 4)}", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as _exc:                    # noqa: BLE001
        import traceback as _tb
        abort(f"unhandled exception in main ({type(_exc).__name__})",
              _RUN_DIR[0], detail=str(_exc)[:400],
              traceback=_tb.format_exc()[-2000:])
