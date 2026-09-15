#!/usr/bin/env python
"""Phase-3 TDT trainer: custom update loop, Lhotse microbatches, A5000 only.

Implements SPEC sections 2-4 on a NeMo 2.7.3 ``EncDecRNNTBPEModel`` restored
from one of the REGISTERED inits (``experiments/inits/INITS.json``; the
``--init`` sha256 is checked against that registry before any GPU work, so an
unregistered copy of a .nemo cannot become a pilot arm).
The init is NEVER rebuilt here and ``change_vocabulary`` is never called:
it silently drops the TDT loss and leaves a model that trains to a plausible
number against the wrong objective.

Why a hand-written loop instead of ``Trainer.fit``:

  * Lhotse gives DYNAMIC microbatches (<= 100 audio-seconds AND <= 8 clips),
    so a microbatch holds anywhere from 3 to 8 utterances. An effective
    batch of "16 utterances" therefore cannot be a fixed accumulation count;
    it has to be decided at run time by counting utterances and weighting
    each microbatch loss by n_i / N_accum (SPEC section 3).
  * The freeze schedule is not a callback: for updates 1-1,000 the encoder is
    frozen AND in eval mode AND SpecAugment is off, and BatchNorm running
    stats stay frozen for the entire run - which means re-asserting eval mode
    on every BN module after every ``model.train()``, every single step.
  * The abort rules (SPEC section 4) must kill the process, not raise into a
    framework that retries. Every abort is ``sys.exit(3)`` with a reason.

Usage (a gate run):

    CUDA_VISIBLE_DEVICES=GPU-3ce8e4c2-3bae-8744-eeec-70e8a0437567 \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv/bin/python bnasr/train_core.py \\
        --init experiments/inits/init_v3_reinit.nemo \\
        --run-name gate2_v3 --max-updates 1000 --val-every 500

Outputs under ``experiments/<run_name>/``: ``steps.jsonl`` (one line per
optimizer update), ``val.jsonl``, ``hourly.jsonl``, ``nvml.csv`` (1 Hz
sidecar), ``ckpt/`` (rolling) and ``protected/`` (never rotated).
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path

# Both of these must happen before torch/NeMo touch the driver.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bnasr.scoring import (REPO, SCHED, HourlyLogger,  # noqa: E402
                             anonymize_process, assert_registered_init,
                             configure_data, configure_optim,
                             configure_schedules, dev_eval, lr_at,
                             make_param_groups, pin_gpu, sha256_file)

anonymize_process()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

GIB = float(1 << 30)
ABORT_EXIT = 3          # SPEC section 4: abort code, no in-process retry

# Set as soon as the run directory exists, so the top-level handler can still
# write an ABORT file for a failure that happens outside the training loop.
_RUN_DIR = [None]


def hard_exit(code):
    """Flush everything, then leave the interpreter immediately.

    A plain ``sys.exit`` does not reliably end this process: the Lhotse
    DataLoader's worker pool, the pinned-memory thread and numba's CUDA
    context all register shutdown work, and in practice the process was
    observed sitting alive for tens of minutes after "TRAIN-EXIT clean",
    still holding several GiB of the A5000 - on a SHARED box, where that is
    a co-tenant's problem as much as ours (three such zombies together took
    the card to an OOM during development). Once the checkpoint is on disk
    and stdout is flushed there is nothing left to unwind that matters, so
    the exit status is made deterministic instead of hopeful.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# --------------------------------------------------------------------------
# Abort
# --------------------------------------------------------------------------
def abort(reason, run_dir=None, **ctx):
    """Stop the process now, loudly, with the reason on disk.

    SPEC section 4 forbids an in-process retry: an OOM or an envelope breach
    means the measured envelope was wrong, and the only honest response is to
    stop and re-measure. Retrying would produce a run whose peak-memory record
    no longer describes what actually happened.
    """
    payload = {"abort": reason, "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                          time.gmtime()), **ctx}
    line = json.dumps(payload, ensure_ascii=False, default=str)
    print("ABORT " + line, flush=True)
    if run_dir is not None:
        with open(Path(run_dir) / "ABORT", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    hard_exit(ABORT_EXIT)


# --------------------------------------------------------------------------
# NVML (pynvml is not installed in this venv; nvidia-smi is the fallback)
# --------------------------------------------------------------------------
class NvmlProbe:
    """used/free MiB for the pinned card, via pynvml if present else smi.

    The 1 Hz sidecar owns the sampling; per-step logging reads the sidecar's
    last sample instead of shelling out again. A subprocess per training step
    would be ~40 ms on a ~800 ms step - 5% of the entire GPU budget spent on
    telemetry - and the whole point of the 1 Hz CSV is that a 1-second
    resolution is enough to see a memory excursion.
    """

    def __init__(self, uuid):
        self.uuid = uuid
        self.handle = None
        self.backend = "nvidia-smi"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
            self.backend = "pynvml"
        except Exception:
            self.pynvml = None
        self.last = {"used_mib": None, "free_mib": None, "t": 0.0}

    def sample(self):
        try:
            if self.handle is not None:
                m = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used, free = m.used // (1 << 20), m.free // (1 << 20)
            else:
                import subprocess
                out = subprocess.run(
                    ["nvidia-smi", "-i", self.uuid, "--format=csv,noheader,nounits",
                     "--query-gpu=memory.used,memory.free"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                used, free = (int(x) for x in out.split(",")[:2])
            self.last = {"used_mib": used, "free_mib": free, "t": time.time()}
        except Exception as exc:                       # telemetry never aborts
            self.last = {"used_mib": None, "free_mib": None, "t": time.time(),
                         "err": str(exc)[:120]}
        return self.last


def nvml_sidecar(probe, path, hz, stop_evt):
    """1 Hz NVML sampler -> CSV (SPEC section 4).

    Runs in a daemon thread and writes on a fixed wall-clock grid so the CSV
    can be aligned against steps.jsonl by timestamp even when a step is slow.
    """
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["unix_time", "used_mib", "free_mib"])
        period = 1.0 / float(hz)
        nxt = time.time()
        while not stop_evt.is_set():
            s = probe.sample()
            w.writerow([round(s["t"], 3), s["used_mib"], s["free_mib"]])
            fh.flush()
            nxt += period
            stop_evt.wait(max(0.0, nxt - time.time()))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_loader(cfg, tokenizer):
    """Lhotse train loader with SPEC section 2's keys, verbatim.

    Returns (dataloader, sampler). For a plain JSON manifest NeMo keeps the
    sampler in THIS process (map-style dataset), which is what makes
    ``sampler.state_dict()`` a real resume point rather than a lie: with
    tarred/Shar data the sampler lives inside the worker and its state is
    unreachable from the training loop.
    """
    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
    ds_cfg = dict(cfg["train_ds"])
    # The registered SPEC section 2 plan, asserted rather than defaulted. A
    # `setdefault` would not do: NeMo 2.7.3 defaults `concurrent_bucketing` to
    # True (a background bucket-filling thread that Lhotse documents as
    # non-deterministic), and a silent default is precisely what made the
    # Gate-1 audit describe a microbatch plan the trainer never drew.
    configure_data(ds_cfg)
    assert ds_cfg.get("concurrent_bucketing") is False, (
        "train_ds.concurrent_bucketing must be explicitly false; NeMo defaults "
        "it True and Lhotse then samples non-deterministically")
    ds_cfg["manifest_filepath"] = str(REPO / ds_cfg["manifest_filepath"])
    loader = get_lhotse_dataloader_from_config(
        ds_cfg, global_rank=0, world_size=1,
        dataset=P3LhotseDataset(tokenizer), tokenizer=tokenizer)
    return loader, loader.sampler


def _make_dataset_class():
    from nemo.collections.asr.data.audio_to_text_lhotse import \
        LhotseSpeechToTextBpeDataset

    class _P3(LhotseSpeechToTextBpeDataset):
        """Adds cut ids, per-cut durations and a SKIPPED-ROW count.

        NeMo's Lhotse dataset loads audio with ``AudioSamples(
        fault_tolerant=True)``, which DROPS a cut whose file is unreadable
        and says nothing. SPEC section 4 makes any skipped row an abort, so
        the drop has to be observable: we count what the sampler handed us
        against what survived I/O and ship the difference with the batch.
        """

        def __init__(self, tokenizer):
            super().__init__(tokenizer, return_cuts=True)

        def __getitem__(self, cuts):
            n_in = len(cuts)
            audio, audio_lens, tokens, token_lens, out = super().__getitem__(cuts)
            return {
                "audio": audio, "audio_lens": audio_lens,
                "tokens": tokens, "token_lens": token_lens,
                "ids": [c.id for c in out],
                "durations": [float(c.duration) for c in out],
                "n_skipped": n_in - len(out),
            }
    return _P3


P3LhotseDataset = None      # bound in main(), after NeMo is importable


class BatchStream:
    """Own the epoch loop, and COUNT what the training loop actually consumed.

    Two jobs, and the second one is the reason this is a class rather than the
    generator it used to be.

    Epochs: a Lhotse sampler over a plain JSON manifest is FINITE (one epoch),
    unlike the tarred/Shar case SPEC section 2 describes as infinite. Rather
    than fake infinity inside the sampler we re-enter it with a bumped epoch,
    which is also what gives each epoch a different shuffle. On resume the
    first pass must NOT call set_epoch: load_state_dict has already
    fast-forwarded the sampler inside the current epoch, and set_epoch would
    throw that away.

    Counting: the DataLoader runs 8 workers with torch's default
    prefetch_factor 2, so torch's ``_MultiProcessingDataLoaderIter`` pulls
    ``num_workers * prefetch_factor`` CutSets out of the in-process Lhotse
    sampler AHEAD of what this loop has consumed. The sampler's own diagnostics
    therefore count ~16 microbatches (~100 utterances) more than were ever
    trained on, and Lhotse's ``_fast_forward()`` would skip exactly those on
    resume - silently dropping rows from the epoch, which SPEC sections 0/2
    forbid, and breaking Gate 3's equal-example-exposure premise. Measured:
    smoke12 vs smoke12r disagreed on every cut id from the resume point on.

    ``consumed`` is the count for the CURRENT epoch as seen by the TRAINING
    loop, so checkpointing it (rather than trusting the sampler's lead) makes
    the resume point exact and independent of any prefetch tuning.
    """

    def __init__(self, loader, sampler, start_epoch=0, resumed=False,
                 consumed=0):
        self.loader, self.sampler = loader, sampler
        self.epoch = int(start_epoch)
        self.consumed = int(consumed)
        self._resumed = bool(resumed)
        self._gen = self._iterate()

    def _iterate(self):
        first = True
        while True:
            if not (first and self._resumed):
                self.sampler.set_epoch(self.epoch)
                self.consumed = 0
            first = False
            for batch in self.loader:
                self.consumed += 1
                yield self.epoch, batch
            self.epoch += 1
            self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)


def save_sampler_snapshot(sampler, consumed):
    """Alias used by selftest 5, so the test exercises the SHIPPED code path."""
    return sampler_state_dict(sampler, consumed)


def sampler_state_dict(sampler, consumed):
    """``DynamicBucketingSampler.state_dict()`` rewound to `consumed` batches.

    Two corrections to the raw call:

    1. Lhotse asserts ``constraint is None`` in state_dict() because a custom
       constraint is not serialisable - but NeMo ALWAYS passes a TimeConstraint
       built from batch_duration/batch_size/quadratic_duration, so the assert
       fires on every NeMo Lhotse run. The constraint is reconstructed from the
       config on the next launch and lives on the sampler object, not in the
       checkpoint, so hiding it for the duration of the call loses nothing.

    2. The per-epoch batch counter inside ``diagnostics`` is what
       ``_fast_forward()`` replays on resume, and the sampler has run AHEAD of
       the training loop by the DataLoader's prefetch lead. It is overwritten
       here with the training loop's own count, minus one: the checkpoint also
       records the cut ids of the last consumed microbatch, so a resume replays
       that microbatch first and asserts the ids match before discarding it.
       That turns "the sampler probably came back to the right place" into a
       checked property of every restart.
    """
    held = sampler.constraint
    try:
        sampler.constraint = None
        sd = copy.deepcopy(sampler.state_dict())
    finally:
        sampler.constraint = held
    diag = sd["diagnostics"]
    ep = diag["current_epoch"]
    stats = diag["stats_per_epoch"][ep]
    stats["kept_batches"] = max(int(consumed) - 1, 0)
    stats["discarded_batches"] = 0
    return sd


def sampler_load_state_dict(sampler, sd):
    """Restore + fast-forward. Mutates its input, so hand it a copy."""
    sampler.load_state_dict(copy.deepcopy(sd))


# --------------------------------------------------------------------------
# Model contract (SPEC section 1) and phase control (SPEC section 3)
# --------------------------------------------------------------------------
def assert_model_contract(model, cfg):
    """Every SPEC section 1 assertion, re-checked at load time.

    build_inits.py already checked these when it wrote the .nemo, but a
    checkpoint is a file on a shared box and a 100-GPU-hour run is too
    expensive to start against a model that is one silent config edit away
    from a different objective. The dropout rates are checked too: SPEC
    section 3 registers encoder 0.1 (restored at update 1,001) and
    decoder/joint 0.2 throughout, and those live in the .nemo config rather
    than anywhere this trainer sets them.
    """
    m = cfg["model"]
    tok_v = model.tokenizer.vocab_size
    assert tok_v == m["vocab_size"], f"tokenizer vocab {tok_v}"
    assert model.decoder.vocab_size == m["vocab_size"], "decoder vocab"
    assert model.decoder.blank_idx == m["blank_idx"], "decoder blank_idx"
    assert model.joint.num_extra_outputs == m["num_extra_outputs"], "extra out"
    assert model.joint.num_classes_with_blank == m["num_classes_with_blank"], \
        "joint classes"
    loss = model.loss._loss
    assert type(loss).__name__ == m["loss_class"], type(loss).__name__
    assert list(loss.durations) == list(m["durations"]), list(loss.durations)
    assert abs(float(loss.sigma) - float(m["sigma"])) < 1e-12, loss.sigma
    assert abs(float(loss.omega) - float(m["omega"])) < 1e-12, loss.omega
    assert str(model.cfg.get("rnnt_reduction")) == m["rnnt_reduction"], \
        model.cfg.get("rnnt_reduction")
    assert str(model.loss.reduction) == m["rnnt_reduction"], model.loss.reduction

    assert float(model.cfg.encoder.dropout) == float(m["encoder_dropout"])
    dropouts = [mod.p for mod in list(model.decoder.modules())
                + list(model.joint.modules()) if isinstance(mod, nn.Dropout)]
    assert dropouts and all(abs(p - float(m["head_dropout"])) < 1e-9
                            for p in dropouts), dropouts

    d = cfg["model"]["decoding"]
    assert str(model.cfg.decoding.strategy) == d["strategy"], "strategy"
    assert list(model.cfg.decoding.durations) == list(m["durations"])
    assert int(model.cfg.decoding.greedy.max_symbols) == d["max_symbols"]


def enforce_decoding(model, cfg):
    """Pin the greedy-batch decoder to the registered flags and verify them.

    The shipped .nemo config carries only `greedy.max_symbols`, so
    `loop_labels` and `use_cuda_graph_decoder` fall through to NeMo's
    defaults - and the default for CUDA graphs is TRUE, while SPEC section 1
    registers FALSE. Asserting on the config would therefore pass while the
    live decoder did something else, so we write the two flags in and rebuild
    the decoding object, then assert on the OBJECT: `loop_labels` shows up as
    the bound `_greedy_decode` implementation and CUDA graphs as
    `decoding_computer.allow_cuda_graphs`.

    This is `change_decoding_strategy`, NOT `change_vocabulary` - the latter
    is the one that silently drops the TDT loss and is never called anywhere
    in Phase 3.
    """
    from omegaconf import open_dict
    d = cfg["model"]["decoding"]
    dec_cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(dec_cfg):
        dec_cfg.greedy.loop_labels = bool(d["loop_labels"])
        dec_cfg.greedy.use_cuda_graph_decoder = bool(d["use_cuda_graph_decoder"])
    model.change_decoding_strategy(dec_cfg, verbose=False)

    inner = model.decoding.decoding
    assert type(inner).__name__ == "GreedyBatchedTDTInfer", type(inner).__name__
    assert inner._greedy_decode.__name__ == \
        "_greedy_decode_blank_as_pad_loop_labels", inner._greedy_decode.__name__
    assert inner.decoding_computer is not None, "label-looping computer missing"
    assert bool(inner.decoding_computer.allow_cuda_graphs) is \
        bool(d["use_cuda_graph_decoder"]), "cuda graph decoder flag"
    assert int(inner.max_symbols) == int(d["max_symbols"])
    assert list(inner.durations) == list(cfg["model"]["durations"])


def batchnorm_modules(model):
    return [mod for mod in model.modules()
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.BatchNorm3d, nn.SyncBatchNorm))]


def freeze_batchnorm(bns):
    """Put every BatchNorm in eval mode and stop its running stats moving.

    SPEC section 3: BN running stats stay frozen for the WHOLE run, not just
    the freeze phase. FastConformer's conv module is BatchNorm1d, and the
    Bengali corpus is 984 h of a different acoustic distribution from the
    English pretraining - letting the running stats chase it would silently
    change what the pretrained encoder computes at inference, which is the
    one thing a frozen-encoder warm start is supposed to preserve.

    momentum=0 belt-and-braces: eval() alone stops the update, but a stray
    ``model.train()`` anywhere in NeMo would re-arm it, and momentum 0 makes
    that harmless. This must run AFTER every ``.train()``, every step.
    """
    for bn in bns:
        bn.eval()
        bn.momentum = 0.0


def set_phase(model, bns, specaug, frozen):
    """Apply the freeze / unfreeze phase for the update about to be made.

    frozen  (updates 1-1,000): encoder requires_grad False, encoder in eval
            mode (so encoder dropout is off too), SpecAugment removed.
    unfrozen(update 1,001+)  : encoder trainable and in train mode, so its
            registered 0.1 dropout is live again, SpecAugment attached.
    BatchNorm is re-frozen last in both cases.
    """
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad = not frozen
    if frozen:
        model.encoder.eval()
        model.spec_augmentation = None
    else:
        model.encoder.train()
        model.spec_augmentation = specaug
    freeze_batchnorm(bns)


# --------------------------------------------------------------------------
# One optimizer update
# --------------------------------------------------------------------------
def forward_microbatch(model, batch, device):
    """encoder -> prediction network -> FUSED joint+loss for one microbatch.

    The fused joint (``set_fuse_loss_wer(True, loss, wer)`` with
    fused_batch_size 4) never materialises the full [B, T, U, V] lattice: it
    walks the microbatch 4 utterances at a time, computing joint+loss and
    freeing each slice. At B=8, T=375, U=165, V=1030 the unfused lattice
    alone would be ~250 GiB, so this is not an optimisation, it is the only
    way the step exists at all.
    """
    sig = batch["audio"].to(device, non_blocking=True)
    sig_len = batch["audio_lens"].to(device, non_blocking=True)
    tok = batch["tokens"].to(device, non_blocking=True)
    tok_len = batch["token_lens"].to(device, non_blocking=True)
    enc, enc_len = model.forward(input_signal=sig, input_signal_length=sig_len)
    dec, _, _ = model.decoder(targets=tok, target_length=tok_len)
    out = model.joint(encoder_outputs=enc, decoder_outputs=dec,
                      encoder_lengths=enc_len, transcripts=tok,
                      transcript_lengths=tok_len, compute_wer=False)
    loss = out[0] if isinstance(out, tuple) else out
    return loss, enc_len.detach(), tok_len.detach()


def gather_microbatches(data_iter, min_utts, cfg, run_dir, update):
    """Pull Lhotse microbatches until >= `min_utts` utterances are in hand.

    The whole group is collected BEFORE any forward pass because the weight
    n_i / N_accum needs N_accum, which is only known once the last microbatch
    has been drawn. What is held is raw waveform on the CPU - about 6 MB per
    microbatch at 100 audio-seconds - so holding two or three costs nothing,
    while holding their ACTIVATIONS would cost the run.

    Sampler-envelope violations (SPEC section 4) are caught here, at the
    point where the sampler's promise is observable - and they are checked in
    PADDED seconds, which is the quantity SPEC section 2 says in bold drives
    memory (B x longest, because Lhotse pads every member of a microbatch up
    to the longest one), not the raw sum of durations. The distinction is not
    academic on this corpus: the worst microbatch the Gate-1a audit found had
    raw 67.7 s (comfortably under the 100 s budget, so a raw-seconds guard can
    never fire) but 114.7 PADDED audio-seconds and a 279 s Lhotse budget. The
    raw check is kept as well - it is cheap and it is the sampler's own stated
    contract - but the padded ceiling is the one that guards the failure mode.

    Both quantities are returned per microbatch so the envelope record in
    steps.jsonl is in padded seconds, as SPEC section 2 requires.
    """
    ceiling = float(cfg["limits"]["padded_audio_ceiling_s"])
    quad = float(cfg["train_ds"]["quadratic_duration"])
    micros, n_utt, epoch = [], 0, None
    while n_utt < min_utts:
        epoch, batch = next(data_iter)
        if batch["n_skipped"]:
            abort("skipped row(s) in a microbatch", run_dir, update=update,
                  n_skipped=int(batch["n_skipped"]), ids=batch["ids"][:4])
        b = len(batch["ids"])
        secs = float(sum(batch["durations"]))
        d_max = float(max(batch["durations"]))
        # Lhotse's own TimeConstraint measure, so the number logged is directly
        # comparable with batch_duration/quadratic_duration in the config.
        padded_audio = b * d_max
        padded_budget = b * (d_max + d_max * d_max / quad)
        batch["padded_audio_s"] = padded_audio
        batch["padded_budget_s"] = padded_budget
        if b > int(cfg["train_ds"]["batch_size"]):
            abort("sampler-envelope violation: B over batch_size", run_dir,
                  update=update, B=b)
        if b > 1 and secs > float(cfg["train_ds"]["batch_duration"]) + 1e-6:
            abort("sampler-envelope violation: seconds over batch_duration",
                  run_dir, update=update, B=b, seconds=secs)
        if b > 1 and padded_audio > ceiling + 1e-6:
            abort("sampler-envelope violation: PADDED seconds over the "
                  "measured-safe ceiling", run_dir, update=update, B=b,
                  padded_audio_s=round(padded_audio, 3),
                  padded_budget_s=round(padded_budget, 3),
                  ceiling_s=ceiling, sec_max=round(d_max, 3),
                  durations=[round(float(x), 3) for x in batch["durations"]],
                  ids=batch["ids"][:8])
        micros.append(batch)
        n_utt += b
    return micros, n_utt, epoch


def run_update(model, opt, micros, n_utt, cfg, device, run_dir, update):
    """One optimizer update over the collected microbatches.

    Weighting (SPEC section 3): each microbatch loss is scaled by
    n_i / N_accum before backward, so the accumulated gradient is the
    gradient of a weighted mean over the update's utterances rather than a
    sum that would make the effective LR depend on how the sampler happened
    to split the group.

    `accum_weight_mode` selects the weight's denominator:
      utterances (SPEC-literal) - n_i / N_accum with n counted in clips;
      tokens                    - t_i / T_accum with t the reference token
                                  count, which under the registered
                                  `mean_volume` reduction (sum losses / sum
                                  target lengths) makes the accumulated
                                  gradient ALGEBRAICALLY identical to one
                                  unpartitioned batch.
    """
    mode = cfg["optim"].get("accum_weight_mode", "utterances")
    weights_den = n_utt
    if mode == "tokens":
        weights_den = sum(int(b["token_lens"].sum()) for b in micros)

    stats = {"micro": [], "loss_terms": [], "T_max": 0, "U_max": 0,
             "B_max": 0, "sum_seconds": 0.0, "max_seconds": 0.0,
             "lattice_cells": 0, "lattice_cells_padded": 0,
             "max_padded_audio_s": 0.0, "max_padded_budget_s": 0.0}
    loss_value = 0.0
    for batch in micros:
        n_i = (len(batch["ids"]) if mode != "tokens"
               else int(batch["token_lens"].sum()))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, enc_len, tok_len = forward_microbatch(model, batch, device)
        # Out of autocast on purpose: the scale-and-backward is fp32, and the
        # master parameters and Adam state are fp32 throughout (bf16 is only
        # the compute dtype inside the forward).
        loss_f = loss.float()
        if not torch.isfinite(loss_f):
            abort("non-finite microbatch loss", run_dir, update=update,
                  loss=float(loss_f), ids=batch["ids"][:4])
        (loss_f * (n_i / weights_den)).backward()

        t = enc_len.to(torch.int64)
        u = tok_len.to(torch.int64)
        cells = int((t * (u + 1)).sum())
        secs = [float(x) for x in batch["durations"]]
        stats["micro"].append({
            "ids": list(batch["ids"]), "B": len(secs),
            "T_max": int(t.max()), "U_max": int(u.max()),
            "sec_sum": round(sum(secs), 3), "sec_max": round(max(secs), 3),
            # SPEC section 2: every envelope check reports PADDED seconds.
            "padded_audio_s": round(float(batch["padded_audio_s"]), 3),
            "padded_budget_s": round(float(batch["padded_budget_s"]), 3),
            "cells": cells, "weight": round(n_i / weights_den, 6)})
        stats["loss_terms"].append(float(loss_f))
        stats["T_max"] = max(stats["T_max"], int(t.max()))
        stats["U_max"] = max(stats["U_max"], int(u.max()))
        stats["B_max"] = max(stats["B_max"], len(secs))
        stats["sum_seconds"] += sum(secs)
        stats["max_seconds"] = max(stats["max_seconds"], max(secs))
        stats["lattice_cells"] += cells
        stats["lattice_cells_padded"] += len(secs) * int(t.max()) * \
            (int(u.max()) + 1)
        stats["max_padded_audio_s"] = max(stats["max_padded_audio_s"],
                                          float(batch["padded_audio_s"]))
        stats["max_padded_budget_s"] = max(stats["max_padded_budget_s"],
                                           float(batch["padded_budget_s"]))
        loss_value += float(loss_f) * (n_i / weights_den)
        del loss, loss_f

    # Pre-clip grad norm is the diagnostic SPEC section 4 asks for: clipping
    # hides a blow-up, so the number logged is the one before the clip.
    params = [p for p in model.parameters() if p.grad is not None]
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        params, float(cfg["optim"]["grad_clip"])))
    if not math.isfinite(gnorm):
        abort("non-finite gradient norm", run_dir, update=update,
              grad_norm=gnorm)

    for g in opt.param_groups:
        g["lr"] = lr_at(update, g)
    opt.step()
    opt.zero_grad(set_to_none=True)

    stats["loss"] = loss_value
    stats["grad_norm"] = gnorm
    return stats


def params_finite(model):
    """Cheap fused finiteness check over every parameter tensor.

    ``_foreach_norm`` does the whole 0.6B-parameter sweep in a handful of
    fused kernels; a non-finite element makes its tensor's norm non-finite,
    so one stack+isfinite is enough and it costs well under a millisecond.
    """
    norms = torch._foreach_norm([p.detach() for p in model.parameters()])
    return bool(torch.isfinite(torch.stack(norms)).all())


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path, model, opt, sampler, update, epoch, cfg,
                    consumed=0, last_micro_ids=None, extra=None):
    """Full resume point: weights, optimizer, sampler, RNG, step counter.

    The LR schedulers are functions of `update` (see scoring.lr_at), so the
    step counter IS both schedulers' state - there is no separate scheduler
    object whose state could drift out of sync with the optimizer's. The
    HORIZON those functions decay to is recorded alongside, because SPEC
    section 3 lets Gate 3 fix H once and a resume under a different H would be
    a different LR curve on the same weights; so is `accum_weight_mode`, so a
    mid-lineage change of the objective cannot pass unnoticed.

    `consumed` (microbatches drawn by the TRAINING loop this epoch) and
    `last_micro_ids` are what make the data stream resumable rather than
    merely restartable - see sampler_state_dict().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "update": update,
        "epoch": epoch,
        "consumed_batches": int(consumed),
        "last_micro_ids": list(last_micro_ids or []),
        "horizon_updates": int(SCHED["horizon"]),
        "accum_weight_mode": cfg["optim"].get("accum_weight_mode"),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "sampler": sampler_state_dict(sampler, consumed),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "cfg": cfg,
        "extra": extra or {},
    }, str(path))
    return path


def rotate(ckpt_dir, keep):
    """Keep the newest `keep` rolling checkpoints; protected copies are made
    BEFORE this runs (SPEC section 4), never recovered from it."""
    files = sorted(ckpt_dir.glob("update_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in files[:-keep] if keep > 0 else []:
        old.unlink()


def protect(ckpt_path, protected_dir, tag):
    protected_dir.mkdir(parents=True, exist_ok=True)
    dst = protected_dir / f"{tag}_{ckpt_path.name}"
    if not dst.exists():
        shutil.copy2(ckpt_path, dst)
    return dst


# --------------------------------------------------------------------------
# Self-tests (--selftest). Cheap enough to run before every gate.
# --------------------------------------------------------------------------
def selftest_lr(cfg):
    """Test 2: lr_at() at the registered anchor updates, both groups.

    Only the anchors SPEC section 3 states in words are asserted (the ramp
    endpoints and the horizon endpoint); the cosine interiors are printed so a
    reader can eyeball the curve without a second implementation of it drifting
    into the repo.

    The endpoint anchor is taken at the HORIZON IN FORCE (SPEC section 3's
    H, default 60,000, lowerable once at Gate 3), not at a literal, because the
    defect this guards against is precisely a schedule whose denominator is not
    the run's horizon: with H=60,000 against a hard-coded 40,000 denominator
    every head LR from update 501 is wrong and the last 20k updates sit flat on
    the floor. A denominator check is therefore included explicitly.
    """
    h = int(SCHED["horizon"])
    want = {("head", 0): 3.0e-5, ("head", 500): 3.0e-4, ("head", h): 3.0e-5,
            # council-3 amendment: encoder warms 1e-6 -> 1e-4 over 0-1000, cosine to h
            ("encoder", 0): 1.0e-6, ("encoder", 500): 5.05e-5,
            ("encoder", 1000): 1.0e-4, ("encoder", h): 1.0e-6}
    out, ok = {"horizon_in_force": h,
               "registered_max_horizon": int(SCHED["total_updates"])}, True
    for g in ("encoder", "head"):
        for s in sorted({0, 500, 1000, 2000, h // 2, h}):
            v = lr_at(s, g)
            out[f"{g}@{s}"] = v
            if (g, s) in want and abs(v - want[(g, s)]) > 1e-12:
                ok = False
                out[f"{g}@{s}_expected"] = want[(g, s)]
    # Monotonic sanity: the head must peak at 500 and decay after.
    ok = ok and lr_at(499, "head") < lr_at(500, "head") > lr_at(501, "head")
    # council-3 amendment: no frozen phase. Encoder ramps from update 0, peaks at
    # 1,000, and is on the cosine (descending) immediately after.
    ok = ok and 0.0 < lr_at(0, "encoder") < lr_at(500, "encoder") < lr_at(1000, "encoder") > lr_at(1001, "encoder")
    # The cosine must still be DECAYING just before the horizon and only reach
    # the floor AT it - the signature of a wrong denominator is a leg that
    # flattened out thousands of updates early.
    out["head@h-1"] = lr_at(h - 1, "head")
    out["encoder@h-1"] = lr_at(h - 1, "encoder")
    ok = ok and lr_at(h - 1, "head") > lr_at(h, "head")
    ok = ok and lr_at(h - 1, "encoder") > lr_at(h, "encoder")
    # And the horizon must be an argument, not a constant: the same step under
    # a different H must give a different LR.
    out["head@1000_under_H40000"] = lr_at(1000, "head", horizon=40000)
    ok = ok and (h == 40000 or
                 lr_at(1000, "head", horizon=40000) != lr_at(1000, "head"))
    return ok, out


def selftest_bn(model, bns, specaug, cfg):
    """Test 3: every BatchNorm is still in eval mode after the unfreeze.

    The failure this guards against is subtle: ``model.train()`` at the top
    of each step re-arms every BN, so if the re-freeze is ever dropped the
    run keeps training and only the running stats quietly rot.
    """
    set_phase(model, bns, specaug, frozen=True)
    frozen_ok = all(not b.training for b in bns) and not model.encoder.training
    set_phase(model, bns, specaug, frozen=False)
    after = all(not b.training for b in bns) and all(b.momentum == 0.0
                                                     for b in bns)
    enc_train = model.encoder.training
    grad_on = all(p.requires_grad for p in model.encoder.parameters())
    return (frozen_ok and after and enc_train and grad_on), {
        "n_batchnorm": len(bns), "frozen_phase_bn_eval": frozen_ok,
        "unfrozen_bn_eval_and_momentum0": after,
        "encoder_train_mode_after_unfreeze": enc_train,
        "encoder_requires_grad_after_unfreeze": grad_on,
        "specaug_attached_after_unfreeze": model.spec_augmentation is not None,
    }


def selftest_accum(model, cfg, device, seed=20260904):
    """Test 1: partitioned+weighted accumulation == one unpartitioned batch.

    Run in EVAL mode (no dropout, no SpecAugment) so the only difference
    between the two paths is the partitioning itself. Synthetic audio and
    targets are used deliberately: the identity being checked is algebraic,
    and real audio would only add I/O and make the test slower to run before
    every gate.

    Both weighting modes are measured, and the PASS is gated on the mode that
    actually SHIPS (`optim.accum_weight_mode`), not on whichever one happens to
    satisfy the criterion. Gating only "tokens" while shipping "utterances" -
    which is what this test used to do - meant --selftest could go green while
    the shipped objective was never tested at all.

    This exposes a genuine INTERNAL CONTRADICTION in SPEC section 3, which is
    reported rather than resolved here. Section 3 registers BOTH (a) the weight
    n_i / N_accum with N_accum counted in UTTERANCES and (b) "matches an
    unpartitioned batch loss+grad within 1e-4". Under the registered
    `mean_volume` reduction each microbatch loss is already a per-TOKEN mean
    over that microbatch (sum_j L_j / sum_j U_j), so utterance weighting yields
    an utterance-weighted average of per-microbatch token-means - which depends
    on how the sampler happened to split the group, systematically up-weighting
    short-transcript microbatches. Only TOKEN weighting satisfies (b). The two
    clauses cannot both hold; the owner has to adjudicate which is registered.
    Until then the SPEC-literal weighting is what ships (the more specific
    clause wins) and this test reports the failure instead of hiding it.

    omega is forced to 0 FOR THE DURATION OF THIS TEST ONLY. The registered
    omega=0.1 makes the TDT loss stochastic by construction: the numba kernel
    draws r ~ U(0,1) per call and computes the plain RNN-T loss instead of
    the TDT loss when r < omega (gpu_rnnt.py). Two forward passes over the
    same tensors therefore disagree about 10% of the time, by a factor of
    ~3, which has nothing to do with the partitioning being tested. Note the
    draw is PER FUSED SUB-BATCH, so fused_batch_size also sets the
    granularity of that mixture - a real interaction between two registered
    values, not something this test introduces. omega is restored before
    returning; training keeps the registered 0.1.
    """
    torch.manual_seed(seed)
    b, sr = 4, 16000
    sig = torch.randn(b, 3 * sr, device=device) * 0.05
    sig_len = torch.full((b,), 3 * sr, dtype=torch.long, device=device)
    tok_len = torch.tensor([6, 9, 7, 11], dtype=torch.long, device=device)
    tok = torch.randint(0, 1024, (b, int(tok_len.max())), device=device)
    for i, n in enumerate(tok_len.tolist()):
        tok[i, n:] = 0
    batches = []
    for lo, hi in ((0, 4),):                                   # full batch
        batches.append((lo, hi))

    def one(lo, hi):
        return {"audio": sig[lo:hi], "audio_lens": sig_len[lo:hi],
                "tokens": tok[lo:hi, :int(tok_len[lo:hi].max())],
                "token_lens": tok_len[lo:hi],
                "ids": [f"x{i}" for i in range(lo, hi)],
                "durations": [3.0] * (hi - lo)}

    shipped = cfg["optim"].get("accum_weight_mode", "utterances")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    # See the docstring: omega=0.1 makes the loss a random TDT/RNN-T mixture,
    # which would swamp the ~1e-7 identity this test is checking.
    omega_held = model.loss._loss.omega
    model.loss._loss.omega = 0.0

    # cuDNN refuses to run an LSTM backward on a module in eval mode
    # ("cudnn RNN backward can only be called in training mode"), and the
    # prediction network is an LSTM. Dropping to the non-fused RNN kernel
    # for the duration of the test keeps the model in EVAL mode - which is
    # what makes the two paths comparable at all - instead of switching the
    # decoder to train mode and re-introducing its 0.2 dropout.
    def grads_of(parts, mode):
        model.zero_grad(set_to_none=True)
        den = (sum(int(tok_len[lo:hi].sum()) for lo, hi in parts)
               if mode == "tokens" else sum(hi - lo for lo, hi in parts))
        total = 0.0
        for lo, hi in parts:
            w = ((int(tok_len[lo:hi].sum()) if mode == "tokens" else hi - lo)
                 / den)
            with torch.backends.cudnn.flags(enabled=False):
                loss, _, _ = forward_microbatch(model, one(lo, hi), device)
                lf = loss.float()
                (lf * w).backward()
            total += float(lf) * w
        g = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                       if p.grad is not None])
        return total, g

    full_loss, full_grad = grads_of([(0, 4)], "tokens")
    out = {"full_loss": full_loss,
           "full_grad_l2": float(full_grad.norm())}
    ok = True
    for mode in ("tokens", "utterances"):
        for tag, parts in (("2+2", [(0, 2), (2, 4)]),
                           ("3+1", [(0, 3), (3, 4)])):
            l, g = grads_of(parts, mode)
            dl = abs(l - full_loss)
            dg = float((g - full_grad).abs().max())
            rel = float((g - full_grad).norm() / max(full_grad.norm(), 1e-12))
            out[f"{mode}/{tag}"] = {"loss": l, "abs_loss_delta": dl,
                                    "max_abs_grad_delta": dg,
                                    "rel_grad_l2": rel}
            if mode == shipped:
                # "within 1e-4" read the conservative way: an ABSOLUTE
                # tolerance on the loss and on the largest single gradient
                # element. The relative L2 over 617M parameters is reported
                # too but not gated - it sits near 1e-4 purely because the
                # two paths sum the same numbers in a different order (and
                # cuDNN uses TF32 in the encoder convolutions), which is
                # floating-point noise, not a difference in the objective.
                ok = ok and dl < 1e-4 and dg < 1e-4
    model.zero_grad(set_to_none=True)
    model.loss._loss.omega = omega_held
    out["omega_restored"] = float(model.loss._loss.omega)
    out["shipped_accum_weight_mode"] = shipped
    out["gated_on"] = shipped
    if not ok and shipped != "tokens":
        out["owner_adjudication_required"] = (
            "SPEC section 3 registers utterance weighting AND a 1e-4 match to "
            "an unpartitioned batch; under mean_volume only token weighting "
            "can satisfy the second. Shipped mode is "
            f"{shipped!r}, so the registered unit test cannot pass. Owner must "
            "either register accum_weight_mode=tokens or drop/reword the "
            "unit-test clause. Not resolved in code.")
    return ok, out


def selftest_omega(model, device, trials=40):
    """Measure the omega mixture rate so the step log's outliers are known.

    Not a pass/fail of the trainer - it is a measurement of a registered
    property (SPEC section 1 fixes omega=0.1). The same fixed logits are
    scored `trials` times; the fraction of calls that come back far from the
    modal value is the empirical probability that the kernel sampled the
    plain RNN-T loss instead of TDT. It is logged so that a reader of
    steps.jsonl knows roughly one update in ten carries an RNN-T-flavoured
    loss term and does not go hunting for a bug.
    """
    torch.manual_seed(1)
    B, T, U, V = 2, 40, 8, 1030
    lp = torch.randn(B, T, U + 1, V, device=device)
    il = torch.full((B,), T, dtype=torch.int32, device=device)
    tl = torch.full((B,), U, dtype=torch.int32, device=device)
    tg = torch.randint(0, 1024, (B, U), device=device).int()
    held = model.loss.reduction
    model.loss.reduction = None
    try:
        vals = []
        for _ in range(trials):
            with torch.no_grad():
                vals.append(float(model.loss(log_probs=lp, targets=tg,
                                             input_lengths=il,
                                             target_lengths=tl)[0]))
    finally:
        model.loss.reduction = held
    lo = min(vals)
    n_alt = sum(1 for v in vals if v > lo * 1.5)
    return True, {"trials": trials, "modal_loss": round(lo, 4),
                  "sampled_rnnt_fraction": round(n_alt / trials, 3),
                  "registered_omega": float(model.loss._loss.omega)}


def selftest_resume_parity(cfg, tokenizer, n=8, cut_at=4):
    """Test 5: a resumed run draws the SAME microbatches, in the same order.

    This is the test whose absence let a real defect ship. Claim (7) of the
    protocol - "a resumed run is a continuation" - was refuted empirically:
    smoke12 (updates 1-12 straight through) and smoke12r (resumed at update 6)
    disagreed on EVERY cut id from the resume point on, because the DataLoader
    prefetches ``num_workers * prefetch_factor`` microbatches out of the
    in-process Lhotse sampler, so the sampler's own batch counter ran ~16
    microbatches ahead of the training loop and ``_fast_forward()`` skipped
    them on resume - dropping rows from the epoch, which SPEC sections 0/2
    forbid, and breaking Gate 3's equal-example-exposure premise.

    The check is at the CUT-ID level and deliberately does no model compute:
    identical cut ids in identical order is the property the protocol needs,
    and it is the property that was violated. Losses follow from ids plus the
    restored RNG/optimizer state, which the checkpoint already covers.

    Sequence: draw `n` microbatches straight through, snapshot the sampler at
    `cut_at` exactly as save_checkpoint() would, then rebuild the loader from
    scratch, restore, and require (a) the first replayed microbatch to be the
    last one consumed before the snapshot - the same assertion the trainer
    makes on every real resume - and (b) microbatches cut_at+1..n to be
    identical to the straight-through run's.
    """
    loader, sampler = build_loader(cfg, tokenizer)
    stream = BatchStream(loader, sampler)
    ids, snap, snap_epoch = [], None, 0
    for i in range(n):
        epoch, batch = next(stream)
        ids.append(list(batch["ids"]))
        if i + 1 == cut_at:
            snap = save_sampler_snapshot(sampler, stream.consumed)
            snap_epoch = epoch
    del stream, loader, sampler           # drop the worker pool before re-open

    loader2, sampler2 = build_loader(cfg, tokenizer)
    sampler_load_state_dict(sampler2, snap)
    stream2 = BatchStream(loader2, sampler2, snap_epoch, resumed=True,
                          consumed=max(cut_at - 1, 0))
    _, replay = next(stream2)
    replayed = [list(replay["ids"])]
    for _ in range(n - cut_at):
        _, b = next(stream2)
        replayed.append(list(b["ids"]))
    del stream2, loader2, sampler2

    handshake = replayed[0] == ids[cut_at - 1]
    same = replayed[1:] == ids[cut_at:]
    return (handshake and same), {
        "microbatches_drawn": n, "snapshot_after": cut_at,
        "resume_handshake_ids_match": handshake,
        "post_resume_ids_identical": same,
        "first_divergence": next(
            (k + cut_at for k, (a, b) in enumerate(zip(replayed[1:],
                                                       ids[cut_at:]))
             if a != b), None),
        "straight_through_ids": ids[cut_at:],
        "resumed_ids": replayed[1:],
    }


def run_selftests(args, cfg):
    """--selftest: tests 1-3 and 5. Test 4 (the 12-update smoke) is a real run."""
    global P3LhotseDataset
    device = torch.device("cuda")
    pin_gpu()
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    P3LhotseDataset = _make_dataset_class()      # selftest 5 builds a loader
    results = {}
    ok_lr, det = selftest_lr(cfg)
    results["lr_at"] = {"pass": ok_lr, **det}

    arm, init_sha = assert_registered_init(args.init)
    results["registered_init"] = {"pass": True, "arm": arm,
                                  "sha256": init_sha}
    model = EncDecRNNTBPEModel.restore_from(args.init, map_location="cpu")
    assert_model_contract(model, cfg)
    enforce_decoding(model, cfg)
    model = model.to(device)
    model.joint.set_fuse_loss_wer(True, model.loss, model.wer)
    model.joint.set_fused_batch_size(int(cfg["model"]["fused_batch_size"]))
    specaug, bns = model.spec_augmentation, batchnorm_modules(model)

    ok_bn, det = selftest_bn(model, bns, specaug, cfg)
    results["batchnorm_frozen"] = {"pass": ok_bn, **det}
    ok_acc, det = selftest_accum(model, cfg, device)
    results["accumulation"] = {"pass": ok_acc, **det}
    ok_om, det = selftest_omega(model, device)
    results["omega_sampling"] = {"pass": ok_om, **det}
    ok_rp, det = selftest_resume_parity(cfg, model.tokenizer)
    results["resume_parity"] = {"pass": ok_rp, **det}

    print(json.dumps(results, indent=1, default=float))
    hard_exit(0 if all(v["pass"] for v in results.values()) else 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True,
                    help="path to init_{v2,v3}_reinit.nemo (SPEC section 1)")
    ap.add_argument("--config", default=str(REPO / "configs/ctc.yaml"))
    ap.add_argument("--run-name", default=None,
                    help="run directory name under experiments/; required "
                         "unless --selftest")
    ap.add_argument("--max-updates", type=int, default=None,
                    help="default: sched.total_updates from the config")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="gate stop; exits 0 after this update, keeping state")
    ap.add_argument("--unregistered-sched-override", default=None,
                    help="JSON like '{\"encoder\": {\"lr_peak\": 1e-4}}'. PROBES ONLY: bypasses the "
                         "registered-schedule assertion, applies the values, and stamps the run dir "
                         "UNREGISTERED_PROBE so it can never be mistaken for a registered run.")
    ap.add_argument("--val-at", default="", help="comma-separated update numbers to validate at (Gate-3 anchors) in addition to --val-every")
    ap.add_argument("--val-every", type=int, default=0,
                    help="0 disables periodic validation")
    ap.add_argument("--val-batch-size", type=int, default=32)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--smoke-panel", default=None,
                    help="manifest to validate on instead of the locked dev "
                         "panel (Gate-2 128-row smoke panel)")
    ap.add_argument("--protect", type=int, action="append", default=[],
                    help="update numbers whose checkpoint is copied out of "
                         "rotation; repeatable")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'auto' for the newest in the run")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--hourly-seconds", type=float, default=None,
                    help="override log.hourly_seconds; 0 summarises every "
                         "update (used by the smoke test, never by a real run)")
    ap.add_argument("--selftest", action="store_true",
                    help="run self-tests 1-3 (needs the GPU, < 10 min) and "
                         "exit; the 12-update smoke is test 4 and is a real "
                         "run of this script")
    return ap.parse_args()


def main():
    args = parse_args()
    anonymize_process()
    device = torch.device("cuda")
    gpu_name = pin_gpu()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    assert cfg["gpu_uuid"] == os.environ["CUDA_VISIBLE_DEVICES"], "GPU pin"
    configure_optim(cfg["optim"])
    if args.unregistered_sched_override:
        import json as _json
        _ov=_json.loads(args.unregistered_sched_override)
        for _g,_kv in _ov.items():
            for _k,_v in _kv.items():
                SCHED[_g][_k]=float(_v); cfg['sched'][_g][_k]=float(_v)
        _rd = REPO / 'experiments' / 'p3' / args.run_name; _rd.mkdir(parents=True, exist_ok=True)
        (_rd / 'UNREGISTERED_PROBE').write_text(args.unregistered_sched_override)
    sched = configure_schedules(cfg["sched"])
    if args.selftest:
        os.chdir(REPO)
        sys.exit(run_selftests(args, cfg))
    assert args.run_name, "--run-name is required unless --selftest"
    max_updates = args.max_updates or sched["total_updates"]
    stop_at = args.stop_at or max_updates

    run_dir = REPO / cfg["paths"]["runs_dir"] / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _RUN_DIR[0] = run_dir
    ckpt_dir, protected_dir = run_dir / "ckpt", run_dir / "protected"

    # Manifest paths are repo-relative in the JSON, and Lhotse resolves them
    # against the process cwd - so pin the cwd rather than rewriting 552,932
    # rows.
    os.chdir(REPO)

    torch.manual_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    global P3LhotseDataset
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    P3LhotseDataset = _make_dataset_class()

    # SPEC section 1.3 requires v2 and v3 to be seeded from ONE shared random
    # decoder+joint. Two byte-different init sets existed on this box at once
    # and the early smoke runs used the unregistered copy, so the sha256 is
    # checked against experiments/inits/INITS.json BEFORE any GPU work.
    init_arm, init_sha = assert_registered_init(args.init)
    print(f"init arm={init_arm} sha256={init_sha[:16]} (registered)",
          flush=True)

    model = EncDecRNNTBPEModel.restore_from(args.init, map_location="cpu")
    assert_model_contract(model, cfg)
    enforce_decoding(model, cfg)
    model = model.to(device)
    model.joint.set_fuse_loss_wer(True, model.loss, model.wer)
    model.joint.set_fused_batch_size(int(cfg["model"]["fused_batch_size"]))

    # SpecAugment is detached during the freeze phase and re-attached after,
    # so keep the module (and check its registered widths) up front.
    specaug = model.spec_augmentation
    assert specaug is not None, "init has no SpectrogramAugmentation module"
    sa = cfg["phase"]["specaug"]
    assert specaug.spec_augment.freq_masks == sa["freq_masks"]
    assert specaug.spec_augment.time_masks == sa["time_masks"]
    assert specaug.spec_augment.freq_width == sa["freq_width"]
    assert abs(specaug.spec_augment.time_width - sa["time_width"]) < 1e-9
    bns = batchnorm_modules(model)

    groups = make_param_groups(model)
    opt = torch.optim.AdamW(groups, lr=groups[0]["lr"],
                            betas=tuple(cfg["optim"]["betas"]),
                            eps=float(cfg["optim"]["eps"]))

    loader, sampler = build_loader(cfg, model.tokenizer)

    update, epoch, resumed = 0, 0, False
    consumed, replay_ids = 0, []
    if args.resume:
        path = (max(ckpt_dir.glob("update_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))
                if args.resume == "auto" else Path(args.resume))
        state = torch.load(path, map_location=device, weights_only=False)
        # A resume under a different horizon is a different LR curve on the
        # same weights, and a resume under a different accumulation weighting
        # is a different objective. Both are refused rather than reconciled.
        ck_h = int(state.get("horizon_updates", -1))
        assert ck_h == int(SCHED["horizon"]), (
            f"checkpoint horizon {ck_h} != config horizon "
            f"{SCHED['horizon']}; SPEC section 3 fixes H once at Gate 3")
        ck_mode = state.get("accum_weight_mode")
        assert ck_mode == cfg["optim"].get("accum_weight_mode"), (
            f"checkpoint accum_weight_mode {ck_mode!r} != config "
            f"{cfg['optim'].get('accum_weight_mode')!r}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sampler_load_state_dict(sampler, state["sampler"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        update, epoch, resumed = state["update"], state["epoch"], True
        consumed = int(state.get("consumed_batches", 0))
        replay_ids = list(state.get("last_micro_ids", []))
        print(f"resumed update={update} epoch={epoch} "
              f"consumed_batches={consumed} from {path}", flush=True)

    probe = NvmlProbe(cfg["gpu_uuid"])
    stop_evt = threading.Event()
    side = threading.Thread(target=nvml_sidecar, daemon=True,
                            args=(probe, run_dir / "nvml.csv",
                                  float(cfg["log"]["nvml_hz"]), stop_evt))
    side.start()

    hourly = HourlyLogger(run_dir, float(cfg["log"]["hourly_seconds"])
                          if args.hourly_seconds is None
                          else float(args.hourly_seconds))
    steps_fh = open(run_dir / "steps.jsonl", "a", encoding="utf-8")
    val_path = run_dir / "val.jsonl"
    val_manifest = (args.smoke_panel
                    or str(REPO / cfg["paths"]["dev_manifest"]))

    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_name": args.run_name, "init": args.init,
        "init_sha256": sha256_file(args.init),
        "config": args.config, "config_sha256": sha256_file(args.config),
        "code_sha256": {f: sha256_file(REPO / "tools" / f)
                        for f in ("scoring.py", "train_core.py")},
        "train_manifest": cfg["paths"]["train_manifest"],
        "val_manifest": val_manifest, "gpu": gpu_name,
        "max_updates": max_updates, "stop_at": stop_at,
        "accum_weight_mode": cfg["optim"].get("accum_weight_mode"),
        "init_arm": init_arm,
        "horizon_updates": int(SCHED["horizon"]),
        "total_updates_registered": int(SCHED["total_updates"]),
        "resumed_from_update": update,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }, indent=1), encoding="utf-8")

    # The sampler was rewound to consumed-1, so the first microbatch the
    # stream yields must be the LAST one the checkpointed run consumed. That
    # handshake is what turns "the data stream probably resumed correctly" into
    # a checked property; if it fails the run stops rather than silently
    # training on a different, row-dropping epoch (SPEC sections 0/2/4).
    data_iter = BatchStream(loader, sampler, epoch, resumed,
                            consumed=max(consumed - 1, 0) if resumed else 0)
    if resumed and replay_ids:
        _, first = next(data_iter)
        if list(first["ids"]) != list(replay_ids):
            abort("resume handshake failed: replayed microbatch is not the "
                  "last one consumed before the checkpoint", run_dir,
                  update=update, expected_ids=list(replay_ids),
                  got_ids=list(first["ids"]),
                  consumed_batches=consumed)
        print(f"resume handshake OK on {len(replay_ids)} cut ids", flush=True)
    freeze_n = int(cfg["phase"]["freeze_updates"])
    min_utts = int(cfg["optim"]["accum_min_utterances"])
    abort_gib = float(cfg["limits"]["abort_reserved_gib"])
    envelope_gib = float(cfg["limits"]["envelope_reserved_gib"])
    protect_set = set(args.protect)

    hour_loss, hour_updates, t_hour = [], 0, time.time()
    last_val = {}
    t_run = time.time()
    # `stage` names what the loop was doing when something threw, so the ABORT
    # record says where. SPEC section 4's abort list is meaningless if half the
    # ways a run can die never write an ABORT file (see the outer handler).
    stage = "startup"
    rec = {"lr_encoder": None, "lr_head": None, "peak_reserved_gib": None}
    # Seeded from the resume handshake so the final checkpoint of a run that
    # does zero updates (already at stop_at) still records a valid data
    # position instead of raising on an unbound name.
    last_ids = list(replay_ids)

    def write_hourly(phase, reason):
        """One hourly summary line (SPEC section 4's field set).

        Called on the interval AND once at clean exit. The exit flush is not
        cosmetic: without it a run shorter than `log.hourly_seconds` produces
        no hourly.jsonl at all, so every sub-hour gate run silently lacked the
        file SPEC section 4 requires (observed on the first dry-run leg, 0.019 h
        -> no file). A real 60k run is unaffected either way; the gates are not.
        """
        el = time.time() - t_hour
        uph = hour_updates / max(el / 3600.0, 1e-9)
        return hourly.log(
            update=update, epoch=epoch, phase=phase, reason=reason,
            mean_loss=round(float(np.mean(hour_loss)), 5) if hour_loss else None,
            dev_wer_frozen=last_val.get("wer_frozen"),
            dev_wer_nemo=last_val.get("wer_nemo"),
            lr_encoder=rec["lr_encoder"], lr_head=rec["lr_head"],
            peak_reserved_gib=rec["peak_reserved_gib"],
            updates_per_hour=round(uph, 1),
            eta_hours_to_max=round((max_updates - update) / max(uph, 1e-9), 2),
            elapsed_hours=round((time.time() - t_run) / 3600.0, 3))

    # The WHOLE loop body is inside this try, not just run_update(). An OOM in
    # dev_eval() (greedy decode of the 5,385-row panel at batch 32 - the largest
    # single allocation in the run after the training step), a DataLoader worker
    # crash surfacing as a bare RuntimeError out of next(data_iter), or a
    # failure while writing a checkpoint would otherwise kill the process with a
    # traceback, no ABORT file on disk and an exit code that is not 3 - so a
    # watchdog keyed on either signal would miss them entirely (SPEC section 4).
    try:
        while update < stop_at:
            phase = "frozen" if update < freeze_n else "unfrozen"
            set_phase(model, bns, specaug, update < freeze_n)

            stage = "gather"
            micros, n_utt, epoch = gather_microbatches(
                data_iter, min_utts, cfg, run_dir, update)
            consumed = data_iter.consumed
            last_ids = list(micros[-1]["ids"])

            # Peak stats are per-update: reset before, synchronize before
            # reading, or the number recorded is whatever the allocator held.
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            stage = "update"
            try:
                stats = run_update(model, opt, micros, n_utt, cfg, device,
                                   run_dir, update)
            except torch.cuda.OutOfMemoryError as exc:
                abort("CUDA OOM", run_dir, stage=stage, update=update,
                      detail=str(exc)[:400])
            except RuntimeError as exc:
                if ("illegal memory access" in str(exc).lower()
                        or "CUDA" in str(exc)):
                    abort("CUDA runtime error", run_dir, stage=stage,
                          update=update, detail=str(exc)[:400])
                raise
            torch.cuda.synchronize()
            wall = time.time() - t0

            if not params_finite(model):
                abort("non-finite parameter after step", run_dir, update=update)

            peak_res = torch.cuda.max_memory_reserved() / GIB
            peak_alloc = torch.cuda.max_memory_allocated() / GIB
            if peak_res > abort_gib:
                abort("reserved memory over the hard ceiling", run_dir,
                      update=update, peak_reserved_gib=round(peak_res, 3),
                      ceiling_gib=abort_gib)

            update += 1
            nv = probe.last
            rec = {
                "update": update, "epoch": epoch, "phase": phase,
                "n_micro": len(micros), "n_utt": n_utt,
                "consumed_batches": consumed,
                "lr_encoder": lr_at(update - 1, "encoder"),
                "lr_head": lr_at(update - 1, "head"),
                "horizon_updates": int(SCHED["horizon"]),
                "loss": round(stats["loss"], 6),
                "loss_micro": [round(x, 6) for x in stats["loss_terms"]],
                "grad_norm": round(stats["grad_norm"], 6),
                "skipped_rows": 0,
                "B_max": stats["B_max"], "T_max": stats["T_max"],
                "U_max": stats["U_max"],
                "BxT_max": stats["B_max"] * stats["T_max"],
                "sum_seconds": round(stats["sum_seconds"], 3),
                "max_seconds": round(stats["max_seconds"], 3),
                # SPEC section 2: the envelope record is in PADDED seconds.
                "max_padded_audio_s": round(stats["max_padded_audio_s"], 3),
                "max_padded_budget_s": round(stats["max_padded_budget_s"], 3),
                "lattice_cells": stats["lattice_cells"],
                "lattice_cells_padded": stats["lattice_cells_padded"],
                "alloc_gib": round(torch.cuda.memory_allocated() / GIB, 3),
                "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 3),
                "peak_alloc_gib": round(peak_alloc, 3),
                "peak_reserved_gib": round(peak_res, 3),
                "over_envelope": bool(peak_res > envelope_gib),
                "nvml_used_mib": nv.get("used_mib"),
                "nvml_free_mib": nv.get("free_mib"),
                "wall_s": round(wall, 4),
                "audio_s_per_s": round(stats["sum_seconds"] / max(wall, 1e-9), 2),
                "micro": stats["micro"],
            }
            if update % args.log_every == 0 or peak_res > envelope_gib:
                steps_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                steps_fh.flush()
            hour_loss.append(stats["loss"])
            hour_updates += 1

            # --- validation (both scorers) ---------------------------------
            val_at = {int(x) for x in args.val_at.split(",") if x.strip()}
            do_val = (args.val_every and update % args.val_every == 0) or (update in val_at)
            if (do_val or update in protect_set
                    or update % int(cfg["ckpt"]["every"]) == 0):
                stage = "checkpoint"
                ck = save_checkpoint(ckpt_dir / f"update_{update}.pt", model,
                                     opt, sampler, update, epoch, cfg,
                                     consumed=consumed, last_micro_ids=last_ids,
                                     extra={"loss": stats["loss"]})
                # Protected copies are taken BEFORE rotation, so a Gate or
                # validation candidate can never be deleted by keep-3.
                if update in protect_set:
                    protect(ck, protected_dir, "protect")
                if do_val:
                    protect(ck, protected_dir, "val")
                rotate(ckpt_dir, int(cfg["ckpt"]["keep_last"]))

            if do_val:
                stage = "validate"
                res = dev_eval(model, val_manifest,
                               batch_size=args.val_batch_size,
                               limit=args.val_limit)
                # Release the evaluation's allocator blocks ONCE per validation. Measured on
                # pilot_v2: the 5,385-row transcribe grew reserved memory 16.74 -> 17.56 GiB
                # at update 2218 and it never came back; ten validations could walk the run
                # into the 18.0 GiB abort. This is not the in-loop empty_cache SPEC 4 forbids:
                # training-step fragmentation is still measured between validations.
                torch.cuda.synchronize()
                _res_before = torch.cuda.memory_reserved() / 2**30
                torch.cuda.empty_cache()
                res['reserved_gib_before_release'] = round(_res_before, 3)
                res['reserved_gib_after_release'] = round(torch.cuda.memory_reserved() / 2**30, 3)
                hyps = res.pop("hyps")
                (run_dir / "val_preds").mkdir(exist_ok=True)
                (run_dir / "val_preds" / f"update_{update}.json").write_text(
                    json.dumps(hyps, ensure_ascii=False), encoding="utf-8")
                res.update({"update": update, "manifest": val_manifest,
                            "phase": phase})
                with open(val_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                last_val = res
                print(f"VAL update={update} wer_frozen={res['wer_frozen']} "
                      f"wer_nemo={res['wer_nemo']} cer={res['cer_frozen']}",
                      flush=True)
                # dev_eval put the model back in train mode; the phase flags
                # are re-applied at the top of the next iteration, but BN must
                # not be left armed even for the instant in between.
                set_phase(model, bns, specaug, update < freeze_n)

            # --- hourly summary --------------------------------------------
            if hourly.due():
                stage = "hourly"
                write_hourly(phase, "interval")
                hour_loss, hour_updates, t_hour = [], 0, time.time()

        stage = "final-checkpoint"
        ck = save_checkpoint(ckpt_dir / f"update_{update}.pt", model, opt,
                             sampler, update, epoch, cfg,
                             consumed=data_iter.consumed,
                             last_micro_ids=last_ids,
                             extra={"final": True})
        protect(ck, protected_dir, "final")
        rotate(ckpt_dir, int(cfg["ckpt"]["keep_last"]))
    except torch.cuda.OutOfMemoryError as exc:
        abort("CUDA OOM", run_dir, stage=stage, update=update,
              detail=str(exc)[:400])
    except BaseException as exc:                    # noqa: BLE001 - see above
        import traceback as _tb
        abort(f"unhandled exception ({type(exc).__name__})", run_dir,
              stage=stage, update=update, detail=str(exc)[:400],
              traceback=_tb.format_exc()[-2000:])

    # Flush a final hourly line so a run shorter than the interval still
    # produces the SPEC section 4 summary file.
    write_hourly("frozen" if update < freeze_n else "unfrozen", "exit")
    stop_evt.set()
    steps_fh.close()
    del data_iter, loader                     # drop the worker pool's iterator
    print(f"TRAIN-EXIT clean update={update} "
          f"elapsed_h={round((time.time() - t_run) / 3600.0, 3)}", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    # Last line of defence: anything that escapes main() - an assertion during
    # start-up, a driver error while the model is being moved to the card - is
    # still an abort in SPEC section 4's sense, so it must leave an ABORT file
    # (when a run directory is known) and exit 3 rather than a bare traceback.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _exc:                   # noqa: BLE001
        import traceback as _tb
        abort(f"unhandled exception in main ({type(_exc).__name__})",
              _RUN_DIR[0], detail=str(_exc)[:400],
              traceback=_tb.format_exc()[-2000:])
