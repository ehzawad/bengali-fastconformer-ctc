#!/usr/bin/env python3
"""Export a this recipe training checkpoint (``.pt``) to a NeMo archive (``.nemo``).

WHY THIS EXISTS
---------------
The trainer validates the LIVE model object every 2,000 updates
(``train.validate`` -> ``common.ctc_panel_eval``), so nothing in the
training loop ever needs to restore a checkpoint from disk. Every *evaluation*
tool does: ``panel_eval.load_ctc_model`` calls ``ASRModel.restore_from``,
which reads a ``.nemo`` archive, and ``final_test`` goes through that same
loader. The run writes ``step_*.pt`` files holding model + optimizer + sampler +
RNG state, which ``restore_from`` cannot read ("tarfile.ReadError: not a gzip
file").

So the path from "the run finished" to "decode the sealed set" had a missing
link, and nothing in ``bnasr/`` or ``experiments/`` closed it. Found
2026-09-13 while rehearsing the final test at update 88,000, i.e. about 37 hours
before it would have blocked the headline decode.

WHAT IT GUARANTEES
------------------
The exported archive must be the SAME MODEL the trainer was scoring, or the
sealed number does not correspond to the development numbers that selected the
checkpoint. Three things enforce that:

1. **The architecture is rebuilt by the trainer's own constructor.**
   ``train.load_init_N`` is called with the config STORED IN THE CHECKPOINT,
   not a config file that may have drifted since. That reproduces the encoder,
   the 1,025-way head over the registered tokenizer, and the registered
   SpecAugment, with all of its assertions intact.
2. **The weights are loaded completely.** ``load_state_dict`` is called
   non-strict and the missing/unexpected lists are then checked explicitly, so
   the failure names the offending tensors instead of raising a wall of text.
   Either list being non-empty is a hard abort: a partially initialised model
   would score as a different system.
3. **The export is verified by re-scoring, not by inspection.** ``--verify``
   decodes the locked dev panel through the exported archive and compares the
   frozen-surface WER against the value the trainer recorded for that update.
   They must agree to 1e-6. A faithful export cannot move the number; an
   unfaithful one almost certainly does.

Usage::

    bnasr/export_nemo.py --ckpt .../best/step_88000.pt \\
        --out .../export/step_88000.nemo --verify

``CUDA_VISIBLE_DEVICES`` must name a registered card (``scoring.pin_gpu``);
the rebuild and save run on CPU, so only ``--verify`` touches the GPU.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import sha256_file  # noqa: E402
from bnasr import common as P4  # noqa: E402


def export(ckpt_path, out_path, verify_sha=True):
    """-> (model, meta). Rebuild the registered architecture, load, save."""
    import torch
    from bnasr.train import load_init_N

    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for k in ("model", "cfg", "update", "init"):
        if k not in blob:
            raise SystemExit(f"ABORT: {ckpt_path} has no '{k}'; this is not a "
                             f"this recipe training checkpoint")
    if str(blob["init"]) != "N":
        raise SystemExit(f"ABORT: checkpoint init is {blob['init']!r}; this "
                         f"exporter rebuilds init N only")

    # The config that TRAINED these weights, not whatever is on disk today.
    cfg = blob["cfg"]
    model, init_sha = load_init_N(cfg, device="cpu", verify_sha=verify_sha)

    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"ABORT: state dict does not match the rebuilt architecture.\n"
            f"  missing ({len(missing)}): {list(missing)[:6]}\n"
            f"  unexpected ({len(unexpected)}): {list(unexpected)[:6]}\n"
            f"A partially loaded model would score as a different system.")

    model.eval()
    model.save_to(str(out_path))

    meta = {
        "source_ckpt": str(ckpt_path),
        "source_ckpt_sha256": sha256_file(ckpt_path),
        "update": int(blob["update"]),
        "epoch": int(blob.get("epoch", -1)),
        "init": str(blob["init"]),
        "init_N_archive_sha256": init_sha,
        "exported_nemo": str(out_path),
        "exported_nemo_sha256": sha256_file(out_path),
        "state_dict_tensors": len(blob["model"]),
        "strict_load": True,
    }
    return model, meta


def recorded_dev_wer(run_dir, update):
    """The frozen-surface panel-D WER the TRAINER recorded at this update."""
    val = Path(run_dir) / "val.jsonl"
    if not val.exists():
        return None
    for line in open(val, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["update"]) == int(update):
            return float(r["panels"]["D"]["wer_frozen"])
    return None


def verify(out_path, run_dir, update, device="cuda", batch_size=32, limit=None):
    """Re-score the locked dev panel through the EXPORTED archive.

    This is the whole point of the tool: an export that is not the model the
    trainer scored is worse than no export, because every downstream number
    would be attributed to a checkpoint that did not produce it.
    """
    from bnasr.panel_eval import load_ctc_model

    want = recorded_dev_wer(run_dir, update)
    model = load_ctc_model(str(out_path), device=device)
    panel = REPO / "data2/val_corpus.json"
    got = P4.ctc_panel_eval(model, str(panel), batch_size=batch_size,
                            limit=limit)
    wer = float(got["wer_frozen"])
    ok = want is not None and abs(wer - want) < 1e-6
    return {"recorded_wer_frozen": want, "exported_wer_frozen": wer,
            "match": ok, "n": got.get("n"), "limit": limit}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a this recipe step_*.pt")
    ap.add_argument("--out", required=True, help="destination .nemo")
    ap.add_argument("--verify", action="store_true",
                    help="re-score the dev panel and require the recorded WER")
    ap.add_argument("--verify-limit", type=int, default=None,
                    help="score only the first N dev rows (a partial check "
                         "CANNOT match the recorded full-panel WER; use only "
                         "as a smoke test, never as the verification)")
    ap.add_argument("--run-dir", default=str(
        REPO / "experiments/runs/N_O+macro_filtered_H120k"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-verify-init-sha", action="store_true",
                    help="skip the NVIDIA archive hash check (diagnostics only)")
    a = ap.parse_args()

    from bnasr.scoring import anonymize_process, pin_gpu
    anonymize_process()
    pin_gpu()

    model, meta = export(a.ckpt, a.out, verify_sha=not a.no_verify_init_sha)
    print(json.dumps(meta, indent=1))

    if a.verify:
        del model
        import torch
        torch.cuda.empty_cache()
        v = verify(a.out, a.run_dir, meta["update"], device=a.device,
                   batch_size=a.batch_size, limit=a.verify_limit)
        print(json.dumps(v, indent=1))
        if a.verify_limit is not None:
            print("PARTIAL smoke check only; not a verification.")
        elif not v["match"]:
            raise SystemExit(
                f"ABORT: the exported archive scores {v['exported_wer_frozen']:.6f} "
                f"on the dev panel but the trainer recorded "
                f"{v['recorded_wer_frozen']} at update {meta['update']}. The "
                f"export is not the model that was selected. Do not use it.")
        else:
            print(f"VERIFIED: exported archive reproduces the recorded dev WER "
                  f"{v['exported_wer_frozen']:.6f} at update {meta['update']}.")


if __name__ == "__main__":
    main()
