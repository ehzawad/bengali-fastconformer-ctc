#!/usr/bin/env python3
"""Select, qualify, export, and mint ``experiments/qualified_selection.json``.

WHY THIS EXISTS
---------------
``final_test.require_qualified_selection`` refuses the sealed FLEURS decode
unless ``experiments/qualified_selection.json`` exists and says the run had
a NORMAL ending whose SELECTED checkpoint PASSED qualification (SPEC §8w.2 item
4). That reader is tested. **Nothing wrote the file.** Found 2026-09-15, about
four hours before the run ends, while walking the end-of-run path step by step
after the exporter (2026-09-13) and the stale baseline constant (2026-09-13)
had already turned out to be missing links in the same chain.

This tool is the single writer of that record. It does, in order:

1. **Normal ending.** The final validation at update H must be in
   ``val.jsonl`` and the final checkpoint must be on disk. A run that logged
   H and died before finalising is not a normal ending (council R, SPEC
   §8w.9), and no record is written.
2. **Selection.** ``common.select_best`` over every eligible validation:
   lowest eligible S, ties within 0.05 pp to the EARLIER checkpoint. Applied
   by the registered function, never by eye.
3. **Qualification (SPEC §8w.2 item 4)** on THAT checkpoint, which need not
   be the literal H checkpoint: S ≤ 22.0296, D ≤ 21.348, every registered
   guard passed, 1e-9 pp representation tolerance.
4. **Export + verify.** The selected ``step_*.pt`` is exported to a ``.nemo``
   with ``export_nemo`` and the archive must reproduce the trainer's
   recorded dev WER exactly. The record names the ``.nemo``, because that is
   what ``final_test`` decodes and what its sha256 check compares against.
5. **Mint the record**, atomically, refusing to overwrite an existing one. A
   second mint is a second selection, which SPEC §8w.2 forbids.

``--dry-run`` performs 2-4 against the run AS IT STANDS (optionally truncated
with ``--as-of``), exports to a scratch path, and writes NOTHING canonical.
That is how the tool was rehearsed at update ~117,000 before the real ending.

Usage::

    bnasr/qualify.py --dry-run --as-of 110000       # rehearse now
    bnasr/qualify.py                                 # the one real mint
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bnasr.scoring import sha256_file  # noqa: E402
from bnasr import common as P4  # noqa: E402

RUN_DIR_DEFAULT = REPO / "experiments/runs/N_O+macro_filtered_H120k"
QUALIFIED_SELECTION = REPO / "experiments/qualified_selection.json"

# SPEC §8w.2 item 4, verbatim thresholds. Asserted so a silent edit is loud.
QUAL_S_PP = 22.0296
QUAL_D_PP = 21.348
TOL_PP = 1e-9
assert QUAL_S_PP == 22.0296 and QUAL_D_PP == 21.348, "SPEC §8w.2 qualification thresholds"


def load_vals(run_dir, as_of=None):
    vals = [json.loads(l) for l in open(run_dir / "val.jsonl", encoding="utf-8") if l.strip()]
    if as_of is not None:
        vals = [v for v in vals if int(v["update"]) <= int(as_of)]
    return vals


def s_of(v):
    e = v.get("S_exact")
    return float(e) if e is not None else float(v["S_pp"])


def normal_ending(run_dir, horizon, vals):
    """(ok, reasons). Final validation AND final checkpoint both present."""
    reasons = []
    if not any(int(v["update"]) == int(horizon) for v in vals):
        reasons.append(f"no validation record at update {horizon}")
    ck = [run_dir / d / f"step_{horizon}.pt" for d in ("milestones", "ckpt", "best", "preserved")]
    if not any(p.exists() for p in ck):
        reasons.append(f"no step_{horizon}.pt under milestones/, ckpt/, best/ or preserved/")
    return (not reasons), reasons


def find_pt(run_dir, update):
    for d in ("best", "preserved", "milestones", "ckpt"):
        p = run_dir / d / f"step_{update}.pt"
        if p.exists():
            return p
    return None


def qualify(v):
    """SPEC §8w.2 item 4 on one validation record -> (ok, checks)."""
    S = s_of(v)
    D = float(v["panels"]["D"]["wer_frozen"]) * 100.0
    guards = bool((v.get("guards") or {}).get("pass"))
    checks = {
        "S_pp": S, "S_le_22.0296": S <= QUAL_S_PP + TOL_PP,
        "D_pp": D, "D_le_21.348": D <= QUAL_D_PP + TOL_PP,
        "all_guards_pass": guards,
        "eligible": P4.is_eligible(v),
    }
    ok = checks["S_le_22.0296"] and checks["D_le_21.348"] and guards and checks["eligible"]
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=str(RUN_DIR_DEFAULT))
    ap.add_argument("--dry-run", action="store_true",
                    help="select/qualify/export to scratch; write no canonical record")
    ap.add_argument("--as-of", type=int, default=None,
                    help="(dry-run only) truncate the validation record at this update")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the exported-archive re-score (dry-run diagnostics only)")
    a = ap.parse_args()

    if a.as_of is not None and not a.dry_run:
        raise SystemExit("ABORT: --as-of is a rehearsal control and requires --dry-run")

    from bnasr.scoring import anonymize_process, pin_gpu
    anonymize_process()
    pin_gpu()

    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")) if (run_dir / "run_meta.json").exists() else {}
    ident = json.loads((run_dir / "IDENTITY.json").read_text(encoding="utf-8"))
    run_uid = (run_dir / "RUN_UID").read_text(encoding="utf-8").strip()
    # The horizon comes from the RUN's own record, cross-checked against the
    # latest checkpoint, never from a module constant: P4.HORIZON is 150,000
    # (the pre-council-N value) and a silent fallback to it would make this
    # tool look for step_150000.pt forever and never see a normal ending.
    horizon = meta.get("horizon")
    if horizon is None:
        raise SystemExit(f"ABORT: {run_dir / 'run_meta.json'} has no 'horizon'; refusing to guess")
    horizon = int(horizon)
    ck_dir = run_dir / "ckpt"
    latest = sorted(ck_dir.glob("step_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
    if latest:
        import torch
        blob = torch.load(str(latest[-1]), map_location="cpu", weights_only=False)
        ck_h = int(blob.get("horizon", -1))
        if ck_h != horizon:
            raise SystemExit(f"ABORT: run_meta horizon {horizon} != checkpoint horizon {ck_h} "
                             f"in {latest[-1].name}; the run identity is inconsistent")

    vals = load_vals(run_dir, a.as_of)
    if not vals:
        raise SystemExit("ABORT: no validations")

    # ---- 1. normal ending ---------------------------------------------------
    ended, why = normal_ending(run_dir, horizon, vals)
    if not ended and not a.dry_run:
        raise SystemExit("ABORT: not a normal ending, no record written: " + "; ".join(why))

    # ---- 2. selection, by the registered function ----------------------------
    hist = [(s_of(v), int(v["update"]), P4.is_eligible(v)) for v in vals]
    sel = P4.select_best(hist)
    if sel is None:
        raise SystemExit("ABORT: no eligible checkpoint; qualification fails by SPEC §8w.2 item 4")
    sel_S, sel_u = sel
    sel_v = next(v for v in vals if int(v["update"]) == sel_u)
    tied = sorted(u for s, u, e in hist if e and s <= min(s for s, _, e2 in hist if e2) + P4.TIE_BAND_PP)

    # ---- 3. qualification ----------------------------------------------------
    qual_ok, checks = qualify(sel_v)

    # ---- 4. export + verify --------------------------------------------------
    pt = find_pt(run_dir, sel_u)
    if pt is None:
        raise SystemExit(f"ABORT: selected step_{sel_u}.pt is not on disk in any checkpoint dir")
    out_dir = run_dir / ("export_dryrun" if a.dry_run else "export")
    nemo = out_dir / f"step_{sel_u}.nemo"
    from bnasr.export_nemo import export, verify
    if nemo.exists() and not a.dry_run:
        # An existing export is fine ONLY if it is byte-bound to this .pt; re-verify below.
        pass
    else:
        export(pt, nemo)
    ver = None
    if not a.no_verify:
        ver = verify(nemo, run_dir, sel_u, device=a.device)
        if not ver["match"]:
            raise SystemExit(f"ABORT: exported archive scores {ver['exported_wer_frozen']} on dev, "
                             f"trainer recorded {ver['recorded_wer_frozen']} at {sel_u}. Not the selected model.")

    rec = {
        "spec": "SPEC §8w.2 item 4 — qualification of the selected checkpoint after a normal ending",
        "tool": "bnasr/qualify.py",
        "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_uid": run_uid,
        "run_dir": str(run_dir),
        "horizon": horizon,
        "ending": "normal" if ended else "INCOMPLETE",
        "ending_reasons": why,
        "dry_run": bool(a.dry_run),
        "as_of": a.as_of,
        "n_validations": len(vals),
        "n_eligible": sum(1 for _, _, e in hist if e),
        "selection_rule": f"lowest eligible S; ties within {P4.TIE_BAND_PP} pp -> earlier",
        "tied_with": tied,
        "selected_update": sel_u,
        "selected_S_pp": sel_S,
        "qualified": bool(qual_ok),
        "qualification": checks,
        "source_pt": str(pt),
        "source_pt_sha256": sha256_file(pt),
        "selected_checkpoint": str(nemo),
        "selected_sha256": sha256_file(nemo),
        "export_verified": ver,
        "identity": {k: ident.get(k) for k in ("run_uid", "init", "init_sha256",
                                                "tokenizer_sha256", "train_manifest_sha256",
                                                "effective_batch")},
    }
    print(json.dumps(rec, indent=1, ensure_ascii=False))

    # ---- 5. mint (never overwrite) -------------------------------------------
    if a.dry_run:
        print(f"\nDRY RUN: nothing canonical written. Selection would be step_{sel_u} "
              f"(S {sel_S:.5f}), qualified={qual_ok}, ending={'normal' if ended else 'INCOMPLETE'}.")
        return
    if not qual_ok:
        raise SystemExit("ABORT: the selected checkpoint did NOT pass qualification. "
                         "Recording a failure is done by the phase report, not by this file; "
                         "no qualified_selection.json is written (SPEC §8w.2 item 4).")
    if QUALIFIED_SELECTION.exists():
        raise SystemExit(f"ABORT: {QUALIFIED_SELECTION} already exists. A second mint is a "
                         f"second selection. Nothing written.")
    # common.atomic_save is torch.save (for checkpoints). The first real mint
    # on 2026-09-15 07:58 UTC used it here and produced a zip/pickle blob that
    # json.loads in require_qualified_selection would have crashed on. This
    # record is JSON: write it atomically ourselves and READ IT BACK through
    # the same parser the consumer uses before claiming success.
    tmp = QUALIFIED_SELECTION.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
        fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, QUALIFIED_SELECTION)
    back = json.loads(QUALIFIED_SELECTION.read_text(encoding="utf-8"))
    for k in ("ending", "qualified", "selected_checkpoint", "selected_sha256"):
        if back.get(k) != rec[k]:
            raise SystemExit(f"ABORT: read-back mismatch on {k}; the record on disk is not what was minted")
    print(f"\nMINTED {QUALIFIED_SELECTION} (JSON, read back OK): step_{sel_u} qualified, "
          f".nemo sha {rec['selected_sha256'][:16]}…")


if __name__ == "__main__":
    main()
