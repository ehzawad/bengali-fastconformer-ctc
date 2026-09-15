#!/usr/bin/env python3
"""Progress table for the owner: CTC loss and WER at every validation, 2,000 upward.

Owner, 2026-09-09: "in every hour and every 2000 validation update you will give me
wer ctc loss in a table showing from 2000 to upto so I can really see the progress."

Loss is the mean training CTC loss over the 500 updates ENDING at each validation
point - the loss the model had while it was being evaluated, not a single noisy step.
WER is frozen-surface panel D (5,385 clips), the headline number.

Owner, 2026-09-09: "I don't need what we got in phase one every fucking time.
Okay, phase one is gone." No the baseline column is printed. The forward-looking number
is `margin`: max(S - 22.0296, D_frozen - 21.348), the distance still to cover
before this run's registered qualification is met. Negative means met.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr import common as P  # noqa: E402

WINDOW = 500


def _S(v):
    """The authoritative S for this validation.

    `S_pp` is rounded to 4 dp; `S_exact` is not. The two differ by up to
    3.4e-4 pp on this run, and every comparison here - the 0.05 pp tie band,
    the qualification margin - is a decision, so it reads the unrounded value
    and falls back only for the early records written before S_exact existed.
    """
    e = v.get("S_exact")
    return float(e) if e is not None else float(v["S_pp"])


def build(run_dir):
    run_dir = Path(run_dir)
    vals = [json.loads(l) for l in open(run_dir / "val.jsonl") if l.strip()]
    want = {v["update"] for v in vals}

    # one streaming pass: collect losses in the window before each validation update
    acc = {u: [] for u in want}
    tail_losses = []
    last = None
    with open(run_dir / "steps.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            u, loss = r["update"], r.get("loss")
            last = r
            if loss is None:
                continue
            tail_losses.append(loss)
            if len(tail_losses) > 300:
                tail_losses.pop(0)
            for w in want:
                if w - WINDOW < u <= w:
                    acc[w].append(loss)

    rows = []
    for v in vals:
        u = v["update"]
        D = v["panels"]["D"]
        ls = acc.get(u) or []
        rows.append({
            "update": u,
            "loss": (round(sum(ls) / len(ls), 4) if ls else None),
            "wer": round(D["wer_frozen"] * 100, 3),
            "wer_internal": round(D["wer_nemo"] * 100, 3),
            "S": _S(v),
            "eligible": P.is_eligible(v),
            "conv": round(D["conversational"]["wer_frozen"] * 100, 2),
            "empties": D["empties"],
            "margin": round(max(_S(v) - 22.0296, D["wer_frozen"] * 100 - 21.348), 3),
        })
    tail = None
    if tail_losses:
        tail = sum(tail_losses[-300:]) / len(tail_losses[-300:])
    # SPEC §4 selection, applied by the registered selector rather than by eye.
    # A plain min() over the eligible set returns the numerically lowest S and
    # would have named step_52000 on 2026-09-11, when it leads step_48000 by
    # 0.0004 pp - inside the 0.05 pp tie band, which awards the tie to the
    # EARLIER checkpoint. The rule belongs in the tool, not in the reporter.
    best = P.select_best([(_S(v), v["update"], P.is_eligible(v)) for v in vals])
    return rows, last, tail, best


def render(rows, last=None, tail_mean=None, best=None):
    out = ["| update | CTC loss | WER % | S | conv % | empties | margin to qualify |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    prev = None
    for r in rows:
        delta = "" if prev is None else f" ({r['wer'] - prev:+.3f})"
        prev = r["wer"]
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "-"
        mark = "" if r["eligible"] else " \u2020"
        out.append(f"| {r['update']:,}{mark} | {loss} | **{r['wer']:.3f}**{delta} | {r['S']:.4f} | "
                   f"{r['conv']:.2f} | {r['empties']} | {r['margin']:+.3f} |")
    if any(not r["eligible"] for r in rows):
        out.append("\n\u2020 not eligible for selection (a registered guard failed)")
    if last:
        shown = f"{tail_mean:.4f}" if tail_mean is not None else f"{last['loss']:.4f}"
        out.append(f"\nnow: update {last['update']:,}, loss {shown} (mean of last 300), "
                   f"lr {last['lr']:.3e}, epoch {last['epoch']}")
    if best is not None:
        bs, bu = best
        tied = [r["update"] for r in rows
                if r["eligible"] and r["S"] <= min(x["S"] for x in rows if x["eligible"]) + P.TIE_BAND_PP]
        note = (f" (tied within {P.TIE_BAND_PP} pp with {', '.join(f'{t:,}' for t in tied if t != bu)}"
                f"; SPEC \u00a74 awards a tie to the earlier checkpoint)") if len(tied) > 1 else ""
        out.append(f"selection: step_{bu:,} at S {bs:.5f}{note}")
    else:
        out.append("selection: none - no checkpoint has passed every guard yet")
    return "\n".join(out)


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        REPO / "experiments/runs/N_O+macro_filtered_H120k"
    rows, last, tail, best = build(rd)
    print(render(rows, last, tail, best))
