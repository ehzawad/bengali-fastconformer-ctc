#!/usr/bin/env python
"""Fixtures for the Phase-4 main-run gate tests (workstream 2).

Everything here builds INPUTS. Not one registered rule is re-implemented: the
validation records handed to the trainer are assembled with the real
``p4_common.selection_score`` and the real ``p4_common.guards`` evaluated
against the real ``experiments/p4/baseline_p1.json``, so a test that says "S is
19.40 and the guards pass" is saying it about the same two functions the
trainer calls at 20,000 updates. A fixture that computed S itself would pass
while the trainer failed.

The panel numbers default to Phase 1's own measurements (baseline_p1.json), so
an "unchanged" checkpoint scores S(P1) exactly and every delta a test asks for
is a delta the test states explicitly.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.p4 import p4_common as P4          # noqa: E402
from tools.p4.p4_common import guards, selection_score   # noqa: E402

BASE = P4.load_baseline()
assert BASE is not None, "experiments/p4/baseline_p1.json is required"
BASE_D = BASE["D"]

# Phase-1's own panel WERs, i.e. the point at which every delta below is zero.
P1_PANELS = {k: BASE[k]["wer_frozen"] for k in ("D", "M", "K", "R")}
P1_PER_SOURCE = dict(BASE_D["per_source"])
P1_CONV = BASE_D["conversational"]["wer_frozen"]


def make_val(update, D=None, M=None, K=None, R=None, per_source=None,
             conversational=None, empty_rate=0.0, source_delta_pp=None):
    """One validation record shaped exactly like ``p4_train.validate``'s.

    Fractions in, percentage points out - the same convention as the trainer:
    panel WERs are fractions (0.2035), S is percentage points (21.03).

    `source_delta_pp` shifts EVERY D source by that many pp off Phase 1, which
    is how the +1.0 pp source guard is exercised without hand-writing five
    numbers; `per_source` overrides it for one-source tests.
    """
    wers = {"D": P1_PANELS["D"] if D is None else D,
            "M": P1_PANELS["M"] if M is None else M,
            "K": P1_PANELS["K"] if K is None else K,
            "R": P1_PANELS["R"] if R is None else R}
    if per_source is None:
        shift = 0.0 if source_delta_pp is None else source_delta_pp / 100.0
        per_source = {k: round(v + shift, 6) for k, v in P1_PER_SOURCE.items()}
    conv = P1_CONV if conversational is None else conversational
    d = {"n": BASE_D["n"], "wer_frozen": wers["D"], "cer_frozen": 0.06856,
         "wer_nemo": wers["D"], "empty_rate": empty_rate,
         "empties": int(round(empty_rate * BASE_D["n"])),
         "per_source": per_source,
         "conversational": {"n_ref_words": BASE_D["conversational"]["n_ref_words"],
                            "wer_frozen": conv}}
    panels = {"D": d}
    for name in ("M", "K", "R"):
        panels[name] = {"n": BASE[name]["n"], "wer_frozen": wers[name],
                        "cer_frozen": 0.05, "wer_nemo": wers[name],
                        "empty_rate": 0.0, "empties": 0}
    s, meta = selection_score({k: v["wer_frozen"] for k, v in panels.items()},
                              madasr_admitted=True)
    return {"update": update, "panels": panels, "S_pp": s, "S_meta": meta,
            "guards": guards(d, BASE_D), "val_limit": None,
            "wall_utc": "2026-09-07T00:00:00"}


def val_with_S(update, target_S_pp, **kw):
    """A validation whose registered S is `target_S_pp`, moved on panel M only.

    S = 0.60 D + 0.20 M + 0.10 R + 0.10 K, so holding D/K/R at Phase 1 and
    solving for M puts the composite exactly where a test wants it WITHOUT
    disturbing panel D - which matters, because D has its own gate threshold
    and its own guards and a test of the S threshold must not move them.
    """
    ref = make_val(update, **kw)
    w = P4.S_WEIGHTS_WITH_MADASR
    others = sum(w[k] * ref["panels"][k]["wer_frozen"] * 100.0
                 for k in ("D", "K", "R"))
    m = (target_S_pp - others) / w["M"] / 100.0
    out = make_val(update, M=m, **kw)
    assert abs(out["S_pp"] - target_S_pp) < 5e-4, (out["S_pp"], target_S_pp)
    return out


def run_dir(tmp):
    """An empty run directory with the layout retain() expects."""
    d = Path(tmp)
    for sub in ("ckpt", "best", "milestones"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def write_ckpt(d, update, size=64):
    """A stand-in for a 1.8 GB checkpoint: retain() only ever links, unlinks
    and stats these files, so their CONTENT is irrelevant and their identity
    (inode, link count) is the whole point."""
    p = Path(d) / "ckpt" / f"step_{update}.pt"
    p.write_bytes(bytes(update % 251 for _ in range(size)))
    return p


def cfg():
    """The real configs/p4_ctc.yaml, through the real configure() guard."""
    import yaml
    return P4.configure(yaml.safe_load(
        open(REPO / "configs/p4_ctc.yaml", encoding="utf-8")))


def oref_shaped_baseline(path):
    """A baseline file with O_ref's numbers (SPEC §8j) and no init-A provenance.

    This is the mistake council F Q2 legislates against: O_ref is the
    ADMISSION comparator, 0.043 pp from P1 on S, and using it for the
    continuation guards would loosen every one of them by an arm's worth of
    training noise without looking wrong anywhere.
    """
    b = json.loads(json.dumps(BASE))
    b["S_pp"] = 21.073                    # SPEC §8j O_ref
    b["D"]["wer_frozen"] = 0.2032
    b["D"]["conversational"]["wer_frozen"] = 0.2147
    b.pop("init_sha256", None)
    Path(path).write_text(json.dumps(b), encoding="utf-8")
    return path
