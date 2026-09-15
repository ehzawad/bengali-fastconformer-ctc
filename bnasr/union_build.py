#!/usr/bin/env python3
"""Council G's ONE union arm U = O ∪ (every bucket that passed its complete single-arm rule).

SPEC §8m, "Execution order adopted": *finish MADASR and Vaani → build exactly ONE union
U = O ∪ (all passing buckets) under the frozen dedup/precedence/cap rules → reuse MaCro's 8k state if U equals it,
else one 8k run of U → admission vs O_ref plus S(U) ≤ S(best passing single) + 0.20 → resume to the unchanged 20k
gate → on failure TERMINATE with no FLEURS decode. No subset search, no second union, no mixture rescue.*

Why this tool does not re-run the corpus builder
------------------------------------------------
"Under the frozen rules" is a byte-level claim, and the cheapest way to make a byte-level claim is to derive the
union FROM the frozen artifacts instead of re-deriving it beside them.  `bnasr/build_corpus.py` (frozen,
sha256 `507f8abb…`, verified below against `experiments/corpus/FREEZE.sha256`) computes ONE global retention
`final[source]` — protected-sentence exclusion, PCM-hash dedup with O-first precedence, G0.9 script/cps/duration
filters, tokenizer feasibility, global MAX_PER_TEXT = 5 with the RESPIN per-(sentence, dialect) exception — and
then writes its arms as source-partition unions of that one partition:

    arm_O            = write(rows of the old-corpus sources)
    arm_O+<bucket>   = write(O + final[bucket])
    arm_N            = write(O + final[macro_filtered] + final[madasr] + final[vaani])

and `write()` sorts by `h64(audio_filepath)` — one fixed global key that does not depend on which sources are
present.  A subsequence of a sorted sequence is still sorted by the same key, so **filtering `arm_N.jsonl` to a
source allowlist reproduces the corresponding frozen arm manifest byte for byte**.  This tool asserts exactly that
on all four frozen unions (O, O+macro_filtered, O+madasr, O+vaani, and arm_N itself) in the same streaming pass
that builds U, and refuses to write anything if any of them differs.  That is a stronger guarantee than re-running
the builder and comparing: re-running would prove the builder is deterministic, whereas this proves that the row
set this tool emits is drawn from the very bytes that were frozen and trained on.

The one place a sub-union could legitimately differ from a from-scratch build, and why it does not here
------------------------------------------------------------------------------------------------------
Dropping a bucket can only change another bucket's retention where the caps make buckets COMPETE.  In the frozen
builder the competition order is O first, then new sources alphabetically (macro_filtered, madasr, vaani), and:

  * the global MAX_PER_TEXT = 5 pass skips `madasr` entirely, so removing `vaani` (which is considered last) frees
    no slot for any earlier source, and removing `madasr` frees nothing in that pass at all;
  * the only cross-bucket coupling left is the RESPIN exception's per-sentence ceiling
    `RESPIN_SENT_TOTAL = 25`, which counts the rows already kept from ALL sources before admitting a madasr row.

So the sub-union equals a from-scratch build iff that 25-row ceiling never bound.  This tool MEASURES it rather
than assuming it: it counts retained recordings per frozen-surface sentence in `arm_N` and refuses unless the
maximum is below RESPIN_SENT_TOTAL (measured 2026-09-07: max 15, zero sentences at ≥25, and all 135 sentences above
the global cap of 5 carry madasr rows — i.e. the RESPIN exception is the only thing that ever exceeded the cap).

What it refuses to do
---------------------
  * refuses if any frozen artifact's sha256 disagrees with FREEZE.sha256;
  * refuses if the union equals an ALREADY-TESTED arm — a single passing bucket means U == arm_O+<bucket>, whose
    8,000-update checkpoint already exists, and §8m says to reuse that state rather than spend a second 8k run;
  * refuses if nothing passed (U == O; SPEC §5: "If nothing passes → STOP this recipe");
  * in --auto mode, refuses while any predeclared arm is still undecided, because building a union from a partial
    pass list is exactly the subset search §8m forbids.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.scoring import sha256_file  # noqa: E402
# Reuse, never duplicate: h64 is the frozen global shuffle key, MAX_PER_TEXT the frozen cap.  Importing them binds
# this tool to the same module FREEZE.sha256 pins (verified in verify_freeze() before anything is written).
from bnasr.build_corpus import MAX_PER_TEXT, h64  # noqa: E402

CORPUS = REPO / "experiments/corpus"
FREEZE = CORPUS / "FREEZE.sha256"
RUNS = REPO / "experiments/runs"
OUT_DEFAULT = REPO / "experiments/union"

# ---------------------------------------------------------------------------------------------------------------
# Registered constants, each asserted against the SPEC section that registers it.
# ---------------------------------------------------------------------------------------------------------------
BUCKETS = ("macro_filtered", "madasr", "vaani")     # SPEC §5 table: the single-bucket arms actually run
                                                    # (spring_inx NOT RUN, failed G0.9(a); cv17 DEFERRED by §8m)
M_ARMS = 4                                          # SPEC §8m: m stays 4 (cv17's slot counts even though deferred)
assert len(BUCKETS) == 3 and M_ARMS == 4, "SPEC §8m: three runnable arms, multiplicity m=4"

MIN_NEW_HOURS = 500.0                               # SPEC §5 / G0.8: ">=500 unique new hours after all exclusions"
S_MARGIN_VS_CONTROL_PP = 0.30                       # SPEC §5: N_final must reach S(O) - 0.30
S_MARGIN_VS_BEST_SINGLE_PP = 0.20                   # SPEC §5 / §8k Q3: "S(N_final) <= S(best single arm) + 0.2 pp"
S_CONTROL_PP = 21.073                               # SPEC §8j: O_ref (runs/O_A5000) at 8,000 updates
assert abs((S_CONTROL_PP - S_MARGIN_VS_CONTROL_PP) - 20.773) < 1e-9, "SPEC §8k Q3: O_ref threshold is S <= 20.773"

RESPIN_PER_CELL, RESPIN_SENT_TOTAL = 5, 25          # SPEC §1 RESPIN exception (council D Q2) — see cap audit below
assert MAX_PER_TEXT == 5, "SPEC §1: global MAX_PER_TEXT = 5 distinct recordings per surface text"
assert (RESPIN_PER_CELL, RESPIN_SENT_TOTAL) == (5, 25)

# The cap constants above are *copies* of literals that live inside build_corpus.main() and so cannot be
# imported.  Pin them to the frozen source text instead: if that line ever changes, this tool stops rather than
# auditing against a ceiling the builder no longer uses.
_BUILDER_SRC = (REPO / "bnasr/build_corpus.py").read_text(encoding="utf-8")
assert f"RESPIN_PER_CELL, RESPIN_SENT_TOTAL = {RESPIN_PER_CELL}, {RESPIN_SENT_TOTAL}" in _BUILDER_SRC, \
    "build_corpus.py no longer declares the RESPIN caps this tool audits against"
assert 'rows = sorted(rows, key=lambda r: h64(r["audio_filepath"]))' in _BUILDER_SRC, \
    "build_corpus.write() no longer sorts by h64(audio_filepath); the filter-of-arm_N derivation is invalid"


def rel(p):
    """Repo-relative path when possible, absolute otherwise — reports must stay readable when --out is elsewhere
    (a scratch directory during a dry run), and Path.relative_to() raises rather than degrading."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


ARM_FILE = {"O": "arm_O.jsonl", **{b: f"arm_O+{b}.jsonl" for b in BUCKETS}, "N": "arm_N.jsonl"}
RUN_DIR = {b: RUNS / f"pilot_O+{b}" for b in BUCKETS}
ADMISSION_JSON = {b: REPO / f"experiments/admission_{b}_m4.json" for b in BUCKETS}


# ---------------------------------------------------------------------------------------------------------------
def verify_freeze():
    """Every artifact this tool reads or reasons about must still hash to its FREEZE.sha256 entry.

    Entries in FREEZE.sha256 are relative to two different roots (the corpus dir for arm_*/panel_M/build_report,
    the repo root for the code and panel manifests), which is why each line is resolved against both.
    Files listed in FREEZE but changed since the freeze are reported as DRIFTED rather than silently ignored;
    the tool only *refuses* when the drift touches something it depends on.
    """
    need = {ARM_FILE[k] for k in ARM_FILE} | {"bnasr/build_corpus.py"}
    ok, drift, checked = [], [], 0
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        name = name.strip()
        p = CORPUS / name if (CORPUS / name).exists() else REPO / name
        if not p.exists():
            drift.append({"file": name, "status": "MISSING"}); continue
        got = sha256_file(p)
        checked += 1
        (ok if got == want else drift).append({"file": name, "status": "ok" if got == want else "DRIFTED",
                                               "expected": want, "actual": got})
    bad = [d for d in drift if d["file"] in need]
    if bad:
        raise SystemExit("REFUSING: frozen artifacts this tool depends on have changed:\n  "
                         + "\n  ".join(f"{d['file']}: {d['status']}" for d in bad))
    return {"entries_checked": checked, "verified": [d["file"] for d in ok],
            "drifted_outside_dependency_set": [d for d in drift]}


def scan(allow, out_path, nvme=False):
    """ONE streaming pass over the frozen arm_N: emit the union, prove the derivation, audit the caps, count hours.

    A single pass is deliberate — arm_N is 449 MiB and two training runs are live on this box, so the tool reads it
    exactly once and derives everything from that pass:
      * the union's bytes (written to `out_path`), its sha256, rows and per-source hours;
      * the sha256 of EVERY frozen source-allowlist filter (O, O+macro_filtered, O+madasr, O+vaani, N), compared
        against FREEZE.sha256 — the byte-identity proof;
      * per-source rows/hours for the whole of arm_N, from which every single arm's inventory and "unique new
        hours beyond O" follow by summation (no extra pass, and no reliance on the ingest reports);
      * the per-sentence retention histogram behind the RESPIN cap audit.
    Returns (union_stats, proof, cap_audit, per_source_rows, per_source_hours).
    """
    src_file = CORPUS / ("arm_N.nvme.jsonl" if nvme else "arm_N.jsonl")
    old = {json.loads(l)["source"] for l in open(CORPUS / "arm_O.jsonl", encoding="utf-8")}
    frozen_sets = {"O": old, **{b: old | {b} for b in BUCKETS}, "N": old | set(BUCKETS)}
    h = {k: hashlib.sha256() for k in frozen_sets}
    n = Counter()
    uh = hashlib.sha256()
    rows = Counter(); hours = defaultdict(float)
    all_rows = Counter(); all_hours = defaultdict(float)
    per_text = Counter(); text_has_madasr = set()
    fh_out = open(out_path, "wb") if out_path else None
    try:
        with open(src_file, "rb") as fh:
            for raw in fh:
                r = json.loads(raw)
                s = r["source"]
                all_rows[s] += 1; all_hours[s] += float(r["duration"]) / 3600.0
                for k, allowed in frozen_sets.items():
                    if s in allowed:
                        h[k].update(raw); n[k] += 1
                if s in allow:
                    uh.update(raw); rows[s] += 1; hours[s] += float(r["duration"]) / 3600.0
                    if fh_out:
                        fh_out.write(raw)
                t = h64(r["text"])
                per_text[t] += 1
                if s == "madasr":
                    text_has_madasr.add(t)
    finally:
        if fh_out:
            fh_out.close()

    suffix = ".nvme.jsonl" if nvme else ".jsonl"
    want = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v, nm = line.split(None, 1)
            want[nm.strip()] = v
    proof = {"source_manifest": rel(src_file), "checks": {}}
    for k, allowed in frozen_sets.items():
        nm = ARM_FILE[k].replace(".jsonl", suffix)
        got = h[k].hexdigest()
        proof["checks"][nm] = {"rows": n[k], "sha256_of_filter": got, "sha256_in_FREEZE": want.get(nm),
                               "byte_identical": got == want.get(nm)}
    proof["all_byte_identical"] = all(c["byte_identical"] for c in proof["checks"].values())
    if not proof["all_byte_identical"]:
        raise SystemExit("REFUSING: filtering arm_N by source allowlist did NOT reproduce the frozen arm "
                         "manifests byte-for-byte; the union derivation is not the frozen build.\n"
                         + json.dumps(proof, indent=1))

    mx = max(per_text.values()) if per_text else 0
    ge = [t for t, v in per_text.items() if v >= RESPIN_SENT_TOTAL]
    over_cap = [t for t, v in per_text.items() if v > MAX_PER_TEXT]
    cap = {"distinct_frozen_surface_sentences": len(per_text),
           "max_retained_recordings_per_sentence": mx,
           "respin_per_sentence_ceiling": RESPIN_SENT_TOTAL,
           "sentences_at_or_above_ceiling": len(ge),
           "sentences_above_global_MAX_PER_TEXT": len(over_cap),
           "of_those_carrying_a_madasr_row": sum(1 for t in over_cap if t in text_has_madasr),
           "SUB_UNION_EQUALS_FROM_SCRATCH_BUILD": bool(len(ge) == 0)}
    if not cap["SUB_UNION_EQUALS_FROM_SCRATCH_BUILD"]:
        raise SystemExit("REFUSING: the RESPIN per-sentence ceiling of "
                         f"{RESPIN_SENT_TOTAL} bound in the frozen build ({len(ge)} sentences), so restricting it to a "
                         "sub-union is NOT what a from-scratch build of that sub-union would produce; the union must "
                         "be rebuilt with build_corpus.py over the passing buckets only.")
    if cap["sentences_above_global_MAX_PER_TEXT"] != cap["of_those_carrying_a_madasr_row"]:
        raise SystemExit("REFUSING: a sentence exceeds MAX_PER_TEXT without a RESPIN (madasr) row — the frozen cap "
                         "logic is not what this tool audits against")

    tot_h = sum(hours.values())
    stats = {"rows": sum(rows.values()), "hours": round(tot_h, 3),
             "by_source": {s: {"rows": rows[s], "hours": round(hours[s], 3),
                               "share": round(hours[s] / max(tot_h, 1e-9), 4)} for s in sorted(rows)},
             "sha256": uh.hexdigest()}
    return stats, proof, cap, all_rows, all_hours


def subset_stats(sources, all_rows, all_hours):
    """rows/hours for an arbitrary source allowlist, summed from the single pass (SPEC §5 hours are corpus hours)."""
    h = sum(all_hours[s] for s in sources)
    return {"rows": sum(all_rows[s] for s in sources), "hours": round(h, 3),
            "by_source": {s: {"rows": all_rows[s], "hours": round(all_hours[s], 3),
                              "share": round(all_hours[s] / max(h, 1e-9), 4)} for s in sorted(sources)}}


def s_at_8k(run_dir, step=8000):
    """S for an arm's 8,000-update checkpoint, from the trainer's own val.jsonl; None while it is still training."""
    p = Path(run_dir) / "val.jsonl"
    if not p.exists():
        return None
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if int(r.get("update", -1)) == step and r.get("S_pp") is not None:
            return float(r["S_pp"])
    return None


def admission_verdict(bucket):
    """(ADMITTED|FAILED|PENDING, evidence) for one bucket, read only from a recorded admission JSON at m=4."""
    p = ADMISSION_JSON[bucket]
    if not p.exists():
        return "PENDING", {"reason": f"no admission record at {rel(p)}",
                           "arm_finished_8k": s_at_8k(RUN_DIR[bucket]) is not None}
    j = json.load(open(p, encoding="utf-8"))
    if int(j.get("m", 0)) != M_ARMS:
        return "PENDING", {"reason": f"admission record was computed at m={j.get('m')}, not the registered m={M_ARMS}"}
    return ("ADMITTED" if j.get("ADMITTED") else "FAILED"), {"record": rel(p),
                                                             "delta_S_pp": j.get("delta_S_pp"),
                                                             "simultaneous_upper_pp": j.get("simultaneous_upper_pp"),
                                                             "S_arm_pp": j.get("S_arm_pp")}


def threshold_table(new_hours):
    """SPEC §5 / §8k Q3, for EVERY possible passing set: the binding admission threshold for the union arm, and
    which single arm could serve as the registered fallback.

    The union's point threshold is the TIGHTER (smaller) of
        S(O_ref) - 0.30                 the ordinary bucket rule against the control  -> 21.073 - 0.30 = 20.773
        S(best passing single) + 0.20   SPEC §5 "adding buckets must not hurt"
    where "best passing single" is the LOWEST-S arm in the passing set.

    Two arms are still training, so their S is unknown — but the table is NOT therefore empty, because admission
    itself bounds them: an arm can only BE in a passing set if it already cleared S <= S(O_ref) - 0.30 = 20.773.
    That necessary condition gives a decidable upper bound on the binding threshold for every set:
        every set containing macro_filtered  ->  binding <= 20.531 + 0.20 = 20.731 (MaCro is best unless another
                                                 passing arm scores lower, which only tightens it further), and the
                                                 best-single rule ALWAYS binds, never the O_ref rule;
        every other non-empty set            ->  binding <= 20.773.
    The FALLBACK is fully determined today and does not depend on any pending S at all: §8k Q3 restricts it to a
    passing single "whose own frozen corpus has >=500 unique new hours", and of the three frozen inventories only
    O+macro_filtered clears that floor (726.5 h vs madasr 149.0 h and vaani 117.4 h).  So a passing set without
    macro_filtered has NO registered fallback: if its union failed, SPEC §5 is "STOP unless the owner amends the
    minimum (council C #9)".
    """
    S = {b: s_at_8k(RUN_DIR[b]) for b in BUCKETS}
    thr_ctl = round(S_CONTROL_PP - S_MARGIN_VS_CONTROL_PP, 3)
    out = []
    for mask in range(1, 1 << len(BUCKETS)):
        subset = [b for i, b in enumerate(BUCKETS) if mask >> i & 1]
        known = {b: S[b] for b in subset if S[b] is not None}
        pending = [b for b in subset if S[b] is None]
        best = min(known, key=known.get) if known else None
        thr_best = round(known[best] + S_MARGIN_VS_BEST_SINGLE_PP, 3) if best else None
        exact = (min(thr_ctl, thr_best) if (thr_best is not None and not pending) else None)
        # Upper bound: a pending arm that ends up in this set must already satisfy S <= thr_ctl, so the lowest S in
        # the set is at most min(known S, thr_ctl) and the best-single threshold is at most that + 0.20.
        s_best_ub = min(list(known.values()) + ([thr_ctl] if pending else []))
        bound = round(min(thr_ctl, s_best_ub + S_MARGIN_VS_BEST_SINGLE_PP), 3)
        elig = [b for b in subset if new_hours[b] >= MIN_NEW_HOURS]
        known_elig = {b: S[b] for b in elig if S[b] is not None}
        fallback = (elig[0] if len(elig) == 1 else
                    (min(known_elig, key=known_elig.get) if (elig and not [b for b in elig if S[b] is None]) else None))
        out.append({
            "passing_set": subset,
            "union_equals": ("arm_O+" + subset[0] if len(subset) == 1 else
                             ("arm_N (all three)" if len(subset) == 3 else "new union U")),
            "already_trained_at_8k": len(subset) == 1,
            "S_of_passing_singles_pp": {b: S[b] for b in subset},
            "pending_arms": pending,
            "threshold_vs_O_ref_pp": thr_ctl,
            "threshold_vs_best_single_pp": thr_best,
            "BINDING_THRESHOLD_pp": exact,
            "BINDING_THRESHOLD_UPPER_BOUND_pp": bound,
            "binding_rule": ("S(best passing single) + 0.20" if bound < thr_ctl else
                             ("S(O_ref) - 0.30" if exact is not None else "undetermined until the pending arms finish")),
            "best_passing_single": best if not pending else f"{best} unless a pending arm scores lower" if best else None,
            "fallback_candidates_meeting_G0.8": elig,
            "registered_fallback_if_union_fails": fallback,
            "fallback_note": (None if fallback else
                              "no passing single in this set clears the 500 unique-new-hour floor (SPEC §5 G0.8 / "
                              "§8k Q3); if the union failed, SPEC §5 is STOP unless the owner amends the minimum"),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passing", default=None,
                    help="comma-separated buckets that passed their complete single-arm rule (e.g. macro_filtered,madasr)")
    ap.add_argument("--auto", action="store_true",
                    help="read the passing set from experiments/admission_<bucket>_m4.json; refuses while any arm is undecided")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--name", default="U", help="arm name; the manifest is arm_<name>.jsonl")
    ap.add_argument("--nvme", action="store_true",
                    help="also derive arm_<name>.nvme.jsonl from the frozen arm_N.nvme.jsonl (no audio is copied: "
                         "the union is a subset of arm_N, whose cache entries already exist)")
    ap.add_argument("--report-only", action="store_true", help="write the report/threshold table, never a manifest")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"spec": "experiments/SPEC.md §5, §8k Q3, §8m", "m": M_ARMS,
           "freeze": verify_freeze()}
    # S(O_ref) is a registered NUMBER (SPEC §8j) but it is also a MEASUREMENT sitting in the control's own val.jsonl.
    # Read it back and require the two to agree: if the control were ever re-run or its record edited, every
    # threshold below would silently move, and a stale literal would be the last thing anyone thought to check.
    s_ctl = s_at_8k(RUNS / "O_A5000")
    if s_ctl is None or abs(round(s_ctl, 3) - S_CONTROL_PP) > 1e-9:
        raise SystemExit(f"REFUSING: SPEC §8j registers S(O_ref) = {S_CONTROL_PP}; runs/O_A5000/val.jsonl says {s_ctl}")
    rep["S_control_pp"] = S_CONTROL_PP
    rep["S_control_source"] = "experiments/runs/O_A5000/val.jsonl update=8000 (SPEC §8j)"

    verdicts = {b: admission_verdict(b) for b in BUCKETS}
    rep["arm_verdicts"] = {b: {"verdict": v, **e} for b, (v, e) in verdicts.items()}
    # A refusal is decided from the pass list alone, BEFORE the scan, so nothing is written on a refused run — but
    # the scan still runs, because the frozen inventory and the threshold table are read-only facts the owner needs
    # while the remaining arms are still training.
    refusal, rc = None, 0
    if a.auto:
        undecided = [b for b, (v, _) in verdicts.items() if v == "PENDING"]
        passing = [b for b, (v, _) in verdicts.items() if v == "ADMITTED"]
        if undecided:
            refusal, rc = (f"--auto: {undecided} have no m={M_ARMS} admission record yet. SPEC §8m allows exactly ONE "
                           "union built after MADASR and Vaani finish; building it from a partial pass list would be "
                           "the subset search §8m forbids."), 3
    elif a.passing is not None:
        passing = [b.strip() for b in a.passing.split(",") if b.strip()]
        bad = [b for b in passing if b not in BUCKETS]
        if bad:
            raise SystemExit(f"unknown bucket(s) {bad}; SPEC §5 registers exactly {list(BUCKETS)}")
        passing = [b for b in BUCKETS if b in passing]          # canonical order
    else:
        raise SystemExit("give --passing or --auto")
    rep["passing_buckets"] = passing
    if refusal is None and not passing:
        refusal, rc = ("nothing passed: U would equal arm_O, which is the control. SPEC §5: "
                       "\"If nothing passes -> STOP this recipe; O is not promoted; FLEURS stays sealed.\""), 2
    elif refusal is None and len(passing) == 1:
        refusal, rc = f"__SINGLE__{passing[0]}", 2

    # ---- ONE pass: build the union, prove the derivation, audit the caps, inventory the hours ------------------
    old_sources = {json.loads(l)["source"] for l in open(CORPUS / "arm_O.jsonl", encoding="utf-8")}
    allow = old_sources | set(passing)
    target = None if (a.report_only or refusal) else out / f"arm_{a.name}.jsonl"
    u, proof, cap, all_rows, all_hours = scan(allow, target)
    rep["cap_audit"] = cap
    rep["derivation_proof"] = proof
    o_full = subset_stats(old_sources, all_rows, all_hours)
    rep["arm_O"] = {**o_full, "sha256": proof["checks"]["arm_O.jsonl"]["sha256_of_filter"]}
    # A bucket's rows are disjoint from O's (different `source` values), so its unique new hours ARE its own hours;
    # asserted rather than assumed, since the whole G0.8 floor rests on it.
    new_hours = {b: round(all_hours[b], 3) for b in BUCKETS}
    for b in BUCKETS:
        d = subset_stats(old_sources | {b}, all_rows, all_hours)["hours"] - o_full["hours"]
        assert abs(d - new_hours[b]) < 1e-6, f"{b}: new-hours identity broken ({d} vs {new_hours[b]})"
    rep["single_arm_inventory"] = {
        b: {**subset_stats(old_sources | {b}, all_rows, all_hours), "unique_new_hours": new_hours[b],
            "min_required_G0.8": MIN_NEW_HOURS, "meets_500h_fallback_floor": new_hours[b] >= MIN_NEW_HOURS,
            "sha256": proof["checks"][f"arm_O+{b}.jsonl"]["sha256_of_filter"]} for b in BUCKETS}
    rep["threshold_table_all_possible_passing_sets"] = threshold_table(new_hours)

    # ---- the refusals ------------------------------------------------------------------------------------------
    if refusal and refusal.startswith("__SINGLE__"):
        b = refusal[len("__SINGLE__"):]
        refusal = (f"U == O + {b} == the already-tested single arm arm_O+{b}.jsonl "
                   f"(sha256 {rep['single_arm_inventory'][b]['sha256']}, byte-identical to the frozen manifest). "
                   f"SPEC §8m: \"reuse MaCro's 8k state if U equals it\" -- do NOT spend a second 8,000-update run.")
        rep["REUSE_CHECKPOINT"] = rel(RUN_DIR[b] / "ckpt" / "step_8000.pt")
        rep["REUSE_RUN_DIR"] = rel(RUN_DIR[b])
        rep["REUSE_MANIFEST"] = rel(CORPUS / ARM_FILE[b])
    if refusal:
        rep["REFUSED"] = refusal
        rep["union_that_was_NOT_built"] = ({k: v for k, v in u.items() if k != "by_source"} if passing else None)
        (out / f"report_{a.name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        print("\nREFUSED: " + refusal, file=sys.stderr)
        if "REUSE_CHECKPOINT" in rep:
            print("Reuse instead: " + rep["REUSE_CHECKPOINT"], file=sys.stderr)
        return rc

    # ---- the union itself ----------------------------------------------------------------------------------------
    u["unique_new_hours_beyond_O"] = round(u["hours"] - o_full["hours"], 3)
    u["new_hours_min_required"] = MIN_NEW_HOURS
    u["meets_500h_floor"] = u["unique_new_hours_beyond_O"] >= MIN_NEW_HOURS
    if target:
        u["manifest"] = rel(target)
        u["sha256_of_file"] = sha256_file(target)
        assert u["sha256_of_file"] == u["sha256"], "streamed digest != file digest"
    equals = [k for k, c in _frozen_equivalence(u["sha256"]).items() if c]
    u["equals_frozen_manifest"] = equals or None
    # Belt and braces for the "never re-run an arm we already trained" rule (SPEC §8m): the single-bucket case is
    # caught above from the pass list, but the authority is the DIGEST, so re-check it against every frozen arm
    # manifest that already owns an 8,000-update checkpoint.
    trained = {ARM_FILE[b]: RUN_DIR[b] for b in BUCKETS if (RUN_DIR[b] / "ckpt" / "step_8000.pt").exists()}
    trained[ARM_FILE["O"]] = RUNS / "O_A5000"
    clash = [nm for nm in (equals or []) if nm in trained]
    if clash:
        if target and target.exists():
            target.unlink()
        rep["REFUSED"] = (f"U is byte-identical to {clash[0]}, which already has an 8,000-update checkpoint; "
                          "SPEC §8m says to reuse that state, not to spend a second 8k run.")
        rep["REUSE_RUN_DIR"] = rel(trained[clash[0]])
        rep["REUSE_CHECKPOINT"] = rel(trained[clash[0]] / "ckpt" / "step_8000.pt")
        (out / f"report_{a.name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        print("\nREFUSED: " + rep["REFUSED"] + "\nReuse instead: " + rep["REUSE_CHECKPOINT"], file=sys.stderr)
        return 2
    if a.nvme and target:
        un, _, _, _, _ = scan(allow, out / f"arm_{a.name}.nvme.jsonl", nvme=True)
        u["nvme_manifest"] = rel(out / f"arm_{a.name}.nvme.jsonl")
        u["nvme_rows"] = un["rows"]
        assert un["rows"] == u["rows"], "nvme manifest row count differs from the union"
    rep["arm_U"] = u
    (out / f"report_{a.name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


def _frozen_equivalence(digest):
    want = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v, nm = line.split(None, 1)
            want[nm.strip()] = v
    return {nm: (v == digest) for nm, v in want.items() if nm.startswith("arm_")}


if __name__ == "__main__":
    sys.exit(main())
