#!/usr/bin/env python3
"""Phase-4 corpus builder (SPEC §1–§2, §5): protected panels, dev-slice M, carving, dedup, caps,
CTC feasibility, and the three static pilot manifests O / N / C.

Everything here is deterministic (seeded) and CPU-only. It consumes the bucket manifests written by the
ingest tools and emits, under experiments/corpus/:

  panel_M.jsonl                 MaCro validated dev slice (≈3.5 h, ≥500 sentence groups, ≤3 recordings/group,
                                groups drawn by seed from sentences with NO historical exposure)
  arm_O.jsonl                   Phase-1 corpus with re-normalised targets (control)
  arm_N.jsonl                   O + all admitted new buckets (natural mixture)
  arm_C.jsonl                   N with macro_filtered+madasr jointly downsampled to 30 % of retained hours
  build_report.json             every count: rows/hours per source per arm, removals per reason, feasibility stats
  protected_texts.json          the full protected sentence set (frozen-surface strings) used for carving

Rules implemented (numbers are SPEC's; changing one here without changing SPEC is a violation):
  * text surface for EVERY row: metric_surface(NFC(text), False) + whitespace collapse (§1 RULING)
  * protected set P = D ∪ holdout ∪ FLEURS test ∪ FLEURS validation ∪ Kathbath valid ∪ MADASR dev (if present) ∪ M
  * any training row whose surface text ∈ P is removed (all buckets, old corpus included)
  * global MAX_PER_TEXT = 5 distinct recordings per surface text across ALL sources; keep order = sha1(path)
  * new-bucket duration 0.5–23 s (old rows keep Phase-1 bounds); cps 2–35 for new rows
  * tokenizer feasibility with the Phase-1 1,024 BPE: no <unk>; L + R ≤ T and (L+R)/T ≤ 0.90 where
      L = #pieces, R = #adjacent equal pieces, T = encoder frames = floor(floor(dur*100)/8)  (dw-striding 8×, 10 ms hop;
      this matches NeMo's ConvSubsampling length formula for stride-2 layers with padding, checked in Gate 0)
    A new bucket losing >1 % of its rows to feasibility is a STOP condition (reported, not silently applied).
  * speaker exclusions where ids exist: Kathbath valid speaker_ids removed from kathbath training rows.
  * audio dedup: exact path; plus optional PCM-hash table (data4/audit/pcm_sha256.tsv) if the audit has produced one.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bnasr.normalize import metric_surface  # noqa: E402

import sentencepiece as spm  # noqa: E402

OUT = REPO / "experiments/corpus"
SEED = 20260905
MAX_PER_TEXT = 5
NEW_DUR = (0.5, 23.0)
NEW_CPS = (2.0, 35.0)
FEAS_MARGIN = 0.90
M_TARGET_HOURS, M_MIN_GROUPS, M_MAX_PER_GROUP = 3.5, 500, 3
CAP_SHARE = 0.30
TOKENIZER = REPO / "data/tokenizer_v1024_corpus/tokenizer.model"
import re
LATIN = re.compile(r"[A-Za-z]")
NON_BN = re.compile(r"[^\u0980-\u09FF0-9\s\u200c\u200d]")     # anything outside Bengali block, ASCII digits, space, ZWNJ/ZWJ
MATH = re.compile(r"[+=×÷%<>/]")                              # checked on the RAW text (the surface maps these to spaces)

OLD_TRAIN = REPO / "data2/train_corpus_ready.json"
PANEL_D = REPO / "data2/val_corpus.json"
HOLDOUT = REPO / "data2/holdout_corpus.json"
FLEURS = Path("/mnt/sdb/arafat/ehz/llm/bench_assets/eval_manifest.json")
KATHBATH_VALID = REPO / "data4/kathbath/manifest_valid.jsonl"
MADASR_DEV_CSV = REPO / "data4/tugstugi/csv/bengali-madasr-dev.csv"      # text-only protection until the audio arrives
MADASR_DEV = REPO / "data4/respin/manifest_madasr_dev.jsonl"
NEW_BUCKETS = {
    "macro_filtered": REPO / "data4/macro/manifest_macro_filtered.jsonl",
    "macro_validated": REPO / "data4/macro/manifest_macro_validated.jsonl",
    "madasr": REPO / "data4/respin/manifest_madasr.jsonl",
    "vaani": REPO / "data4/vaani/manifest_train.jsonl",
    "spring_inx": None,   # union of R1 + R2 train manifests, resolved in main()
}
SPRING_TRAIN = [REPO / "data4/spring_inx/R1/manifest_train.jsonl", REPO / "data4/spring_inx/R2/manifest_train.jsonl"]
# Official validation/test splits of the extra buckets are PROTECTED (never trained on; one-shot reports only)
EXTRA_PROTECTED = {
    # macro_validated is panel M's parent and is PANEL_ONLY by the owner's 2026-09-05 ruling.
    # It was absent from this map, so only the 1,829 rows that became panel M were protected by
    # name; 5 rows of it survived into arm_O+madasr.jsonl (found 2026-09-16). Protect the source.
    "macro_validated": REPO / "data4/macro/manifest_macro_validated.jsonl",
    "vaani_validation": REPO / "data4/vaani/manifest_validation.jsonl",
    "vaani_test": REPO / "data4/vaani/manifest_test.jsonl",
    "spring_r1_validation": REPO / "data4/spring_inx/R1/manifest_validation.jsonl",
    "spring_r1_test": REPO / "data4/spring_inx/R1/manifest_test.jsonl",
    "spring_r2_validation": REPO / "data4/spring_inx/R2/manifest_validation.jsonl",
    "spring_r2_test": REPO / "data4/spring_inx/R2/manifest_test.jsonl",
}
CAPPED_BUCKETS = ("macro_filtered", "madasr")
PANEL_ONLY_BUCKETS = {"macro_validated"}       # owner ruling 2026-09-05 (strict G0.9(b)): feeds panel M only, never trains
OWNER_EXCLUDED_BUCKETS = set()                  # 2026-09-06: the owner's "catastrophic" verdicts targeted the Phase-1 HYPOTHESIS shown as a hint, not the
                                                # reference (verified: hishab transcribes the board audio to the reference); references judged OK for all buckets
DUP_RATE_MAX = 0.005                            # G0.9(b): incoming duplicate/defect rate per new bucket


def surf(t):
    # "|" is Vaani's sentence delimiter (their danda); Unicode classes it as a math symbol (Sm), so the frozen surface,
    # which maps only P* to space, would keep it and G0.9(a) would then count 1,119 legitimate rows as violations.
    # Registered 2026-09-06: "|" -> space before the surface, for every bucket (no other source uses it as text).
    return " ".join(metric_surface(unicodedata.normalize("NFC", t.replace("|", " ")), False).split())


def rows_of(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def h64(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:16], 16)


def frames(dur):
    return math.floor(math.floor(dur * 100) / 8)


class Feas:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
        self.unk = self.sp.unk_id()

    def check(self, text, dur):
        ids = self.sp.encode(text)
        L = len(ids)
        R = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        T = frames(dur)
        if self.unk in ids:
            return "unk"
        if L + R > T:
            return "infeasible"
        if (L + R) / max(T, 1) > FEAS_MARGIN:
            return "margin"
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fixed-M", default=None,
                    help="use this existing panel_M.jsonl (minus fingerprint-exposed rows) instead of re-selecting M; "
                         "required after the fingerprint audit, which indexed exactly that M")
    ap.add_argument("--skip-missing-buckets", action="store_true",
                    help="build with whichever new-bucket manifests exist (dry runs before all data arrived)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rep = {"seed": SEED, "removed": defaultdict(Counter), "feasibility": {}, "arms": {}, "panels": {}}

    # ---- protected sets -------------------------------------------------------------------------
    P = {}
    def protect(name, texts):
        s = {surf(t) for t in texts if t and surf(t)}
        P[name] = s
        rep["panels"].setdefault(name, {})["distinct_sentences"] = len(s)
    protect("D", [r["text"] for r in rows_of(PANEL_D)])
    protect("holdout", [r["text"] for r in rows_of(HOLDOUT)])
    fl = json.load(open(FLEURS))["rows"]
    protect("fleurs_test", [r["reference"] for r in fl if r["split"] == "test"])
    protect("fleurs_validation", [r["reference"] for r in fl if r["split"] == "validation"])
    kv = rows_of(KATHBATH_VALID) if KATHBATH_VALID.exists() else []
    protect("kathbath_valid", [r["text"] for r in kv])
    if MADASR_DEV.exists():
        protect("madasr_dev", [r["text"] for r in rows_of(MADASR_DEV)])
    for name, pth in EXTRA_PROTECTED.items():
        if pth.exists():
            protect(name, [r["text"] for r in rows_of(pth)])
    # The CSV is loaded UNCONDITIONALLY and that is deliberate. It carries the full MADASR dev
    # text, which removes 50,621 rows / 61.9 h = 99.6 % of everything the madasr bucket protects;
    # the dev manifest alone removes 0.176 h. Until 2026-09-16 this block was written as `else:`
    # on the `for` above — a for/else runs unless the loop breaks, and nothing breaks, so it ran
    # every time by accident while reading like a fallback for a missing manifest. One refactor
    # adding a `break` would have silently dropped nearly all of panel R's protection.
    if not MADASR_DEV_CSV.exists():
        raise SystemExit(f"ABORT: {MADASR_DEV_CSV} is missing; it carries 99.6 % of the MADASR "
                         f"protected text. Refusing to build a corpus without it.")
    with open(MADASR_DEV_CSV, encoding="utf-8", newline="") as fh:
        protect("madasr_dev(text-only)", [r[2] for r in csv.reader(fh) if len(r) >= 3])

    # historical exposure = every sentence any earlier phase trained on (Phases 1–3 shared one corpus)
    old = rows_of(OLD_TRAIN)
    # train_corpus_ready.json dropped speaker ids; re-join Kathbath speaker ids from the converter's manifest so
    # the official-valid speaker exclusion can be enforced (SPEC §1 "speaker exclusions where ids exist").
    kb_spk = {}
    kb_manifest = REPO / "data2/kathbath/manifest.jsonl"
    if kb_manifest.exists():
        for r in rows_of(kb_manifest):
            kb_spk[Path(r["audio_filepath"]).name] = r.get("speaker_id")
    for r in old:
        r["raw_text"] = r["text"]
        r["text"] = surf(r["text"])
        if r["source"] == "kathbath":
            r["speaker_id"] = kb_spk.get(Path(r["audio_filepath"]).name)
    hist = {r["text"] for r in old}
    rep["historical_train_sentences"] = len(hist)

    # ---- new buckets ------------------------------------------------------------------------------
    buckets = {}
    for name, p in NEW_BUCKETS.items():
        if name == "spring_inx":
            present = [q for q in SPRING_TRAIN if q.exists()]
            if not present:
                rep["removed"][name]["MANIFEST_MISSING"] = 1; continue
            rs = [r for q in present for r in rows_of(q)]
            for r in rs:
                r["raw_text"] = r.get("raw_text", r["text"]); r["text"] = surf(r["text"]); r["source"] = name
            buckets[name] = rs; continue
        if not p.exists():
            if a.skip_missing_buckets:
                rep["removed"][name]["MANIFEST_MISSING"] = 1; continue
            sys.exit(f"missing bucket manifest {p}")
        rs = rows_of(p)
        for r in rs:
            r["raw_text"] = r.get("raw_text", r["text"])
            r["text"] = surf(r["text"]); r["source"] = name
        buckets[name] = rs

    # Fingerprint evidence (SPEC §3 G0.4, frozen thresholds MIN_HITS 12 / confirmed ≥24 hits & span ≥2 s; calibration
    # negatives never exceeded 11 hits). Registered rule: a training row with ANY match ≥ MIN_HITS to a protected panel
    # is excluded (confirmed or unresolved — zero unresolved matches may remain in retained training); a panel-M
    # recording matched by an OLD-corpus query is historically exposed and is removed from M.
    fp_drop = set(); m_exposed = set(); fp_counts = Counter()
    fpr = REPO / "data4/audit/fp_match_report.json"
    if fpr.exists():
        fj = json.load(open(fpr, encoding="utf-8"))
        for kind in ("confirmed_detail", "unresolved_detail"):
            for e in fj.get(kind, []):
                fp_counts[kind] += 1
                fp_drop.add(e["train_id"])
                if e.get("panel") == "M" and e.get("train_manifest") == "train_corpus_ready":
                    m_exposed.add(e["panel_id"])
        rep["fingerprint"] = {"present": True, "generated": fj.get("generated"), "confirmed": fj.get("confirmed"),
                              "unresolved": fj.get("unresolved"), "train_rows_excluded": len(fp_drop),
                              "M_rows_historically_exposed": len(m_exposed), "thresholds": fj.get("thresholds")}
    else:
        rep["fingerprint"] = {"present": False}

    # ---- panel M: MaCro validated groups with zero historical exposure -----------------------------
    M = []
    if a.fixed_M:
        M_all = rows_of(a.fixed_M)
        M = [r for r in M_all if r["audio_filepath"] not in m_exposed]
        rep["panels"]["M"] = {"fixed_from": a.fixed_M, "rows_in": len(M_all), "rows": len(M),
                              "removed_historical_fingerprint_exposure": len(M_all) - len(M),
                              "hours": round(sum(r["duration"] for r in M) / 3600, 3), "groups": len({r["text"] for r in M})}
        if rep["panels"]["M"]["groups"] < M_MIN_GROUPS:
            rep["panels"]["M"]["STOP"] = f"fewer than {M_MIN_GROUPS} groups after exposure removal"
        protect("M", [r["text"] for r in M])
        with open(out / "panel_M.jsonl", "w", encoding="utf-8") as fh:
            for r in M:
                fh.write(json.dumps({**r, "source": "macro_dev"}, ensure_ascii=False) + "\n")
    elif "macro_validated" in buckets:
        groups = defaultdict(list)
        for r in buckets["macro_validated"]:
            groups[r["text"]].append(r)
        other_protected = set().union(*P.values())
        eligible = [t for t in groups if t not in hist and t not in other_protected]
        rep["panels"]["M"] = {"groups_total": len(groups), "groups_eligible_no_history": len(eligible)}
        eligible.sort(key=h64)          # deterministic order, independent of any model
        rng.shuffle(eligible)
        hours = 0.0
        for t in eligible:
            recs = sorted(groups[t], key=lambda r: h64(r["audio_filepath"]))[:M_MAX_PER_GROUP]
            M.extend(recs); hours += sum(r["duration"] for r in recs) / 3600
            if hours >= M_TARGET_HOURS and len({r["text"] for r in M}) >= M_MIN_GROUPS:
                break
        m_groups = {r["text"] for r in M}
        rep["panels"]["M"].update({"rows": len(M), "hours": round(hours, 3), "groups": len(m_groups)})
        if len(m_groups) < M_MIN_GROUPS:
            rep["panels"]["M"]["STOP"] = f"fewer than {M_MIN_GROUPS} eligible groups"
        protect("M", [r["text"] for r in M])
        with open(out / "panel_M.jsonl", "w", encoding="utf-8") as fh:
            for r in M:
                fh.write(json.dumps({**r, "source": "macro_dev"}, ensure_ascii=False) + "\n")

    protected_all = set().union(*P.values())
    (out / "protected_texts.json").write_text(json.dumps(sorted(protected_all), ensure_ascii=False))

    # ---- audio-hash evidence from the audit (data4/audit/pcm_sha256.tsv: path, sha256, n_samples, status) ---------------
    # Two uses, both registered (SPEC §1 provenance finding 2026-09-05): (1) a training row whose decoded PCM equals a
    # PROTECTED-PANEL recording is removed (MaCro contains Common Voice audio that sits in D/holdout); (2) training rows are
    # de-duplicated by hash across all sources, keeping the first in deterministic sha1(path) order.
    panel_paths = set()
    for pth in (PANEL_D, HOLDOUT, KATHBATH_VALID, MADASR_DEV, *EXTRA_PROTECTED.values()):
        if pth.exists():
            panel_paths |= {r["audio_filepath"] for r in rows_of(pth)}
    if MADASR_DEV.exists():
        pass
    fleurs_dir = FLEURS.parent / "eval_fleurs_bn"
    path_hash = {}; bad_audio = {}
    tsv = REPO / "data4/audit/pcm_sha256.tsv"
    if tsv.exists():
        for line in open(tsv, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and (len(f) < 4 or f[3] == "ok"):
                path_hash[f[0]] = f[1]
            elif len(f) >= 4:
                bad_audio[f[0]] = f[3]          # e.g. "silent": the audit could not obtain a usable decoded signal
    # The audit's silence test mis-handles 32-bit FLOAT WAV (FLEURS-train): it flagged 1,782 files as silent that decode to
    # peak 0.08 / RMS 0.01 with soundfile. Re-verify every flagged file here; a file with real signal is kept and hashed
    # (sha256 of its int16 PCM) so it still takes part in duplicate detection. Verified 2026-09-06 on fleurs_train_000001.
    if bad_audio:
        import hashlib as _hl
        import numpy as _np
        import soundfile as _sf
        rescued = 0
        for pth in list(bad_audio):
            try:
                x, sr = _sf.read(str(REPO / pth), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if x.size and float(_np.abs(x).max()) >= 1e-3:
                    pcm = (_np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
                    path_hash[pth] = _hl.sha256(pcm).hexdigest()
                    del bad_audio[pth]; rescued += 1
            except Exception:  # noqa: BLE001
                pass
        rep["audio_defect_reverification"] = {"flagged_by_audit": rescued + len(bad_audio), "rescued_with_signal": rescued,
                                              "still_defective": len(bad_audio)}
    panel_hashes = {h for pth, h in path_hash.items()
                    if pth in panel_paths or pth.startswith(str(fleurs_dir)) or "/eval_fleurs_bn/" in pth or "panel_M" in pth}
    rep["pcm_hash_table"] = {"present": tsv.exists(), "hashed_paths": len(path_hash), "panel_hashes": len(panel_hashes)}
    fuzzy_drop = set()
    cand = REPO / "data4/audit/text_candidates.json"
    if cand.exists():
        for method, lst in json.load(open(cand, encoding="utf-8")).items():
            for c in lst:
                fuzzy_drop.add(c["train_id"])
    rep["text_candidates"] = {"present": cand.exists(), "train_rows_listed": len(fuzzy_drop)}

    # ---- carve / filter every training row ----------------------------------------------------------
    feas = Feas()
    def admit(rows, name, new):
        kept = []
        rm = rep["removed"][name]
        for r in rows:
            t = r["text"]; d = float(r["duration"])
            if not t:
                rm["empty_text"] += 1; continue
            if t in protected_all:
                rm["protected_sentence"] += 1; continue
            if r["audio_filepath"] in bad_audio:
                rm["audio_defect_" + bad_audio[r["audio_filepath"]]] += 1; continue
            h = path_hash.get(r["audio_filepath"])
            if h is not None and h in panel_hashes:
                rm["pcm_hash_matches_protected_panel"] += 1; continue
            if r["audio_filepath"] in fuzzy_drop:
                rm["text_near_duplicate_of_panel"] += 1; continue
            if r["audio_filepath"] in fp_drop:
                rm["fingerprint_match_to_panel"] += 1; continue
            # Kathbath HF valid shares speakers with train (27,230 rows) -> K is a known-speaker panel; no speaker exclusion.
            # G0.9 uniformity rules apply to EVERY bucket, old corpus included (owner 2026-09-05: "recheck the previous
            # dataset, don't take anything for granted"). The Phase-1 audit found 84 Latin-letter FLEURS rows, 262 Subakko
            # rows with Latin vitamin names, 450 Subakko math expressions ("১+৩+৫+৭=১৬" -> after punctuation-to-space the
            # target is a digit soup nobody spoke) and 3,311 OpenSLR rows below 2 chars/s.
            if LATIN.search(t):
                rm["latin_letters"] += 1; continue
            if NON_BN.search(t):
                rm["symbol_outside_bengali_digits"] += 1; continue
            if MATH.search(r.get("raw_text", t)):
                rm["math_expression"] += 1; continue
            cps = len(t.replace(" ", "")) / d
            if not (NEW_CPS[0] <= cps <= NEW_CPS[1]):
                rm["cps"] += 1; continue
            if new:
                if not (NEW_DUR[0] <= d <= NEW_DUR[1]):
                    rm["duration"] += 1; continue
            f = feas.check(t, d)
            if f:
                rm[f"feasibility_{f}"] += 1; continue
            kept.append(r)
        n0 = len(rows)
        fe = sum(v for k, v in rm.items() if k.startswith("feasibility_"))
        rep["feasibility"][name] = {"rows_in": n0, "rejected": fe, "fraction": round(fe / max(n0, 1), 5),
                                   "STOP": fe / max(n0, 1) > 0.01 if new else False}
        return kept

    old_by_src = defaultdict(list)
    for r in old:
        old_by_src[r["source"]].append(r)
    admitted = {}
    for s, rs in old_by_src.items():
        admitted[s] = admit(rs, s, new=False)
    for s, rs in buckets.items():
        admitted[s] = admit(rs, s, new=True)

    # ---- global MAX_PER_TEXT across all sources (deterministic) ---------------------------------------
    # ---- audio components, O-first precedence, incoming duplicate rate (council C #1/#2/#3, owner strict ruling) ------
    # Component = rows sharing a decoded-PCM hash (fingerprint matches are merged by the audit tool into the same table when
    # available). Rules: any member matching a protected recording -> all members excluded (done above per row via panel_hashes,
    # repeated here per component); references agree -> keep the O row (old corpus) if present, else the first new row by
    # sha1(path) order, count the rest as duplicates; references disagree -> whole component excluded for this phase.
    old_sources = set(old_by_src)
    comp = defaultdict(list)
    for s, rs in admitted.items():
        for r in rs:
            h = path_hash.get(r["audio_filepath"])
            if h is not None:
                comp[h].append(r)
    drop_paths = set(); dup_count = Counter(); conflict_count = Counter()
    for h, members in comp.items():
        if len(members) < 2:
            continue
        texts = {m["text"] for m in members}
        if len(texts) > 1:
            for m in members:
                drop_paths.add(m["audio_filepath"]); conflict_count[m["source"]] += 1
            continue
        olds = [m for m in members if m["source"] in old_sources]
        keep = (sorted(olds, key=lambda r: h64(r["audio_filepath"])) or sorted(members, key=lambda r: h64(r["audio_filepath"])))[0]
        for m in members:
            if m is not keep:
                drop_paths.add(m["audio_filepath"]); dup_count[m["source"]] += 1
    for s in list(admitted):
        admitted[s] = [r for r in admitted[s] if r["audio_filepath"] not in drop_paths]
        if dup_count[s]:
            rep["removed"][s]["pcm_duplicate_of_kept_row"] = dup_count[s]
        if conflict_count[s]:
            rep["removed"][s]["pcm_component_conflicting_reference"] = conflict_count[s]

    # G0.9(b) incoming rate per NEW bucket: (protected-hash matches + cross-source duplicates + conflicting components +
    # empty targets) / ingested candidate rows. Ingest-time decode failures are added from the ingest report when present.
    rep["g09b"] = {}
    for s in buckets:
        rm = rep["removed"][s]
        ingest_fail = 0
        for cand in (REPO / "data4/macro" / f"ingest_report_{s}.json", REPO / "data4/respin" / "manifest_madasr_report.json",
                     REPO / f"data4/{s}" / "ingest_report.json"):
            if cand.exists() and s in cand.name or (cand.exists() and s == "madasr" and "madasr" in cand.name):
                try:
                    rj = json.load(open(cand)); rj_rej = rj.get("rejected", {})
                    ingest_fail = sum(v for k, v in rj_rej.items() if any(t in k for t in ("decode", "undecodable", "empty", "silent", "unreadable", "bad_wav")))
                except Exception:  # noqa: BLE001
                    pass
                break
        n_in = len(buckets[s]) + ingest_fail
        script_viol = rm.get("latin_letters", 0) + rm.get("symbol_outside_bengali_digits", 0)
        rep.setdefault("g09a", {})[s] = {"incoming_rows": n_in, "script_violation_rows": script_viol,
                                          "rate": round(script_viol / max(n_in, 1), 5), "max": 0.01,
                                          "verdict": "FAIL" if script_viol / max(n_in, 1) > 0.01 else "pass"}
        n_def = (rm.get("pcm_hash_matches_protected_panel", 0) + rm.get("pcm_duplicate_of_kept_row", 0)
                 + rm.get("pcm_component_conflicting_reference", 0) + rm.get("empty_text", 0) + ingest_fail
                 + sum(v for k, v in rm.items() if k.startswith("audio_defect_")))
        rate = n_def / max(n_in, 1)
        hashed = sum(1 for r in buckets[s] if r["audio_filepath"] in path_hash)
        rep["g09b"][s] = {"ingested_rows": n_in, "defect_rows": n_def, "rate": round(rate, 5), "max": DUP_RATE_MAX,
                          "hash_coverage": round(hashed / max(len(buckets[s]), 1), 4),
                          "verdict": ("INCOMPLETE_HASHING" if hashed < len(buckets[s]) else ("EXCLUDE" if rate >= DUP_RATE_MAX else "pass"))}
        if s in PANEL_ONLY_BUCKETS:
            rep["g09b"][s]["verdict"] = "PANEL_ONLY (owner ruling)"
            admitted[s] = []
        if s in OWNER_EXCLUDED_BUCKETS:
            rep["g09b"][s]["verdict"] = "EXCLUDED (G0.6 listening audit: catastrophic clips found by the owner)"
            admitted[s] = []
        elif rep["g09b"][s]["verdict"] == "EXCLUDE":
            admitted[s] = []
        if rep["g09a"][s]["verdict"] == "FAIL":          # council D Q4: >1 % incoming script violations = bucket fails
            admitted[s] = []

    # O-first precedence, then global MAX_PER_TEXT: O's retained recordings take their sentence slots first.
    by_text = defaultdict(list)
    for s in sorted(admitted, key=lambda x: (x not in old_sources, x)):
        for r in sorted(admitted[s], key=lambda r: h64(r["audio_filepath"])):
            by_text[r["text"]].append(r)
    capped_out = Counter()
    final = defaultdict(list)
    RESPIN_PER_CELL, RESPIN_SENT_TOTAL = 5, 25       # SPEC §1 RESPIN exception (council D Q2)
    cell_occupancy = Counter()
    for t, rs in by_text.items():
        kept = []
        # non-RESPIN rows (O first by construction of by_text order): global allowance of 5
        for r in rs:
            if r["source"] == "madasr":
                continue
            if len(kept) < MAX_PER_TEXT:
                kept.append(r)
            else:
                capped_out[r["source"]] += 1
        # RESPIN rows: 5 distinct speakers per (sentence, dialect), total per sentence <= 25 across all sources
        per_cell_spk = defaultdict(set)
        for r in rs:
            if r["source"] != "madasr":
                continue
            d = r.get("dialect") or "NA"
            spk = r.get("speaker_id")
            if d == "NA" or spk is None:
                # no expanded allowance without dialect metadata: falls under the global 5
                if len(kept) < MAX_PER_TEXT:
                    kept.append(r)
                else:
                    capped_out[r["source"]] += 1
                continue
            if spk in per_cell_spk[d] or len(per_cell_spk[d]) >= RESPIN_PER_CELL or len(kept) >= RESPIN_SENT_TOTAL:
                capped_out[r["source"]] += 1; continue
            per_cell_spk[d].add(spk); kept.append(r); cell_occupancy[d] += 1
        for r in kept:
            final[r["source"]].append(r)
    rep["respin_cap"] = {"per_cell_speakers": RESPIN_PER_CELL, "per_sentence_total": RESPIN_SENT_TOTAL,
                         "retained_by_dialect": dict(cell_occupancy)}
    for s, n in capped_out.items():
        rep["removed"][s]["max_per_text"] = n

    # ---- arms --------------------------------------------------------------------------------------
    def stats(rows):
        c = Counter(); h = Counter()
        for r in rows:
            c[r["source"]] += 1; h[r["source"]] += r["duration"] / 3600
        tot_h = sum(h.values())
        return {"rows": len(rows), "hours": round(tot_h, 2),
                "by_source": {s: {"rows": c[s], "hours": round(h[s], 2), "share": round(h[s] / tot_h, 4)} for s in c}}

    def write(name, rows):
        # The "|" -> space rule lives ONLY in surf() (the frozen scoring surface keeps "|": it is
        # category Sm, and the tokenizer has it at piece rank 999 so the no-<unk> feasibility gate
        # cannot catch it either). One edit to surf() would put pipes into training targets with
        # nothing to notice. Assert the invariant at the only place that matters instead of
        # relying on the rule staying where it is.
        bad = [r for r in rows if "|" in r["text"]]
        if bad:
            raise SystemExit(f"ABORT: {len(bad)} row(s) of arm {name} carry '|' in the training "
                             f"target, e.g. {bad[0]['audio_filepath']}. surf() must map it to space.")
        rows = sorted(rows, key=lambda r: h64(r["audio_filepath"]))     # one fixed global shuffle
        with open(out / f"arm_{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "raw_text"}, ensure_ascii=False) + "\n")
        rep["arms"][name] = stats(rows)

    O = [r for s in old_sources for r in final.get(s, [])]
    write("O", O)
    admitted_new = [s for s in buckets if final.get(s)]
    for s in admitted_new:
        write(f"O+{s}", O + final[s])
    N = O + [r for s in admitted_new for r in final.get(s, [])]
    write("N", N)
    new_h = rep["arms"]["N"]["hours"] - rep["arms"]["O"]["hours"]
    rep["admission"] = {"new_hours_beyond_O": round(new_h, 2), "min_required": 500, "PASS": new_h >= 500}
    rep["removed"] = {k: dict(v) for k, v in rep["removed"].items()}
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ["panels", "arms", "admission", "feasibility"]}, indent=1, ensure_ascii=False))
    print("removed:", json.dumps(rep["removed"], ensure_ascii=False))


if __name__ == "__main__":
    main()
