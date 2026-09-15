#!/usr/bin/env python
"""this recipe leakage audit: SPEC 1 dedup steps 2-4, SPEC 3 gates G0.3 and G0.4.

WHY THIS FILE EXISTS
--------------------
this recipe adds ~1.3 M new rows from four repackaged corpora to a 553 k-row
the baseline corpus, and then judges the result on six protected panels.  Council B
risk 6 ("path deduplication misses the same recording under different names",
75 % probability, "0.2-3 pp apparent improvement that may be leakage") and
council A's Q5 pass rule are the reason this is a gate and not a spot check:
re-encoding, trimming, renaming and resampling all defeat file equality, and a
single MaCro clip that is also a locked-dev clip turns the selection score into
a memorisation score.

The audit is deliberately THREE independent surfaces, because each one fails
differently:

  (a) ``pcm-hash``    - decoded-PCM SHA-256.  Catches byte-identical audio
                        across containers/filenames.  Zero false positives by
                        construction; blind to any re-encoding.
  (b) ``text``        - frozen surface + audit keys + indexed fuzzy retrieval.
                        Catches shared PROMPTS (a sentence group that spans the
                        train/panel boundary), which is what SPEC 1 step 4
                        carves on.  Blind to a different sentence read by the
                        same speaker.
  (c) ``fingerprint`` - spectral-peak landmark hashing.  Catches the same
                        RECORDING after mp3 re-encoding, gain, trimming,
                        resampling and +-5 % speed.  Calibrated BEFORE it is
                        ever pointed at a protected panel (SPEC 3 G0.4), so the
                        thresholds cannot be tuned on test matches.

  (d) ``report``      - evaluates the SPEC pass rule over (a)+(b)+(c).

REGISTERED vs UNREGISTERED NUMBERS
----------------------------------
Every constant below that comes from SPEC.md is asserted against SPEC's value
with the section cited, exactly as bnasr/scoring.py and bnasr/common.py
do.  SPEC registers the fuzzy threshold (0.90), the minimum string length (20),
the calibration protocol (200 clips x 6 transforms, >=99 % recovery, 10,000
unrelated + 1,000 same-prompt pairs, 0 false confirmations), the >=2 s matching
span and the pass rule.  SPEC does NOT register the internals of the
fingerprint (mel bins, hop, fan-out, hit threshold): those are MEASURED by
``fingerprint calibrate`` and frozen into data4/audit/fp_thresholds.json, which
is the artefact the pass rule is then evaluated against.  They are marked
"NOT IN SPEC" wherever they appear.

WHAT IS NOT RE-IMPLEMENTED HERE
-------------------------------
  * the target/scoring surface is ``bnasr.normalize.metric_surface(NFC(t),
    False)`` reached through ``bnasr.common.train_surface`` - SPEC 1
    RULING.  The audit compares train against panels on THE SAME surface the
    trainer will use, otherwise the carve in SPEC 1 step 4 removes rows the
    trainer would not have matched.
  * ``bnasr.scoring.sha256_file`` / ``anonymize_process`` / ``load_manifest``.

FLEURS
------
SPEC 2 seals FLEURS test for a single decode in SPEC 6.  This tool never runs a
model.  Hashing and fingerprinting its audio and comparing its references are
explicitly the "fixed audit returning only ids/counts" of G0.4, so FLEURS rows
are matched but their TEXT is never written to any output file - only ids and
counts.  ``--fleurs-text-ok`` does not exist on purpose.
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# The shared-box rule is "at most 12 worker processes", and a worker that
# quietly opens 4 BLAS threads is 48 cores, not 12.  numpy's melbank matmul is
# the only BLAS call in the hot loop and it is tiny, so threading buys nothing
# and costs a co-tenant.  Measured: without this, 12 workers ran at ~370 % CPU
# each and the box load average went to 156.  Must precede `import numpy`.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bnasr.scoring import anonymize_process, load_manifest  # noqa: E402
from bnasr.common import train_surface  # noqa: E402

AUDIT = REPO / "data4/audit"

# ---------------------------------------------------------------------------
# Registered constants (SPEC 3 G0.3 / G0.4).  Each is asserted against the
# value written in SPEC.md, with the section quoted.
# ---------------------------------------------------------------------------
FUZZY_MIN_CHARS = 20          # SPEC 3 G0.3: "for strings >=20 chars"
FUZZY_MIN_SIM = 0.90          # SPEC 3 G0.3: "char similarity >=0.90"
assert FUZZY_MIN_CHARS == 20 and FUZZY_MIN_SIM == 0.90, \
    "SPEC 3 G0.3: fuzzy candidates are char similarity >=0.90 for strings >=20 chars"

CAL_CLIPS = 200               # SPEC 3 G0.4: "200 train-only clips"
CAL_MIN_RECOVERY = 0.99       # SPEC 3 G0.4: ">=99 % recovery per transform"
CAL_UNRELATED_PAIRS = 10000   # SPEC 3 G0.4: "10,000 unrelated"
CAL_SAMEPROMPT_PAIRS = 1000   # SPEC 3 G0.4: "+ 1,000 same-prompt pairs"
CAL_MAX_FALSE_CONFIRM = 0     # SPEC 3 G0.4: "0 false confirmations"
assert (CAL_CLIPS, CAL_MIN_RECOVERY, CAL_UNRELATED_PAIRS,
        CAL_SAMEPROMPT_PAIRS, CAL_MAX_FALSE_CONFIRM) == (200, 0.99, 10000, 1000, 0), \
    "SPEC 3 G0.4 calibration protocol"

MIN_MATCH_SPAN_S = 2.0        # workstream brief / council B item 3:
assert MIN_MATCH_SPAN_S == 2.0, "audio match requires a >=2 s matching span"

# SPEC 3 G0.4 names exactly six transforms; the ffmpeg filter chain for each is
# an implementation detail (NOT IN SPEC) but the SET is registered.
TRANSFORMS = ("mp3_48k", "gain_p6db", "trim_0p3s", "resample_8k", "speed_0p95", "speed_1p05")
assert len(TRANSFORMS) == 6, "SPEC 3 G0.4: six transforms"

# SPEC 2 protected panels.  FLEURS is split test/validation because SPEC 2
# protects the test split as the sealed headline and marks validation
# "never used (contaminated)" - a train row matching either is leakage.
PROTECTED_DEFAULT = {
    "D": REPO / "data2/val_corpus.json",
    "holdout": REPO / "data2/holdout_corpus.json",
    "K": REPO / "data4/kathbath/manifest_valid.jsonl",
    "R": REPO / "data4/respin/manifest_madasr_dev.jsonl",
    "M": REPO / "experiments/corpus/panel_M.jsonl",          # may not exist yet
    "FLEURS_test": None,        # synthesised from bench_assets/eval_manifest.json
    "FLEURS_validation": None,  # idem
}
FLEURS_MANIFEST = Path("/mnt/sdb/arafat/ehz/llm/bench_assets/eval_manifest.json")
FLEURS_AUDIO_DIR = Path("/mnt/sdb/arafat/ehz/llm/bench_assets/eval_fleurs_bn")
FLEURS_PANELS = ("FLEURS_test", "FLEURS_validation")

TRAIN_DEFAULT = [
    REPO / "data2/train_corpus_ready.json",
    REPO / "data4/macro/manifest_macro_validated.jsonl",
    REPO / "data4/macro/manifest_macro_filtered.jsonl",   # arriving
    REPO / "data4/respin/manifest_madasr.jsonl",          # arriving
]

MAX_PROCS = 12                # shared-box rule: at most 12 worker processes
NICE = 10                     # shared-box rule: CPU-heavy work nice -n 10


# ---------------------------------------------------------------------------
# Manifest plumbing
# ---------------------------------------------------------------------------
def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def read_rows(path, panel_name=None):
    """Rows of a manifest as {id, path, text, source}.

    FLEURS is not a NeMo manifest - it is the benchmark's eval_manifest.json
    ({"rows": [{file, split, raw_reference, ...}]}) - so it gets its own
    branch rather than a second copy of the schema somewhere else.
    """
    path = Path(path)
    if panel_name in FLEURS_PANELS:
        want = panel_name.split("_", 1)[1]
        blob = json.loads(FLEURS_MANIFEST.read_text(encoding="utf-8"))
        out = []
        for r in blob["rows"]:
            if r.get("split") != want:
                continue
            out.append({"id": r["file"],
                        "path": str(FLEURS_AUDIO_DIR / r["file"]),
                        "text": r.get("raw_reference", ""),
                        "source": panel_name})
        return out
    rows = []
    for r in load_manifest(path):
        ap = r["audio_filepath"]
        rows.append({"id": ap, "path": str(_resolve(ap)), "text": r.get("text", ""),
                     "source": r.get("source", path.stem)})
    return rows


def existing(paths):
    return [p for p in paths if Path(p).exists()]


# ---------------------------------------------------------------------------
# (a) decoded-PCM SHA-256           SPEC 1 dedup step (2), SPEC 3 G0.4 first half
# ---------------------------------------------------------------------------
# The canonical form is int16 mono 16 kHz, because that is what the corpus is
# and because soundfile's float->int16 conversion is deterministic, so a FLOAT
# wav and a PCM_16 wav of the same recording hash the same.  That matters here:
# the 2,019 fleurs_train rows are the ONLY FLOAT wavs in the the baseline corpus
# (measured: every other source is PCM_16), and FLEURS test/validation audio is
# PCM_16 - hashing the container bytes would silently miss exactly the overlap
# that matters most.
PCM_SR = 16000
PCM_CHANNELS = 1


def pcm_digest(path):
    """(sha256, n_samples, status).  status == 'ok' or a rejection reason.

    Rejections are recorded rather than raised: this pass is resumable over
    1.4 M files and one unreadable clip must not cost the other 1.4 M.  A
    rejected path is written to the TSV WITH its reason so that a resume skips
    it instead of re-opening it forever.
    """
    import soundfile as sf
    try:
        info = sf.info(path)
    except Exception as exc:                      # corrupt / missing / not audio
        return "", 0, f"unreadable:{type(exc).__name__}"
    if info.samplerate != PCM_SR:
        return "", 0, f"bad_samplerate:{info.samplerate}"
    if info.channels != PCM_CHANNELS:
        return "", 0, f"bad_channels:{info.channels}"
    if info.frames == 0:
        return "", 0, "empty"
    try:
        data, _ = sf.read(path, dtype="int16", always_2d=False)
    except Exception as exc:
        return "", 0, f"decode_failed:{type(exc).__name__}"
    if data.size == 0:
        return "", 0, "empty"
    peak = int(np.abs(data.astype(np.int32)).max())
    if peak == 0:
        return "", 0, "silent:allzero"
    if peak < 8:                                  # NOT IN SPEC; -72 dBFS floor,
        return "", 0, f"silent:peak{peak}"        # reported, never auto-dropped
    h = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    return h, int(data.size), "ok"


def _pcm_worker(item):
    key, path = item
    h, n, st = pcm_digest(path)
    return key, h, n, st


def load_pcm_tsv(tsv):
    """path -> (sha256, n_samples, status) from a possibly-partial TSV."""
    out = {}
    if not Path(tsv).exists():
        return out
    with open(tsv, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            out[parts[0]] = (parts[1], int(parts[2] or 0), parts[3])
    return out


def cmd_pcm_hash(args):
    t0 = time.time()
    tsv = Path(args.out)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    done = load_pcm_tsv(tsv)

    todo, keyed = [], {}
    for label, path, panel in _all_manifests(args):
        for r in read_rows(path, panel_name=panel if panel else None):
            keyed[r["id"]] = (label, r)
            if r["id"] not in done:
                todo.append((r["id"], r["path"]))
    print(f"[pcm-hash] {len(keyed)} rows across manifests; {len(done)} already hashed; "
          f"{len(todo)} to do", flush=True)

    n_new = 0
    if todo:
        with open(tsv, "a", encoding="utf-8") as out, \
                mp.Pool(min(args.procs, MAX_PROCS), initializer=_nice) as pool:
            for key, h, n, st in pool.imap_unordered(_pcm_worker, todo, chunksize=32):
                out.write(f"{key}\t{h}\t{n}\t{st}\n")
                n_new += 1
                if n_new % 20000 == 0:
                    out.flush()
                    print(f"[pcm-hash] {n_new}/{len(todo)} "
                          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    done = load_pcm_tsv(tsv)

    # ---- duplicate groups -------------------------------------------------
    by_hash = defaultdict(list)
    rejects = Counter()
    for key, (h, _n, st) in done.items():
        if key not in keyed:
            continue          # a stale row from an earlier manifest revision
        if st != "ok":
            rejects[st.split(":")[0]] += 1
            continue
        by_hash[h].append(key)

    train_labels = {lab for lab, _p, panel in _all_manifests(args) if panel is None}
    dup_train, cross = [], []
    for h, keys in by_hash.items():
        if len(keys) < 2:
            continue
        labs = {keyed[k][0] for k in keys}
        tr = [k for k in keys if keyed[k][0] in train_labels]
        pa = [k for k in keys if keyed[k][0] not in train_labels]
        if tr and pa:
            cross.append({"sha256": h,
                          "train": tr[:20], "n_train": len(tr),
                          "panels": sorted({keyed[k][0] for k in pa}),
                          "panel_ids": _redact(pa)})
        elif len(tr) > 1:
            dup_train.append({"sha256": h, "n": len(tr), "ids": tr[:20],
                              "labels": sorted(labs)})

    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tsv": str(tsv), "rows_hashed": len(done), "rows_new_this_run": n_new,
        "rejects_by_reason": dict(rejects),
        "exact_duplicate_groups_within_training": len(dup_train),
        "exact_duplicate_rows_within_training": sum(g["n"] - 1 for g in dup_train),
        "train_x_panel_hash_matches": len(cross),
        "examples_within_training": dup_train[:50],
        "train_x_panel": cross,
        "wall_s": round(time.time() - t0, 1),
    }
    _dump(AUDIT / "pcm_hash_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("examples_within_training", "train_x_panel")}, indent=1))
    return rep


def _redact(keys):
    """FLEURS ids only; SPEC 3 G0.4 'returning only ids/counts'."""
    return sorted(keys)[:50]


def _nice():
    try:
        os.nice(NICE)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# (b) text audit                                       SPEC 3 G0.3
# ---------------------------------------------------------------------------
# Three surfaces, deliberately nested from strictest to loosest:
#   surface   = train_surface() = metric_surface(NFC(t), False) + ws collapse.
#               This is the SPEC 1 RULING target surface; two rows equal here
#               are the same training target and step 4 must carve one out.
#   key_punct = punctuation DELETED (not spaced), joiners stripped, Bengali
#               digits folded to ASCII, Latin lower-cased, whitespace collapsed.
#               Catches tugstugi-style "algorithmic cleaning" that deleted a
#               hyphen where the frozen surface would have inserted a space
#               (council B risk 11, 9,363 affected OpenSLR rows).
#   key_tight = key_punct with ALL whitespace removed.  Catches pure spacing
#               differences, which Bengali repackagers introduce constantly.
_BN_DIGITS = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}
_JOINERS = dict.fromkeys([0x200C, 0x200D, 0x200B, 0xFEFF, 0x00AD], None)


def audit_keys(text):
    """(surface, key_punct, key_tight) for one raw reference."""
    surface = train_surface(text)
    t = unicodedata.normalize("NFC", text).translate(_JOINERS)
    t = "".join(ch for ch in t if not unicodedata.category(ch).startswith("P"))
    t = t.translate(_BN_DIGITS).lower()
    key_punct = " ".join(t.split())
    key_tight = "".join(key_punct.split())
    return surface, key_punct, key_tight


# --- indexed fuzzy retrieval ------------------------------------------------
# All-pairs over 1.4 M x 17 k is 2.4e10 comparisons and is not run.  Instead the
# SMALL side (the panels, ~17 k strings) is put in a character 5-gram inverted
# index and each train row queries it once.  The candidate threshold is not a
# guess: for q=5, if edit distance between A and B is <= k then the number of
# shared 5-grams is >= max(nA, nB) - 5k, where nX = len(X) - 4.  Normalised
# similarity >= 0.90 means k <= floor(0.10 * max(|A|, |B|)).  Anything below
# that many shared grams CANNOT reach 0.90, so the filter is exact (no false
# negatives) - it is a lower bound, not a heuristic cut-off.
QGRAM = 5                     # NOT IN SPEC; q of the q-gram bound above


def _grams(s, q=QGRAM):
    return [s[i:i + q] for i in range(len(s) - q + 1)]


class GramIndex:
    def __init__(self):
        self.post = defaultdict(list)   # gram -> [row_idx]
        self.strings = []
        self.meta = []

    def add(self, s, meta):
        i = len(self.strings)
        self.strings.append(s)
        self.meta.append(meta)
        for g in set(_grams(s)):
            self.post[g].append(i)

    def query(self, s):
        """Indices whose shared-gram count clears the exact q-gram lower bound."""
        gs = set(_grams(s))
        if not gs:
            return []
        cnt = Counter()
        for g in gs:
            for i in self.post.get(g, ()):
                cnt[i] += 1
        na = len(s) - QGRAM + 1
        out = []
        for i, c in cnt.items():
            nb = len(self.strings[i]) - QGRAM + 1
            mx = max(len(s), len(self.strings[i]))
            k = int(mx * (1.0 - FUZZY_MIN_SIM))
            if c >= max(na, nb) - QGRAM * k:
                out.append(i)
        return out


_TEXT_STATE = {}


def _text_init(state):
    _nice()
    _TEXT_STATE.update(state)


def _text_worker(chunk):
    from rapidfuzz.distance import Levenshtein
    idx = _TEXT_STATE["index"]
    exact = _TEXT_STATE["exact"]        # surface -> [(panel, pid)]
    kp = _TEXT_STATE["kp"]
    kt = _TEXT_STATE["kt"]
    hits = {"surface": [], "key_punct": [], "key_tight": [], "fuzzy": []}
    counts = Counter()
    for tid, label, text in chunk:
        s, a, b = audit_keys(text)
        if not s:
            counts["empty_surface"] += 1
            continue
        for name, table, val in (("surface", exact, s), ("key_punct", kp, a),
                                 ("key_tight", kt, b)):
            tgt = table.get(val)
            if tgt:
                for panel, pid in tgt:
                    counts[f"{name}|{label}|{panel}"] += 1
                    hits[name].append({"train_id": tid, "train_manifest": label,
                                       "panel": panel, "panel_id": pid})
        if len(s) >= FUZZY_MIN_CHARS:
            for i in idx.query(s):
                ps = idx.strings[i]
                if len(ps) < FUZZY_MIN_CHARS:
                    continue
                sim = Levenshtein.normalized_similarity(s, ps)
                if sim >= FUZZY_MIN_SIM:
                    panel, pid = idx.meta[i]
                    counts[f"fuzzy|{label}|{panel}"] += 1
                    hits["fuzzy"].append({"train_id": tid, "train_manifest": label,
                                          "panel": panel, "panel_id": pid,
                                          "similarity": round(float(sim), 4),
                                          "exact": bool(sim >= 0.99999)})
    return hits, counts


def cmd_text(args):
    t0 = time.time()
    AUDIT.mkdir(parents=True, exist_ok=True)

    idx = GramIndex()
    exact, kp, kt = defaultdict(list), defaultdict(list), defaultdict(list)
    panel_rows = {}
    for label, path, panel in _all_manifests(args):
        if panel is None:
            continue
        rows = read_rows(path, panel_name=panel)
        panel_rows[panel] = len(rows)
        for r in rows:
            s, a, b = audit_keys(r["text"])
            if not s:
                continue
            exact[s].append((panel, r["id"]))
            kp[a].append((panel, r["id"]))
            kt[b].append((panel, r["id"]))
            if len(s) >= FUZZY_MIN_CHARS:
                idx.add(s, (panel, r["id"]))
    print(f"[text] panels indexed: {panel_rows}; "
          f"{len(idx.strings)} fuzzy-indexed strings, "
          f"{len(idx.post)} distinct {QGRAM}-grams", flush=True)

    train = []
    train_counts = {}
    for label, path, panel in _all_manifests(args):
        if panel is not None:
            continue
        rows = read_rows(path)
        train_counts[label] = len(rows)
        train.extend((r["id"], label, r["text"]) for r in rows)
    print(f"[text] train rows: {train_counts} (total {len(train)})", flush=True)

    chunks = [train[i:i + 2000] for i in range(0, len(train), 2000)]
    all_hits = {"surface": [], "key_punct": [], "key_tight": [], "fuzzy": []}
    total = Counter()
    state = {"index": idx, "exact": dict(exact), "kp": dict(kp), "kt": dict(kt)}
    with mp.Pool(min(args.procs, MAX_PROCS), initializer=_text_init,
                 initargs=(state,)) as pool:
        for i, (hits, counts) in enumerate(
                pool.imap_unordered(_text_worker, chunks, chunksize=1)):
            for k in all_hits:
                all_hits[k].extend(hits[k])
            total.update(counts)
            if (i + 1) % 50 == 0:
                print(f"[text] {(i+1)*2000}/{len(train)} rows "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # FLEURS: ids/counts only.  A fuzzy pair carries no text in any branch, but
    # the FLEURS branch additionally drops the train id's TEXT (it never had
    # any) and is emitted to its own file so it can be reviewed without opening
    # the adjudication file.
    def split_fleurs(rows):
        keep = [h for h in rows if h["panel"] not in FLEURS_PANELS]
        fl = [h for h in rows if h["panel"] in FLEURS_PANELS]
        return keep, fl

    out, fleurs = {}, {}
    for k, v in all_hits.items():
        out[k], fleurs[k] = split_fleurs(v)

    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {"fuzzy_min_chars": FUZZY_MIN_CHARS, "fuzzy_min_similarity": FUZZY_MIN_SIM,
                 "qgram": QGRAM, "bound": "shared q-grams >= max(nA,nB) - q*k, k = floor((1-sim)*maxlen)"},
        "train_manifests": train_counts, "panel_rows": panel_rows,
        "counts": {k: v for k, v in sorted(total.items())},
        "totals": {k: len(v) for k, v in all_hits.items()},
        "totals_non_fleurs": {k: len(v) for k, v in out.items()},
        "totals_fleurs": {k: len(v) for k, v in fleurs.items()},
        "fleurs_matched_ids": sorted({h["panel_id"] for v in fleurs.values() for h in v}),
        "wall_s": round(time.time() - t0, 1),
    }
    _dump(AUDIT / "text_report.json", rep)
    _dump(AUDIT / "text_candidates.json",
          {k: v[:20000] for k, v in out.items()})
    print(json.dumps({k: rep[k] for k in
                      ("totals", "totals_non_fleurs", "totals_fleurs", "wall_s")}, indent=1))
    return rep


# ---------------------------------------------------------------------------
# (c) audio fingerprint                                SPEC 3 G0.4 second half
# ---------------------------------------------------------------------------
# fpcalc/chromaprint is checked for first (see have_fpcalc()).  It is NOT
# installed on this box (measured), and Chromaprint's own fingerprint is a
# whole-file 32-bit/frame sequence that does not support the ">=2 s matching
# SPAN inside a longer clip" requirement without a second matcher anyway, so
# the fallback here is a Shazam-style constellation: log-mel spectrogram ->
# local-maximum peaks -> (f_anchor, df, dt) landmark hashes -> inverted index
# -> time-offset histogram.  Properties that matter for the six transforms:
#   gain            local maxima are amplitude-ORDER, not amplitude-VALUE
#   mp3 48 kbps     peaks are the loudest bins, which is what the codec keeps
#   8 kHz resample  the analysis band stops at FMAX = 3800 Hz, below Nyquist/2
#                   of 8 kHz, so an 8 kHz-decimated copy loses nothing we use
#   0.3 s trim      absorbed by the offset histogram
#   speed +-5 %     ffmpeg atempo preserves pitch, so f is unchanged and only
#                   dt scales; the INDEX side is therefore built at three time
#                   scales and the winning scale is reported with the match
FP = {                            # NOT IN SPEC - calibrated, then frozen
    "sr": 16000, "n_fft": 1024, "hop": 256, "n_mels": 128,
    "fmin": 60.0, "fmax": 3800.0,
    "nbhd_f": 9, "nbhd_t": 9, "peaks_per_sec": 14,
    "whiten_win": 51,          # per-mel-bin temporal median window, in frames
    "peak_pct": 70.0,          # keep peaks above this percentile of whitened S
    "fan_out": 6, "dt_min": 2, "dt_max": 63, "df_max": 63,
    "max_hash_multiplicity": 3,
    # QUERY-side time scales.  atempo=0.95 stretches the probe, so its dt is
    # dt_orig/0.95; multiplying the QUERY dt by 0.95 restores dt_orig and the
    # index can stay at scale 1.0.  Scaling the query instead of the index is
    # not cosmetic: the index holds ~17 k panel clips and tripling it would
    # cost ~1 GB of postings, whereas tripling the query costs only CPU on a
    # pass that is already disk-bound.
    "scales": (0.95, 1.0, 1.05),
    "offset_tol": 2,
}
FPS = FP["sr"] / FP["hop"]                       # 62.5 analysis frames / second
MIN_SPAN_FRAMES = int(round(MIN_MATCH_SPAN_S * FPS))
CAL_MIN_DUR_S = 4.0           # NOT IN SPEC; see cmd_fp_calibrate()
_MEL = {}


def have_fpcalc():
    from shutil import which
    return which("fpcalc")


def _melbank():
    if "fb" not in _MEL:
        import librosa
        _MEL["fb"] = librosa.filters.mel(sr=FP["sr"], n_fft=FP["n_fft"],
                                         n_mels=FP["n_mels"], fmin=FP["fmin"],
                                         fmax=FP["fmax"]).astype(np.float32)
    return _MEL["fb"]


def fp_peaks(samples):
    """(freq_bin, time_frame) constellation peaks of one waveform.

    Two steps that the textbook constellation omits and that this corpus
    forced (measured, calibration round 1): a stationary low-frequency hum
    produces a local maximum in EVERY frame of the same mel bin, so two
    unrelated recordings from the same collection shared >1,200 landmarks in a
    single offset bin - a guaranteed false confirmation.  Subtracting a
    per-mel-bin running median (spectral whitening) removes anything constant
    in time, and a percentile floor on the whitened surface removes the
    near-flat ties inside silence.
    """
    from scipy.ndimage import maximum_filter, median_filter
    x = samples.astype(np.float32)
    if x.size < FP["n_fft"]:
        x = np.pad(x, (0, FP["n_fft"] - x.size))
    n_fft, hop = FP["n_fft"], FP["hop"]
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (x.size - n_fft) // hop
    if n < 1:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0])) * win
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32).T
    mel = _melbank() @ spec
    S = np.log1p(mel * 1000.0)          # gain-monotone: peak ORDER is invariant
    w = min(FP["whiten_win"], max(3, (S.shape[1] // 2) * 2 + 1))
    S = S - median_filter(S, size=(1, w), mode="nearest")
    mx = maximum_filter(S, size=(FP["nbhd_f"], FP["nbhd_t"]), mode="nearest")
    floor = np.percentile(S, FP["peak_pct"])
    cand = (S == mx) & (S > floor)
    f, t = np.nonzero(cand)
    if f.size == 0:
        return f.astype(np.int32), t.astype(np.int32)
    amp = S[f, t]
    blk = (t / FPS).astype(np.int64)
    order = np.lexsort((-amp, blk))
    f, t, blk = f[order], t[order], blk[order]
    rank = np.arange(blk.size) - np.searchsorted(blk, blk, side="left")
    keep = rank < FP["peaks_per_sec"]
    f, t = f[keep], t[keep]
    o = np.argsort(t, kind="stable")
    return f[o].astype(np.int32), t[o].astype(np.int32)


def fp_hashes(samples, scale=1.0, peaks=None):
    """(hash, anchor_frame) landmark pairs, optionally time-rescaled.

    hash = f1(7 bits) | (df+64)(7 bits) | dt(6 bits) -> 20 bits.

    A non-unit scale rescales the anchor time as well as dt.  Scaling only dt
    is the mistake that took speed recovery to 0.01 in calibration round 2: the
    matcher's evidence is a CONSTANT time offset between query and index, and
    if the query's anchors are still on the stretched clock the offset drifts
    linearly across the clip and every landmark lands in a different bin.

    For a non-unit scale BOTH the floor and the ceil of dt*scale are emitted:
    dt lives on a 1-frame (16 ms) grid, and a 5 % rescale of dt=40 moves it by
    2 frames, so rounding alone silently loses the landmark.  Emitting both
    neighbours is what took speed_1p05 recovery from 0.935 to the registered
    >=0.99 (measured, calibration rounds 1 and 2).

    Landmarks whose hash repeats more than `max_hash_multiplicity` times in one
    clip are dropped: a repeated hash carries no positional information and is
    exactly what an unrelated pair can accumulate in a single offset bin.
    """
    f, t = fp_peaks(samples) if peaks is None else peaks
    if f.size < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    H, A = [], []
    dtmin, dtmax, dfmax, fan = FP["dt_min"], FP["dt_max"], FP["df_max"], FP["fan_out"]
    n = f.size
    for i in range(n):
        ti, fi = int(t[i]), int(f[i])
        j, got = i + 1, 0
        while j < n and got < fan:
            dt = int(t[j]) - ti
            if dt < dtmin:
                j += 1
                continue
            if dt > dtmax:
                break
            df = int(f[j]) - fi
            if abs(df) <= dfmax:
                base = (fi & 0x7F) << 13 | ((df + 64) & 0x7F) << 6
                sd = dt * scale
                sti = ti if scale == 1.0 else int(round(ti * scale))
                for sdt in ({dt} if scale == 1.0 else
                            {int(np.floor(sd)), int(np.ceil(sd))}):
                    if dtmin <= sdt <= dtmax:
                        H.append(base | (sdt & 0x3F))
                        A.append(sti)
                got += 1
            j += 1
    if not H:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    h = np.asarray(H, np.int64)
    a = np.asarray(A, np.int64)
    cnt = Counter(h.tolist())
    keep = np.fromiter((cnt[v] <= FP["max_hash_multiplicity"] for v in h.tolist()),
                       bool, h.size)
    return h[keep], a[keep]


def load_samples(path):
    import soundfile as sf
    d, sr = sf.read(path, dtype="float32", always_2d=False)
    if d.ndim > 1:
        d = d.mean(axis=1)
    assert sr == FP["sr"], f"{path}: {sr} Hz, expected {FP['sr']}"
    return d


class FpIndex:
    """Sorted-array inverted index; forked workers share it copy-on-write."""

    def __init__(self, hashes, clips, times, scales):
        o = np.argsort(hashes, kind="stable")
        self.h = hashes[o]
        self.c = clips[o]
        self.t = times[o]
        self.s = scales[o]
        self.n_clips = int(clips.max()) + 1 if clips.size else 0

    @staticmethod
    def build(items, procs):
        """items: [(clip_idx, path)] -> FpIndex at scale 1.0.

        The index is unscaled on purpose; the QUERY carries the +-5 % scales
        (see FP["scales"]).  Storing three scaled copies of a 17 k-clip panel
        index costs ~1 GB of RAM for nothing.
        """
        H, C, T = [], [], []
        with mp.Pool(min(procs, MAX_PROCS), initializer=_nice) as pool:
            for ci, h, a in pool.imap_unordered(_index_worker, items, chunksize=8):
                if h.size:
                    H.append(h); T.append(a)
                    C.append(np.full(h.size, ci, np.int32))
        if not H:
            return FpIndex(np.empty(0, np.int64), np.empty(0, np.int32),
                           np.empty(0, np.int64), np.empty(0, np.int8))
        h = np.concatenate(H)
        return FpIndex(h, np.concatenate(C), np.concatenate(T),
                       np.zeros(h.size, np.int8))

    def match(self, qh, qa):
        """Strongest (clip, offset) bin for one query -> (clip, hits, span, 0).

        The offset histogram is what turns landmark collisions into evidence:
        an unrelated pair scatters its shared hashes across every offset, a
        true copy piles them all onto ONE offset (the trim/padding delta).
        `span` is measured over the QUERY anchor times inside the winning bin,
        so it is the duration of audio that actually lines up - the >=2 s rule.
        """
        if qh.size == 0 or self.h.size == 0:
            return -1, 0, 0, 0
        lo = np.searchsorted(self.h, qh, "left")
        hi = np.searchsorted(self.h, qh, "right")
        cnt = hi - lo
        if int(cnt.sum()) == 0:
            return -1, 0, 0, 0
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])
        qrep = np.repeat(qa, cnt)
        off = self.t[idx] - qrep
        key = (self.c[idx].astype(np.int64) << 24) \
            | ((off // FP["offset_tol"]) + (1 << 20))
        order = np.argsort(key, kind="stable")
        key, qrep = key[order], qrep[order]
        bnd = np.flatnonzero(np.diff(key)) + 1
        starts = np.concatenate(([0], bnd))
        ends = np.concatenate((bnd, [key.size]))
        counts = ends - starts
        best = int(np.argmax(counts))
        b0, b1 = starts[best], ends[best]
        span = int(qrep[b0:b1].max() - qrep[b0:b1].min())
        return int(key[b0]) >> 24, int(counts[best]), span, 0


_FPI = {}


def _index_worker(item):
    ci, path = item
    _nice()
    h, a = fp_hashes(load_samples(path), 1.0)
    return ci, h, a


def _match_init(index):
    _nice()
    _FPI["idx"] = index


def _match_worker(item):
    """Best panel/index clip for one query, maximised over FP["scales"]."""
    qi, path, want_pcm = item
    try:
        x = load_samples(path)
    except Exception as exc:
        return qi, -1, 0, 0, 0, "", 0, f"unreadable:{type(exc).__name__}"
    pk = fp_peaks(x)                      # computed once, reused for all scales
    best = (-1, 0, 0, 0)
    for si, sc in enumerate(FP["scales"]):
        h, a = fp_hashes(x, sc, peaks=pk)
        c, hits, span, _ = _FPI["idx"].match(h, a)
        if hits > best[1]:
            best = (c, hits, span, si)
    dig, ns, st = "", 0, "ok"
    if want_pcm:
        dig, ns, st = pcm_digest(path)
    return qi, best[0], best[1], best[2], best[3], dig, ns, st


# --- calibration ------------------------------------------------------------
def _ffmpeg(src, dst, filt, extra=()):
    import subprocess
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if filt:
        cmd += ["-af", filt]
    cmd += list(extra) + ["-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)


def _make_transform(src, outdir, kind, i):
    """One of the six SPEC 3 G0.4 transforms; returns the 16 kHz wav path."""
    tmp = outdir / f"{i:04d}_{kind}"
    wav = Path(f"{tmp}.wav")
    if wav.exists():
        return wav
    if kind == "mp3_48k":
        m = Path(f"{tmp}.mp3")
        _ffmpeg(src, m, None, ["-b:a", "48k"])
        _ffmpeg(m, wav, None)
        m.unlink(missing_ok=True)
    elif kind == "gain_p6db":
        _ffmpeg(src, wav, "volume=6dB")
    elif kind == "trim_0p3s":
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.3",
                        "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    elif kind == "resample_8k":
        eight = Path(f"{tmp}_8k.wav")
        _ffmpeg(src, eight, None, ["-ar", "8000"])
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(eight),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        eight.unlink(missing_ok=True)
    elif kind == "speed_0p95":
        _ffmpeg(src, wav, "atempo=0.95")
    elif kind == "speed_1p05":
        _ffmpeg(src, wav, "atempo=1.05")
    else:
        raise ValueError(kind)
    return wav


def _tf_worker(job):
    src, outdir, kind, i = job
    _nice()
    try:
        return kind, i, str(_make_transform(Path(src), Path(outdir), kind, i))
    except Exception as exc:
        return kind, i, f"ERROR:{exc}"


def cmd_fp_calibrate(args):
    """SPEC 3 G0.4: calibrate BEFORE any protected panel is touched.

    Runs entirely on train-only clips.  Emits fp_thresholds.json, which every
    later `fingerprint match` loads; a match run with no thresholds file is
    refused, so the ordering rule ("thresholds are not tuned using test
    matches", council B item 4) is enforced by the code, not by discipline.
    """
    import random
    t0 = time.time()
    rng = random.Random(args.seed)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for label, path, panel in _all_manifests(args):
        if panel is None:
            train_rows.extend((label, r) for r in read_rows(path))
    assert train_rows, "no training manifests found to calibrate on"
    # Only clips long enough for the registered >=2 s span to be REACHABLE.
    # A 2.7 s clip cannot exhibit a 2 s matching span once 0.3 s is trimmed, so
    # calibrating on it would measure the span rule's geometry rather than the
    # fingerprint's robustness.  The consequence is a stated LIMITATION, not a
    # hidden one: recordings shorter than CAL_MIN_DUR_S can never be
    # *confirmed* by this audit, only flagged unresolved, and the share of the
    # corpus that is that short is reported in fp_thresholds.json.
    pool_rows = [r for _l, r in train_rows]
    rng.shuffle(pool_rows)
    picked, seen = [], set()
    for r in pool_rows:
        if len(picked) >= CAL_CLIPS:
            break
        if r["path"] in seen or not Path(r["path"]).exists():
            continue
        try:
            import soundfile as sf
            if sf.info(r["path"]).duration < CAL_MIN_DUR_S:
                continue
        except Exception:
            continue
        seen.add(r["path"])
        picked.append(r)
    assert len(picked) == CAL_CLIPS, f"SPEC 3 G0.4 needs {CAL_CLIPS} clips, got {len(picked)}"

    jobs = [(picked[i]["path"], str(work), k, i)
            for k in TRANSFORMS for i in range(CAL_CLIPS)]
    made = defaultdict(dict)
    with mp.Pool(min(args.procs, MAX_PROCS)) as pool:
        for kind, i, out in pool.imap_unordered(_tf_worker, jobs, chunksize=4):
            made[kind][i] = out
    bad = {k: [i for i, v in d.items() if v.startswith("ERROR")] for k, d in made.items()}
    assert not any(bad.values()), f"transform generation failed: {bad}"

    idx = FpIndex.build([(i, picked[i]["path"]) for i in range(CAL_CLIPS)], args.procs)
    print(f"[calibrate] index: {idx.h.size} postings over {CAL_CLIPS} clips "
          f"x {len(FP['scales'])} scales ({(time.time()-t0)/60:.1f} min)", flush=True)

    # --- positives: every transformed copy must find its own original -------
    pos = {}
    hit_values = []
    with mp.Pool(min(args.procs, MAX_PROCS), initializer=_match_init,
                 initargs=(idx,)) as pool:
        for kind in TRANSFORMS:
            items = [(i, made[kind][i], False) for i in range(CAL_CLIPS)]
            res = list(pool.imap_unordered(_match_worker, items, chunksize=4))
            ok = 0
            rows = []
            for qi, c, hits, span, si, *_ in res:
                right = (c == qi) and span >= MIN_SPAN_FRAMES
                ok += bool(right)
                rows.append({"i": qi, "matched": int(c), "hits": int(hits),
                             "span_frames": int(span), "scale_idx": int(si)})
                if right:
                    hit_values.append(int(hits))
            pos[kind] = {"recovery": ok / CAL_CLIPS, "detail": rows}
            print(f"[calibrate] {kind}: correct-clip+span recovery "
                  f"{ok}/{CAL_CLIPS}", flush=True)

    # --- negatives ---------------------------------------------------------
    # (i) 10,000 unrelated pairs: random query clip vs an index of OTHER clips.
    #     Implemented as "query clip i against the full index, best NON-i clip".
    # (ii) 1,000 same-normalised-sentence / different-recording pairs, which is
    #     the hard negative: two speakers reading one prompt must NOT confirm.
    unrelated = _unrelated_scores(picked, args.procs, CAL_UNRELATED_PAIRS)
    sameprompt_pairs = _same_prompt_pairs(train_rows, CAL_SAMEPROMPT_PAIRS, rng)
    sameprompt = _pair_scores(sameprompt_pairs, args.procs)

    # --- (iii) negatives at OPERATIONAL index size -------------------------
    # A landmark index's chance-collision rate grows with the number of clips
    # in it: the more clips, the more likely SOME clip accumulates k collisions
    # in one offset bin.  Calibration round 4 measured negatives against a
    # 100-clip index (max 5 hits -> MIN_HITS 6) and the first real run against
    # the 17,210-clip panel index returned 17,920 "unresolved" of 29,462
    # queries - 61 % of a bucket, which is noise, not evidence.  So the
    # negative distribution is re-measured against a TRAIN-ONLY index of the
    # size the audit will actually face.  Train-only is what keeps this legal:
    # SPEC 3 G0.4 forbids calibrating on the protected panels themselves.
    excl = {r["path"] for r in picked} | {p for pr in sameprompt_pairs for p in pr}
    neg_pool = [r["path"] for r in pool_rows if r["path"] not in excl]
    neg_pool = neg_pool[:args.neg_index_clips]
    big = FpIndex.build(list(enumerate(neg_pool)), args.procs)
    print(f"[calibrate] operational-scale negative index: {len(neg_pool)} "
          f"train-only clips, {big.h.size} postings", flush=True)
    with mp.Pool(min(args.procs, MAX_PROCS), initializer=_match_init,
                 initargs=(big,)) as pool:
        items = [(i, picked[i]["path"], False) for i in range(CAL_CLIPS)]
        scaled = [(int(h), int(sp)) for _qi, c, h, sp, _si, *_ in
                  pool.imap_unordered(_match_worker, items, chunksize=4) if c >= 0]
    neg_hits = [h for h, _sp in unrelated + sameprompt + scaled]
    unrel_conf = [(h, sp) for h, sp in unrelated if sp >= MIN_SPAN_FRAMES]
    samep_conf = [(h, sp) for h, sp in sameprompt if sp >= MIN_SPAN_FRAMES]
    scaled_conf = [(h, sp) for h, sp in scaled if sp >= MIN_SPAN_FRAMES]
    neg_conf = unrel_conf + samep_conf + scaled_conf
    # MIN_HITS must clear the WHOLE negative distribution, not only the
    # span-clearing tail: a below-span negative at k hits is what produces an
    # "unresolved" row, and SPEC 3 G0.4's pass rule counts unresolved matches
    # as failures.  So the threshold is set above the strongest negative of
    # ANY kind seen at operational index size.
    max_neg = max([h for h, _ in neg_conf]
                  + [h for h, _sp in scaled], default=0)

    min_pos = min(hit_values) if hit_values else 0
    # MIN_HITS is set strictly above the strongest negative and at/below the
    # weakest true positive.  If those two windows cross, calibration FAILS -
    # the threshold is never split to make the numbers look good.
    min_hits = max_neg + 1
    # The unresolved band is defined by SPAN, not by a second hit threshold.
    # Setting REVIEW_HITS below MIN_HITS would put the review band inside the
    # measured negative distribution (max_hits among span-clearing negatives is
    # `max_neg`), which floods adjudication with noise that calibration already
    # showed to be indistinguishable from chance.  So: enough landmarks but too
    # short a span -> UNRESOLVED (the short-clip and partial-overlap case);
    # fewer landmarks than any calibrated positive -> no match.
    review_hits = min_hits
    recovery_ok = all(v["recovery"] >= CAL_MIN_RECOVERY for v in pos.values())
    sep_ok = min_pos >= min_hits
    thr = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "landmark_constellation",
        "fpcalc_present": bool(have_fpcalc()),
        "fp_params": {k: list(v) if isinstance(v, tuple) else v for k, v in FP.items()},
        "fps": FPS, "min_span_s": MIN_MATCH_SPAN_S, "min_span_frames": MIN_SPAN_FRAMES,
        "seed": args.seed, "n_clips": CAL_CLIPS,
        "recovery": {k: round(v["recovery"], 4) for k, v in pos.items()},
        "recovery_required": CAL_MIN_RECOVERY,
        "recovery_pass": recovery_ok,
        "negatives": {"unrelated_pairs_implied": CAL_CLIPS * (CAL_CLIPS - 1),
                      "unrelated_pairs_required": CAL_UNRELATED_PAIRS,
                      "unrelated_queries_scored": len(unrelated),
                      "same_prompt_pairs": len(sameprompt),
                      "same_prompt_pairs_required": CAL_SAMEPROMPT_PAIRS,
                      "operational_index_clips": len(neg_pool),
                      "operational_queries_scored": len(scaled),
                      "max_hits_operational_index": max([h for h, _ in scaled], default=0),
                      "span_clearing_operational": len(scaled_conf),
                      "span_clearing_unrelated": len(unrel_conf),
                      "span_clearing_same_prompt": len(samep_conf),
                      "max_hits_unrelated": max([h for h, _ in unrelated], default=0),
                      "max_hits_same_prompt": max([h for h, _ in sameprompt], default=0),
                      "max_hits_among_span_clearing_negatives": int(max_neg),
                      "p999_hits": int(np.percentile(neg_hits, 99.9)) if neg_hits else 0},
        "min_positive_hits": int(min_pos),
        "MIN_HITS": int(min_hits), "REVIEW_HITS": int(review_hits),
        "separation_pass": bool(sep_ok),
        "false_confirmations": 0 if sep_ok else len(neg_conf),
        "short_clip_limitation": {
            "min_confirmable_duration_s": MIN_MATCH_SPAN_S,
            "calibration_min_duration_s": CAL_MIN_DUR_S,
            "note": ("a recording shorter than the >=2 s span rule cannot be "
                     "CONFIRMED by this audit; such a match is emitted as "
                     "unresolved and must be adjudicated")},
        "PASS": bool(recovery_ok and sep_ok
                     and len(sameprompt) >= CAL_SAMEPROMPT_PAIRS),
        "wall_s": round(time.time() - t0, 1),
    }
    _dump(AUDIT / "fp_thresholds.json", thr)
    _dump(AUDIT / "fp_calibration_detail.json",
          {k: v["detail"] for k, v in pos.items()})
    print(json.dumps({k: v for k, v in thr.items() if k != "fp_params"}, indent=1))
    return thr


def _unrelated_scores(picked, procs, n_pairs):
    """(hits, span) of the strongest UNRELATED bin for each calibration clip.

    One clip queried against the other 199 yields 199 unrelated pairs, so the
    200 clips imply 39,800 ordered pairs - well past SPEC 3 G0.4's 10,000.
    Only the ARGMAX over each query's pairs can breach a threshold, so scoring
    the best wrong bin per query is equivalent to scoring every pair for the
    purpose of "0 false confirmations", and it is 200 queries instead of
    39,800.  Self-matches are removed by construction: the index is built from
    the OTHER half of the clip list (two complementary halves), so a query's
    own recording is never in the index it is scored against.
    """
    n_implied = len(picked) * (len(picked) - 1)
    assert n_implied >= n_pairs, \
        f"SPEC 3 G0.4 needs {n_pairs} unrelated pairs; only {n_implied} available"
    return _masked_self_scores(picked, procs)


def _masked_self_scores(picked, procs):
    """Best-wrong-clip (hits, span) per query, with the query's own clip absent."""
    res = []
    n = len(picked)
    half = n // 2
    for a, b in ((slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))):
        qs = list(range(n))[a]
        idxs = list(range(n))[b]
        sub = FpIndex.build([(i, picked[i]["path"]) for i in idxs], procs)
        remap = {k: i for k, i in enumerate(idxs)}
        with mp.Pool(min(procs, MAX_PROCS), initializer=_match_init,
                     initargs=(sub,)) as pool:
            items = [(i, picked[i]["path"], False) for i in qs]
            for qi, c, hits, span, si, *_ in pool.imap_unordered(_match_worker, items, chunksize=4):
                res.append((int(hits), int(span)))
    return res


def _same_prompt_pairs(train_rows, n, rng):
    """(pathA, pathB): distinct RECORDINGS of one normalised sentence.

    This is the hard negative of SPEC 3 G0.4: two speakers reading the same
    prompt must not be confirmed as the same recording.  Three properties are
    enforced, each of which a round of calibration proved necessary:

      * ONE pair per sentence group, and the A-side and B-side path sets are
        disjoint.  Round 2 emitted chained pairs (g0,g1),(g1,g2),... so a query
        was frequently present in the index as some other pair's B side and
        matched ITSELF - 556 "false confirmations" that were self-matches.
      * pairs whose two clips have the same decoded-PCM SHA-256 are dropped:
        those are genuine duplicate recordings (3 of 150 sampled), i.e. exactly
        what the audit exists to find, and letting them raise the threshold
        would blind the audit to the thing it is calibrating for.
      * pairs with an identical frame count are dropped for the same reason at
        one remove: two independent human takes of one sentence essentially
        never agree to the sample, so equal length signals a re-encode.
    """
    import soundfile as sf
    groups = defaultdict(list)
    for _label, r in train_rows:
        s = train_surface(r["text"])
        if len(s) >= FUZZY_MIN_CHARS:
            groups[s].append(r["path"])
    cands = [sorted(set(g)) for g in groups.values() if len(set(g)) >= 2]
    rng.shuffle(cands)
    pairs, used = [], set()
    dropped = Counter()
    for g in cands:
        if len(pairs) >= n:
            break
        a, b = g[0], g[1]
        if a in used or b in used:
            dropped["overlapping_path"] += 1
            continue
        try:
            if sf.info(a).frames == sf.info(b).frames:
                dropped["equal_frame_count"] += 1
                continue
            if pcm_digest(a)[0] == pcm_digest(b)[0]:
                dropped["identical_pcm"] += 1
                continue
        except Exception:
            dropped["unreadable"] += 1
            continue
        used.add(a); used.add(b)
        pairs.append((a, b))
    print(f"[calibrate] same-prompt pairs: {len(pairs)} kept, dropped {dict(dropped)}",
          flush=True)
    return pairs


def _pair_scores(pairs, procs):
    """(hits, span) for A-vs-B over same-prompt pairs, A never in the index.

    Round 1 of calibration indexed A and B together and then discarded every
    pair whose winner was A itself - which was almost all of them, leaving 21
    of the required 1,000 pairs.  The B side is now its own index, so the
    reported score really is A-against-B.
    """
    if not pairs:
        return []
    bs = sorted({b for _a, b in pairs})
    bi = {p: i for i, p in enumerate(bs)}
    idx = FpIndex.build([(bi[p], p) for p in bs], procs)
    out = []
    with mp.Pool(min(procs, MAX_PROCS), initializer=_match_init, initargs=(idx,)) as pool:
        items = [(k, a, False) for k, (a, _b) in enumerate(pairs)]
        for _qi, c, hits, span, _si, *_ in pool.imap_unordered(
                _match_worker, items, chunksize=8):
            if c >= 0:
                out.append((int(hits), int(span)))
    return out


def cmd_fp_match(args):
    """train x protected panels.  Refuses to run without frozen thresholds."""
    t0 = time.time()
    thr_path = AUDIT / "fp_thresholds.json"
    assert thr_path.exists(), (
        "SPEC 3 G0.4: run `fingerprint calibrate` first - thresholds must be "
        "frozen before any protected panel is touched")
    thr = json.loads(thr_path.read_text())
    assert thr["PASS"], f"calibration did not pass; matching is not authorised: {thr['recovery']}"
    min_hits, review_hits = thr["MIN_HITS"], thr["REVIEW_HITS"]

    panels, pmeta = [], []
    for label, path, panel in _all_manifests(args):
        if panel is None:
            continue
        for r in read_rows(path, panel_name=panel):
            if Path(r["path"]).exists():
                pmeta.append((panel, r["id"]))
                panels.append(r["path"])
    assert panels, "no protected-panel audio found"
    idx = FpIndex.build(list(enumerate(panels)), args.procs)
    print(f"[fp-match] panel index: {len(panels)} clips, {idx.h.size} postings "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    train = []
    for label, path, panel in _all_manifests(args):
        if panel is not None:
            continue
        for r in read_rows(path):
            train.append((label, r["id"], r["path"]))
    if args.limit:
        train = train[:args.limit]
    print(f"[fp-match] querying {len(train)} training clips", flush=True)

    pcm_out = None
    if args.emit_pcm_hash:
        # the query pass already has the samples in memory; writing the PCM
        # digest here saves a SECOND full read of ~1,800 h of audio, which on
        # this contended disk is the single largest cost in the whole audit.
        pcm_out = open(args.pcm_tsv, "a", encoding="utf-8")

    confirmed, unresolved = [], []
    n_bad = Counter()
    items = [(i, p, bool(args.emit_pcm_hash)) for i, (_l, _id, p) in enumerate(train)]
    done = 0
    with mp.Pool(min(args.procs, MAX_PROCS), initializer=_match_init,
                 initargs=(idx,)) as pool:
        for qi, c, hits, span, si, dig, ns, st in pool.imap_unordered(
                _match_worker, items, chunksize=16):
            done += 1
            label, tid, tpath = train[qi]
            if pcm_out is not None:
                pcm_out.write(f"{tid}\t{dig}\t{ns}\t{st}\n")
            if st != "ok":
                n_bad[st.split(":")[0]] += 1
            if c < 0:
                continue
            panel, pid = pmeta[c]
            rec = {"train_id": tid, "train_manifest": label, "panel": panel,
                   "panel_id": pid, "hits": int(hits),
                   "span_s": round(span / FPS, 2),
                   "scale": FP["scales"][si] if si < len(FP["scales"]) else None}
            if hits >= min_hits and span >= MIN_SPAN_FRAMES:
                confirmed.append(rec)
            elif hits >= review_hits:
                unresolved.append(rec)
            if done % 20000 == 0:
                print(f"[fp-match] {done}/{len(train)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    if pcm_out is not None:
        pcm_out.close()

    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {"MIN_HITS": min_hits, "REVIEW_HITS": review_hits,
                       "min_span_s": MIN_MATCH_SPAN_S},
        "panel_clips": len(panels), "train_clips_queried": len(train),
        "confirmed": len(confirmed), "unresolved": len(unresolved),
        "audio_rejects": dict(n_bad),
        "confirmed_detail": confirmed[:5000],
        "unresolved_detail": unresolved[:5000],
        "wall_s": round(time.time() - t0, 1),
    }
    _dump(AUDIT / "fp_match_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.endswith("_detail")}, indent=1))
    return rep


# ---------------------------------------------------------------------------
# (d) report                                           SPEC 3 G0.3 + G0.4
# ---------------------------------------------------------------------------
def cmd_report(args):
    def maybe(p):
        p = AUDIT / p
        return json.loads(p.read_text()) if p.exists() else None

    pcm, txt, cal, fpm = (maybe("pcm_hash_report.json"), maybe("text_report.json"),
                          maybe("fp_thresholds.json"), maybe("fp_match_report.json"))
    text_unresolved = None
    if txt:
        # An adjudicated file marks candidates resolved; until one exists every
        # non-identical fuzzy candidate is UNRESOLVED, which is the conservative
        # reading of "zero unresolved matches" (SPEC 3 G0.3).
        adj = maybe("text_adjudication.json") or {"resolved": []}
        resolved = {(a["train_id"], a["panel"], a["panel_id"]) for a in adj["resolved"]}
        cand = json.loads((AUDIT / "text_candidates.json").read_text()) \
            if (AUDIT / "text_candidates.json").exists() else {}
        allc = [h for k in ("surface", "key_punct", "key_tight", "fuzzy") for h in cand.get(k, [])]
        text_unresolved = len([h for h in allc
                               if (h["train_id"], h["panel"], h["panel_id"]) not in resolved])
        text_unresolved += sum(txt["totals_fleurs"].values())

    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "experiments/SPEC.md 1 (dedup 2-4), 3 G0.3, 3 G0.4",
        "components": {
            "pcm_hash": bool(pcm), "text": bool(txt),
            "fingerprint_calibration": bool(cal), "fingerprint_match": bool(fpm)},
        "pcm_hash": None if not pcm else {
            k: pcm[k] for k in ("rows_hashed", "rejects_by_reason",
                                "exact_duplicate_groups_within_training",
                                "exact_duplicate_rows_within_training",
                                "train_x_panel_hash_matches")},
        "text": None if not txt else {
            "totals": txt["totals"], "totals_fleurs": txt["totals_fleurs"],
            "unresolved": text_unresolved},
        "fingerprint_calibration": None if not cal else {
            k: cal[k] for k in ("recovery", "recovery_pass", "negatives",
                                "MIN_HITS", "separation_pass", "PASS")},
        "fingerprint_match": None if not fpm else {
            k: fpm[k] for k in ("panel_clips", "train_clips_queried",
                                "confirmed", "unresolved")},
    }
    checks = {
        "G0.3_zero_unresolved_text_matches":
            None if txt is None else (text_unresolved == 0),
        "G0.4_calibration_passed": None if cal is None else bool(cal["PASS"]),
        "G0.4_zero_pcm_hash_train_x_panel":
            None if pcm is None else (pcm["train_x_panel_hash_matches"] == 0),
        "G0.4_zero_confirmed_audio_matches":
            None if fpm is None else (fpm["confirmed"] == 0),
        "G0.4_zero_unresolved_audio_matches":
            None if fpm is None else (fpm["unresolved"] == 0),
    }
    rep["checks"] = checks
    rep["PASS"] = (all(v is True for v in checks.values())
                   if all(v is not None for v in checks.values()) else None)
    rep["incomplete_because"] = [k for k, v in checks.items() if v is None]
    _dump(AUDIT / "leakage_report.json", rep)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _all_manifests(args):
    """[(label, path, panel_or_None)] for every manifest THIS run touches.

    A manifest that does not exist yet (macro_filtered, madasr train, panel M)
    is skipped with a printed line rather than an error: the ingestion jobs are
    still running and the audit must be runnable, and resumable, today.
    """
    out = []
    for p in (args.train or [str(x) for x in TRAIN_DEFAULT]):
        if Path(p).exists():
            out.append((Path(p).stem, Path(p), None))
        else:
            print(f"[skip] train manifest not present yet: {p}", flush=True)
    if getattr(args, "no_panels", False):
        return out
    for name, p in PROTECTED_DEFAULT.items():
        if name in FLEURS_PANELS:
            if FLEURS_MANIFEST.exists():
                out.append((name, FLEURS_MANIFEST, name))
            else:
                print(f"[skip] FLEURS manifest missing: {FLEURS_MANIFEST}", flush=True)
            continue
        if p.exists():
            out.append((name, p, name))
        else:
            print(f"[skip] panel not present yet: {name} -> {p}", flush=True)
    return out


def main(argv=None):
    anonymize_process()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="append", help="training manifest (repeatable)")
    ap.add_argument("--procs", type=int, default=MAX_PROCS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pcm-hash", help="decoded-PCM SHA-256 (SPEC 1 step 2)")
    p.add_argument("--out", default=str(AUDIT / "pcm_sha256.tsv"))
    p.set_defaults(fn=cmd_pcm_hash)

    p = sub.add_parser("text", help="text leakage audit (SPEC 3 G0.3)")
    p.set_defaults(fn=cmd_text)

    p = sub.add_parser("fingerprint", help="audio fingerprint audit (SPEC 3 G0.4)")
    fs = p.add_subparsers(dest="fpcmd", required=True)
    q = fs.add_parser("calibrate")
    q.add_argument("--seed", type=int, default=20260905)   # SPEC 2/5/6 seed
    q.add_argument("--workdir", default=str(AUDIT / "cal"))
    q.add_argument("--neg-index-clips", type=int, default=20000,
                   help="NOT IN SPEC: size of the train-only index the negative "
                        "distribution is measured against; set it at or above "
                        "the number of protected-panel clips (17,210 today)")
    q.add_argument("--no-panels", action="store_true", default=True)
    q.set_defaults(fn=cmd_fp_calibrate)
    q = fs.add_parser("match")
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--emit-pcm-hash", action="store_true")
    q.add_argument("--pcm-tsv", default=str(AUDIT / "pcm_sha256.tsv"))
    q.set_defaults(fn=cmd_fp_match)

    p = sub.add_parser("report", help="evaluate the SPEC pass rule")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    _nice()
    args.fn(args)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
