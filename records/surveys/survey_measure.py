#!/usr/bin/env python
"""Reproduce every measured number in `conversational_survey_2026-09-07.md`.

WHY THIS EXISTS AT ALL
----------------------
Council G Q5 (SPEC §8m) asked for a *conversational-data survey*, not another arm.
A survey whose numbers come from dataset cards and paper abstracts is worth very
little to a phase whose whole discipline is "a number that is not here is not
registered" (SPEC preamble): abstracts round, cards go stale, and the two figures
that actually decide admission -- how many hours survive our registered cleaning,
and what fraction of rows our registered gates delete -- are never printed in a
paper. So every quantitative claim in the report is recomputed here, from text and
from HTTP metadata only, and nothing here downloads or decodes a single second of
audio.

WHY IT IS TEXT-ONLY AND CPU-ONLY
--------------------------------
SPEC §8m allocates 0 further Phase-4 GPU-h to expansion reconnaissance, and both
cards are occupied by the O+madasr and O+vaani arms plus a co-tenant. The Ben-10
duration distribution is therefore derived from LFS blob sizes plus one 96-byte
HTTP range read of a WAV header (which pins the PCM geometry, so bytes -> seconds
is exact arithmetic rather than a guess), and the SPRING-INX script statistics come
from the Hugging Face datasets-server `/rows` endpoint, which returns transcripts
without the audio payload.

WHY THE FROZEN SURFACE, NOT A LOCAL RE-IMPLEMENTATION
----------------------------------------------------
Script purity is only meaningful if it is measured through the *same* transform the
trainer would apply. SPEC §1 fixes that transform as
`metric_surface(NFC(text), score_punctuation=False)`, and G0.9(a) counts violations
on the surfaced string -- so a row whose only Latin characters sit inside
punctuation that the surface deletes is NOT a violation. We import the repo's
`src.bn_normalize.metric_surface` rather than re-deriving it, per the reuse rule.

Run:  .venv/bin/python experiments/p4/surveys/survey_measure.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.bn_normalize import metric_surface  # noqa: E402  (path shim must precede)

# --- Registered constants, each asserted against the SPEC clause it comes from ---

# SPEC §1, "Cleaning for new rows (registered limits)": duration band 0.5-23.0 s.
# A candidate's share of rows outside this band is a *deletion* forecast, which is
# what decides whether a bucket can still clear G0.8's hours floor after cleaning.
DURATION_MIN_S = 0.5
DURATION_MAX_S = 23.0
assert (DURATION_MIN_S, DURATION_MAX_S) == (0.5, 23.0), "SPEC §1 duration band"

# SPEC §3 G0.9(a): >1 % script violations on the INCOMING candidate rows fails the
# bucket outright (council D Q4 ruling); the >5 % clause only adds an owner report,
# it never rescues a bucket. Both thresholds are carried so the report can say which
# side of each line a candidate falls on.
SCRIPT_VIOLATION_FAIL = 0.01
SCRIPT_VIOLATION_REVIEW = 0.05
assert SCRIPT_VIOLATION_FAIL == 0.01, "SPEC §3 G0.9(a) ruling (council D Q4)"

# SPEC §3 G0.8: >=500 non-duplicate NEW hours after all gates, else STOP Phase 4.
# Also the floor SPEC §8k Q3 reuses to decide which single arm may serve as the
# union fallback. Any candidate below this cannot carry the phase alone.
MIN_NEW_HOURS = 500.0
assert MIN_NEW_HOURS == 500.0, "SPEC §3 G0.8 / §8k Q3 minimum-hours rule"

# Ben-10 audio geometry, pinned by reading the 96-byte RIFF header of
# train/folder_1/train_barishal (1).wav and valid/valid_barishal (1).wav:
# audio_format 1 (PCM), channels 1, rate 16000, bits 16 -> 32000 bytes of PCM per
# second, plus a 44-byte canonical header. This is what makes the blob-size ->
# duration conversion exact instead of an assumption.
BEN10_BYTES_PER_SEC = 16000 * 1 * 2
BEN10_HEADER_BYTES = 44
assert BEN10_BYTES_PER_SEC == 32000, "measured Ben-10 WAV header, 2026-09-07"


def _surface(text: str) -> str:
    """Apply the repo's frozen target surface exactly as SPEC §1 registers it."""
    return metric_surface(unicodedata.normalize("NFC", text or ""), score_punctuation=False)


def ben10_text_stats(csv_paths: list[Path]) -> dict:
    """Script purity and annotation-marker load for Ben-10, per G0.9(a)/(e).

    The interesting quantity is NOT how many rows contain a Latin letter in the raw
    CSV -- it is how many still do after the frozen surface, because that is the
    string the tokenizer and the WER scorer would ever see. Ben-10 writes English
    loanwords in Bengali script ("ডিপার্টমেন্ট অব সফটওয়্যার"), so the raw and
    surfaced counts diverge sharply from a corpus like SPRING-INX that keeps Latin.

    The `<>` count is tracked separately because SPEC §1's Vaani annotation policy
    DROPS rows carrying an unintelligible marker (partial labels teach deletion --
    the Phase-1 lesson). If `<>` is that marker in Ben-10, this count is the bucket's
    real yield loss, and it dwarfs every other cleaning rule.
    """
    out = {}
    for path in csv_paths:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        n = len(rows)
        latin = angle = empty = 0
        residual: dict[str, int] = {}
        for r in rows:
            raw = r["transcripts"] or ""
            if not raw.strip():
                empty += 1
            if "<" in raw or ">" in raw:
                angle += 1
            s = _surface(raw)
            if re.search(r"[A-Za-z]", s):
                latin += 1
            for ch in s:
                if ch == " " or "ঀ" <= ch <= "৿" or ch.isdigit():
                    continue
                residual[ch] = residual.get(ch, 0) + 1
        out[path.stem] = {
            "rows": n,
            "latin_rows_after_surface": latin,
            "latin_rate": latin / n,
            "angle_marker_rows": angle,
            "angle_marker_rate": angle / n,
            "empty_rows": empty,
            "residual_non_bengali_chars": sorted(residual.items(), key=lambda kv: -kv[1])[:10],
            "g0_9a_verdict": "PASS" if latin / n <= SCRIPT_VIOLATION_FAIL else "FAIL",
        }
    return out


def ben10_duration_stats(tree_json: Path) -> dict:
    """Duration distribution from LFS blob sizes -- no audio fetched, no decode.

    Each Ben-10 entry is canonical 16 kHz/16-bit/mono PCM WAV (header measured, see
    BEN10_BYTES_PER_SEC), so (blob_size - 44) / 32000 is the exact duration. This
    matters because the paper reports a 16.60 s mean clip, i.e. an order of magnitude
    longer than any bucket currently in the corpus, and SPEC §1 caps admitted rows at
    23.0 s. Without this measurement the survey could not say whether the cap costs
    Ben-10 1 % of its rows or 30 %.

    `tree_json` holds the raw pages of GET /api/datasets/<id>/tree/main/<path>.
    """
    sizes: dict[str, int] = {}
    for line in tree_json.read_text().splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        for e in json.loads(line):
            if e.get("type") == "file" and e["path"].endswith(".wav"):
                sizes[e["path"]] = (e.get("lfs") or {}).get("size") or e.get("size")
    durations = sorted((s - BEN10_HEADER_BYTES) / BEN10_BYTES_PER_SEC for s in sizes.values())
    n = len(durations)
    over = sum(1 for d in durations if d > DURATION_MAX_S)
    under = sum(1 for d in durations if d < DURATION_MIN_S)
    return {
        "wav_files": n,
        "total_hours": sum(durations) / 3600.0,
        "mean_s": statistics.fmean(durations),
        "median_s": statistics.median(durations),
        "p95_s": durations[int(0.95 * n)],
        "max_s": durations[-1],
        "rows_over_23s": over,
        "rate_over_23s": over / n,
        "rows_under_0p5s": under,
        # Hours that survive the band; the band is applied row-wise, and long rows
        # carry disproportionate duration, so this is well below total * (1 - rate).
        "hours_within_band": sum(d for d in durations if DURATION_MIN_S <= d <= DURATION_MAX_S) / 3600.0,
    }


def spring_script_stats(row_jsons: list[Path]) -> dict:
    """SPRING-INX Latin-script load, measured per ROW and per CHARACTER.

    SPEC §3 G0.9(a) recorded 48.3 % of SPRING R1 rows as Latin-bearing and excluded
    the bucket. The survey question is whether a Bengali-script-ONLY subset is worth
    an owner amendment, and that needs an HOURS estimate, not a row estimate. Rows
    and hours are not interchangeable here: a code-switched utterance is a longer
    utterance (English is reached for mid-sentence in longer turns), so the retained
    *hours* share is materially smaller than the retained *rows* share. Surfaced
    character count is the cheapest duration proxy available without touching audio,
    and reporting both makes that bias visible instead of hiding it inside one number.
    """
    per_ds: dict[str, dict] = {}
    for path in row_jsons:
        ds = "R1" if "_R1_" in path.name else "R2"
        d = json.loads(path.read_text())
        acc = per_ds.setdefault(ds, {"clean_rows": 0, "latin_rows": 0, "clean_chars": 0, "latin_chars": 0})
        for r in d.get("rows", []):
            s = _surface(r["row"].get("text", ""))
            if re.search(r"[A-Za-z]", s):
                acc["latin_rows"] += 1
                acc["latin_chars"] += len(s)
            else:
                acc["clean_rows"] += 1
                acc["clean_chars"] += len(s)
    for ds, a in per_ds.items():
        rows = a["clean_rows"] + a["latin_rows"]
        chars = a["clean_chars"] + a["latin_chars"]
        a["sampled_rows"] = rows
        a["latin_rate_rows"] = a["latin_rows"] / rows
        a["retained_share_rows"] = a["clean_rows"] / rows
        a["retained_share_chars"] = a["clean_chars"] / chars
    return per_ds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ben10-csv", nargs="*", type=Path, default=[])
    ap.add_argument("--ben10-tree", type=Path)
    ap.add_argument("--spring-rows", nargs="*", type=Path, default=[])
    a = ap.parse_args()
    report: dict = {}
    if a.ben10_csv:
        report["ben10_text"] = ben10_text_stats(a.ben10_csv)
    if a.ben10_tree:
        report["ben10_duration"] = ben10_duration_stats(a.ben10_tree)
    if a.spring_rows:
        report["spring_script"] = spring_script_stats(a.spring_rows)
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
