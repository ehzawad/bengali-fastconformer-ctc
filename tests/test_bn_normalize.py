"""Tests that freeze the behaviour of bnasr/normalize.py.

These tests ARE the freeze.  If one fails, either the normalizer was edited
(make a ``normalize_bn_v2`` instead) or the Unicode database underneath moved.
Both are reasons to stop and re-derive the published numbers, not to update the
expected value.

Every invisible character is written as an escape on purpose: a reviewer must be
able to see that a U+0093 or a U+200C is present.  Where a Bengali literal would
be visually ambiguous (the pre-composed nukta letters U+09DC / U+09DD / U+09DF
render identically to their decomposed forms) it is spelled out in code points.

Nothing here needs a model, a GPU, or experiments/fastconformer_finetune/final.nemo.
The manifest-backed tests skip cleanly if data/test_manifest.json is absent.

Run:  .venv/bin/python -m pytest tests/test_bn_normalize.py -q
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bnasr.normalize import (  # noqa: E402
    DANDA,
    DOUBLE_DANDA,
    EmptyNormalizationError,
    NORMALIZER_VERSION,
    UNK_MARKER,
    ZWJ,
    ZWNJ,
    ZWSP,
    audit_residual,
    metric_surface,
    normalize_bn_v1,
    source_sha256,
)

TEST_MANIFEST = REPO_ROOT / "data" / "test_manifest.json"

KA = "ক"      # BENGALI LETTER KA
KHA = "খ"     # BENGALI LETTER KHA
NUKTA = "়"   # BENGALI SIGN NUKTA
LRM = "\u200e"     # LEFT-TO-RIGHT MARK
BOM = "\ufeff"     # ZERO WIDTH NO-BREAK SPACE
WORD_JOINER = "\u2060"
SOFT_HYPHEN = "\u00ad"
ABBREV_SIGN = "৽"  # BENGALI ABBREVIATION SIGN (Po in Unicode 15.0)

# --- rows copied verbatim out of data/test_manifest.json --------------------
# Kept as literals so the core tests still run in a bare clone with no data.
ROW_C1_LEADING = "\u0093" + "পেশাদার মানুষ হিসেবে"          # openslr_8b33332599.wav
ROW_C1_STANDALONE = "\u0094" + " বলছিলেন মিসেস মুখার্জী"      # openslr_2d1b3ef50b.wav
ROW_ZWNJ = "র" + ZWNJ + "্যাব ও বিজিবির সঙ্গে"                # openslr_c4d86f2b51.wav
ROW_ZWJ = "উল্লেখ করেন র" + ZWJ + "্যাব মহাপরিচালক"           # openslr_f658c699d1.wav
# openslr_58bcea8705.wav -- carries the pre-composed U+09DF; one of the 20
# non-NFC openslr_ rows.  Spelled out because it is visually identical to its
# canonical decomposition.
ROW_NON_NFC = "ম\u09dfমনসিংহ"
ROW_NON_NFC_CANONICAL = "ময়মনসিংহ"
# fleurs_test_000416.wav -- two dandas and a doubled space.
ROW_FLEURS_DANDA = (
    "বস্তুত এর অস্তিত্ব থাকলেও তা খুঁজে পাওয়া একেবারেই সহজ নয়" + DANDA + " "
    "একবার গুহার ভিতরে  এটা সম্পূর্ণ বিচ্ছিন্নতা" + DANDA
)

ALL_REAL_ROWS = [
    ROW_C1_LEADING,
    ROW_C1_STANDALONE,
    ROW_ZWNJ,
    ROW_ZWJ,
    ROW_NON_NFC,
    ROW_FLEURS_DANDA,
]


def _manifest_rows():
    if not TEST_MANIFEST.exists():
        pytest.skip(f"{TEST_MANIFEST} not present")
    with TEST_MANIFEST.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group(audio_filepath: str) -> str:
    name = os.path.basename(audio_filepath)
    for prefix in ("openslr_", "fleurs_test_", "fleurs_train_"):
        if name.startswith(prefix):
            return prefix
    return "OTHER"


# ---------------------------------------------------------------------------
# Idempotence:  f(f(x)) == f(x)
# ---------------------------------------------------------------------------

IDEMPOTENCE_CORPUS = ALL_REAL_ROWS + [
    "",
    " ",
    "   \t\n  ",
    KA,
    "  " + KA + "   " + KHA + "  ",
    "\u0093" + KA + "\u0094 " + KHA,
    KA + LRM + KHA,
    KA + ZWSP + KHA,
    KA + "\u00a0" + KHA,
    KA + BOM + KHA,
    KA + ZWNJ + KHA,
    KA + ZWJ + KHA,
    DANDA,
    DOUBLE_DANDA,
    KA + DANDA,
    KA + DANDA + DOUBLE_DANDA + KHA,
    KA + UNK_MARKER + KHA,
    KA + "-" + KHA,
    '"' + KA + ", " + KHA + DANDA + '"',
    "Dhaka BBC ঢাকা",
    KA + ABBREV_SIGN + KHA,
    "é",                      # base + combining acute: NFC must compose it
    "\u0093" + "́" + KA,       # combining mark orphaned by control removal
    "১০০ " + KA + " 100",
    "$ ~ ó ğ " + KA,      # the four OOV characters seen in real refs
]


@pytest.mark.parametrize("text", IDEMPOTENCE_CORPUS)
def test_normalize_bn_v1_is_idempotent(text):
    once = normalize_bn_v1(text)
    assert normalize_bn_v1(once) == once


@pytest.mark.parametrize("score_punctuation", [True, False])
@pytest.mark.parametrize("text", IDEMPOTENCE_CORPUS)
def test_metric_surface_is_idempotent(text, score_punctuation):
    once = metric_surface(text, score_punctuation)
    assert metric_surface(once, score_punctuation) == once


def test_idempotent_over_every_unicode_codepoint():
    """The actual proof, not a spot check.

    Exercises every assignable and unassigned code point in Unicode -- every Cc,
    Cf, Zs, P*, combining mark -- both alone and wedged between two Bengali
    consonants, which is where the post-removal NFC recomposition can bite.
    """
    failures = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:  # lone surrogates cannot come out of JSON
            continue
        char = chr(codepoint)
        for probe in (char, KA + char + KHA):
            once = normalize_bn_v1(probe)
            if normalize_bn_v1(once) != once:
                failures.append(("normalize_bn_v1", hex(codepoint), repr(probe)))
            for score_punctuation in (True, False):
                surface = metric_surface(probe, score_punctuation)
                if metric_surface(surface, score_punctuation) != surface:
                    failures.append(
                        ("metric_surface", score_punctuation, hex(codepoint), repr(probe))
                    )
        if failures:
            break
    assert failures == [], f"idempotence broken: {failures[:5]}"


def test_idempotent_over_every_test_manifest_row():
    rows = _manifest_rows()
    assert len(rows) == 3874, "test_manifest.json changed; re-verify every published count"
    for row in rows:
        text = row["text"]
        once = normalize_bn_v1(text)
        assert normalize_bn_v1(once) == once, row["audio_filepath"]
        for score_punctuation in (True, False):
            surface = metric_surface(text, score_punctuation)
            assert metric_surface(surface, score_punctuation) == surface, row["audio_filepath"]


# ---------------------------------------------------------------------------
# ZWJ / ZWNJ are contrastive in Bengali and must survive untouched
# ---------------------------------------------------------------------------


def test_zwnj_and_zwj_survive_normalization():
    assert normalize_bn_v1(ROW_ZWNJ) == ROW_ZWNJ
    assert normalize_bn_v1(ROW_ZWJ) == ROW_ZWJ
    assert ZWNJ in normalize_bn_v1(ROW_ZWNJ)
    assert ZWJ in normalize_bn_v1(ROW_ZWJ)


@pytest.mark.parametrize("score_punctuation", [True, False])
def test_zwnj_and_zwj_survive_both_metric_surfaces(score_punctuation):
    assert ZWNJ in metric_surface(ROW_ZWNJ, score_punctuation)
    assert ZWJ in metric_surface(ROW_ZWJ, score_punctuation)


def test_joiners_are_never_word_boundaries():
    """ra + ZWNJ + hasanta + ya must stay ONE word, not two."""
    assert len(normalize_bn_v1(ROW_ZWNJ).split()) == 4
    assert len(metric_surface(ROW_ZWNJ, False).split()) == 4
    assert len(normalize_bn_v1(KA + ZWNJ + KHA).split()) == 1
    assert len(normalize_bn_v1(KA + ZWJ + KHA).split()) == 1


def test_zwnj_and_zwj_stay_contrastive():
    """The two joiners must not collapse into each other, or into nothing."""
    with_zwnj = normalize_bn_v1("র" + ZWNJ + "্যাব")
    with_zwj = normalize_bn_v1("র" + ZWJ + "্যাব")
    without = normalize_bn_v1("র্যাব")
    assert len({with_zwnj, with_zwj, without}) == 3


def test_zwsp_is_a_word_boundary_but_the_joiners_are_not():
    """U+200B is a space by name and by function; U+200C/U+200D are joiners."""
    assert normalize_bn_v1(KA + ZWSP + KHA) == KA + " " + KHA
    assert normalize_bn_v1(KA + ZWNJ + KHA) == KA + ZWNJ + KHA
    assert normalize_bn_v1(KA + ZWJ + KHA) == KA + ZWJ + KHA


def test_audit_reports_joiners_separately_from_stray_format_characters():
    residual = audit_residual(normalize_bn_v1(KA + ZWNJ + ZWJ + LRM + KHA))
    assert residual["format_joiner"] == 2
    assert residual["format_other"] == 0


# ---------------------------------------------------------------------------
# C0 / C1 controls and U+200E
# ---------------------------------------------------------------------------


def test_c1_control_removed_from_real_openslr_row():
    """openslr_8b33332599.wav carries a raw U+0093 (mojibake left curly quote)."""
    assert unicodedata.category(ROW_C1_LEADING[0]) == "Cc"
    assert normalize_bn_v1(ROW_C1_LEADING) == "পেশাদার মানুষ হিসেবে"
    assert audit_residual(normalize_bn_v1(ROW_C1_LEADING))["control"] == 0


def test_c1_control_becomes_a_boundary_not_a_deletion():
    """openslr_2d1b3ef50b.wav starts with a standalone U+0094.

    Mapping the control to a space rather than deleting it drops the bogus token
    instead of welding it onto the next word: 4 raw whitespace tokens become the
    3 real words.  Deleting would be indistinguishable here, but not in general.
    """
    assert normalize_bn_v1(ROW_C1_STANDALONE) == "বলছিলেন মিসেস মুখার্জী"
    assert len(ROW_C1_STANDALONE.split()) == 4
    assert len(normalize_bn_v1(ROW_C1_STANDALONE).split()) == 3
    assert normalize_bn_v1(KA + "\u0093" + KHA) == KA + " " + KHA


@pytest.mark.parametrize(
    "codepoint",
    [0x00, 0x01, 0x07, 0x08, 0x0B, 0x0C, 0x0E, 0x1F, 0x7F, 0x80, 0x85, 0x93, 0x94, 0x9F],
)
def test_every_c0_and_c1_control_is_stripped(codepoint):
    char = chr(codepoint)
    assert unicodedata.category(char) == "Cc"
    normalized = normalize_bn_v1(KA + char + KHA)
    assert normalized == KA + " " + KHA
    assert audit_residual(normalized)["control"] == 0


def test_tab_newline_and_carriage_return_collapse_to_one_space():
    assert normalize_bn_v1(KA + "\t\r\n " + KHA) == KA + " " + KHA


def test_u200e_and_the_other_bidi_format_characters_are_deleted():
    """U+200E is called out in the brief; it must vanish without splitting a word."""
    assert normalize_bn_v1(KA + LRM + KHA) == KA + KHA
    for bidi in ("\u200e", "\u200f", "\u061c", "\u202a", "\u202e", "\u2066", "\u2069"):
        assert unicodedata.category(bidi) == "Cf"
        assert normalize_bn_v1(KA + bidi + KHA) == KA + KHA
    assert audit_residual(normalize_bn_v1(KA + LRM + KHA))["format_other"] == 0


def test_no_break_format_characters_are_deleted_not_spaced():
    """U+FEFF and U+2060 are explicitly 'no-break': they must not split a word."""
    assert normalize_bn_v1(KA + BOM + KHA) == KA + KHA
    assert normalize_bn_v1(KA + WORD_JOINER + KHA) == KA + KHA
    assert normalize_bn_v1(KA + SOFT_HYPHEN + KHA) == KA + KHA


def test_whitespace_is_collapsed_and_stripped():
    assert normalize_bn_v1("   " + KA + "    " + KHA + "   ") == KA + " " + KHA
    for space in ("\u00a0", "\u2009", "\u3000", "\u2028", "\u202f", "\u1680"):
        assert normalize_bn_v1(KA + space + KHA) == KA + " " + KHA


# ---------------------------------------------------------------------------
# Danda, double danda, and the punctuation switch
# ---------------------------------------------------------------------------


def test_danda_and_double_danda_are_punctuation_in_this_unicode_build():
    for char in (DANDA, DOUBLE_DANDA, ABBREV_SIGN):
        assert unicodedata.category(char).startswith("P"), (
            f"U+{ord(char):04X} is no longer P* in unicodedata "
            f"{unicodedata.unidata_version}; the frozen punctuation surface moved"
        )


def test_danda_is_kept_when_punctuation_is_scored():
    text = KA + " " + KHA + DANDA
    assert metric_surface(text, True) == KA + " " + KHA + DANDA
    assert metric_surface(text, True).split() == [KA, KHA + DANDA]


def test_danda_becomes_a_space_when_punctuation_is_not_scored():
    text = KA + " " + KHA + DANDA
    assert metric_surface(text, False) == KA + " " + KHA
    assert metric_surface(text, False).split() == [KA, KHA]


def test_double_danda_handled_in_both_modes():
    text = KA + DOUBLE_DANDA + KHA
    assert metric_surface(text, True) == KA + DOUBLE_DANDA + KHA
    assert metric_surface(text, True).split() == [KA + DOUBLE_DANDA + KHA]
    assert metric_surface(text, False) == KA + " " + KHA
    assert metric_surface(text, False).split() == [KA, KHA]


def test_the_punctuation_mode_changes_the_word_denominator():
    """Mapping punctuation to a space splits tokens; the scorer must expect it.

    Measured on the clean 920: 17,590 reference words punctuation-sensitive,
    17,764 punctuation-insensitive (+0.99%).  The two numbers are not
    interchangeable and must never be mixed in one table.
    """
    text = KA + DANDA + KHA
    assert len(metric_surface(text, True).split()) == 1
    assert len(metric_surface(text, False).split()) == 2


def test_punctuation_insensitive_mode_removes_every_p_star_class():
    for char in ('"', "'", ",", ".", ":", ";", "-", "–", "‘", "’",
                 "(", ")", "[", "]", "¿", ABBREV_SIGN):
        assert unicodedata.category(char).startswith("P")
        assert metric_surface(KA + char + KHA, False) == KA + " " + KHA
        assert metric_surface(KA + char + KHA, True) == KA + char + KHA


def test_symbols_are_not_punctuation_and_survive_both_modes():
    """U+0024, U+007E, U+00B0, U+002B are S*, not P*: they stay, and stay errors."""
    for char in ("$", "~", "°", "+"):
        assert not unicodedata.category(char).startswith("P")
        assert char in metric_surface(KA + char + KHA, False)
        assert char in metric_surface(KA + char + KHA, True)


def test_unk_marker_is_exempt_from_the_punctuation_mapping():
    """U+2047 is Po, but it is the decoder's OOV marker, not label typography.

    Mapping it to a space would delete the evidence of a decoding failure and
    flatter the reported WER, so it is exempt in BOTH modes.
    """
    assert unicodedata.category(UNK_MARKER) == "Po"
    assert metric_surface(KA + UNK_MARKER + KHA, False) == KA + UNK_MARKER + KHA
    assert metric_surface(KA + UNK_MARKER + KHA, True) == KA + UNK_MARKER + KHA
    assert audit_residual(metric_surface(KA + UNK_MARKER, False))["unk_marker"] == 1


def test_punctuation_only_input_may_normalize_to_empty_without_raising():
    """A danda-only reference is a legitimate zero-word row, not a bug."""
    assert metric_surface(DANDA, False) == ""
    assert metric_surface(DANDA + " " + DOUBLE_DANDA, False) == ""
    assert metric_surface(DANDA, True) == DANDA


# ---------------------------------------------------------------------------
# NFC, and the measured case against NFKC
# ---------------------------------------------------------------------------


def test_nfc_decomposes_the_precomposed_nukta_letters():
    """U+09DF/U+09DC/U+09DD are composition exclusions: NFC splits them.

    The vocab-256 tokenizer contains U+09BC but not U+09DF, so without this the
    20 pre-composed openslr_ test rows could never be matched by the model.
    """
    assert ROW_NON_NFC != ROW_NON_NFC_CANONICAL
    assert normalize_bn_v1(ROW_NON_NFC) == ROW_NON_NFC_CANONICAL
    assert normalize_bn_v1(ROW_NON_NFC_CANONICAL) == ROW_NON_NFC_CANONICAL
    for composed, base in (("\u09df", "য"), ("\u09dc", "ড"), ("\u09dd", "ঢ")):
        assert normalize_bn_v1(composed) == base + NUKTA


def test_nfkc_would_destroy_the_unk_marker_but_nfc_does_not():
    """The measured reason NFC was chosen over NFKC."""
    assert unicodedata.normalize("NFKC", UNK_MARKER) == "??"
    assert unicodedata.normalize("NFC", UNK_MARKER) == UNK_MARKER
    assert UNK_MARKER in normalize_bn_v1(KA + UNK_MARKER)


def test_nfkc_and_nfc_agree_on_every_test_manifest_reference():
    """The other half of the justification: on scored rows the choice is free."""
    rows = _manifest_rows()
    differing = [
        row["audio_filepath"]
        for row in rows
        if unicodedata.normalize("NFKC", row["text"]) != unicodedata.normalize("NFC", row["text"])
    ]
    assert differing == []


def test_nfkc_would_not_have_unified_bengali_and_ascii_digits_either():
    assert unicodedata.normalize("NFKC", "১") == "১"
    assert normalize_bn_v1("১") == "১"
    assert normalize_bn_v1("1") == "1"


def test_nfc_runs_after_control_and_format_removal():
    """Deleting an interposed format character must still let NFC compose."""
    assert normalize_bn_v1("য" + LRM + NUKTA) == "য" + NUKTA
    assert normalize_bn_v1("e" + LRM + "́") == "é"


# ---------------------------------------------------------------------------
# Case folding
# ---------------------------------------------------------------------------


def test_case_is_folded_identically_on_both_sides():
    assert normalize_bn_v1("Dhaka BBC") == "dhaka bbc"
    assert normalize_bn_v1("ঢাকা BBC") == "ঢাকা bbc"


def test_case_folding_does_not_touch_bengali():
    text = "বাংলাদেশ পরিসংখ্যান ব্যুরো"
    assert normalize_bn_v1(text) == text


def test_case_folding_changes_no_test_manifest_reference():
    """Measured: not one Lu character in the whole 3,874-row test manifest."""
    rows = _manifest_rows()
    changed = [row["audio_filepath"] for row in rows if row["text"].casefold() != row["text"]]
    assert changed == []


# ---------------------------------------------------------------------------
# The empty-output guard
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_without_raising():
    assert normalize_bn_v1("") == ""
    assert normalize_bn_v1("    ") == ""
    assert normalize_bn_v1(LRM) == ""
    assert metric_surface("", True) == ""
    assert metric_surface("", False) == ""


def test_never_returns_empty_for_bengali_input():
    for text in ALL_REAL_ROWS:
        assert normalize_bn_v1(text) != ""
        assert metric_surface(text, True) != ""
        assert metric_surface(text, False) != ""


def test_no_test_manifest_reference_normalizes_to_empty():
    rows = _manifest_rows()
    empty = [row["audio_filepath"] for row in rows if not metric_surface(row["text"], False)]
    assert empty == []


def test_empty_normalization_error_is_a_value_error():
    assert issubclass(EmptyNormalizationError, ValueError)


def test_guard_fires_loudly_when_a_letter_would_be_erased(monkeypatch):
    """Force the failure mode the guard exists for; it must raise, not return ''."""
    import bnasr.normalize as module

    monkeypatch.setattr(module, "_char_action", lambda ch: module._DROP)
    with pytest.raises(EmptyNormalizationError) as excinfo:
        module.normalize_bn_v1("ঢাকা")
    message = str(excinfo.value)
    assert NORMALIZER_VERSION in message
    assert "U+09A2" in message  # code points are listed; the text itself is unreadable


def test_guard_also_covers_the_punctuation_surface(monkeypatch):
    import bnasr.normalize as module

    monkeypatch.setattr(module, "_is_scoring_punct", lambda ch: True)
    with pytest.raises(EmptyNormalizationError):
        module.metric_surface("ঢাকা", False)


def test_non_string_and_non_bool_inputs_are_rejected():
    with pytest.raises(TypeError):
        normalize_bn_v1(None)
    with pytest.raises(TypeError):
        normalize_bn_v1(b"\xe0\xa6\xa2")
    with pytest.raises(TypeError):
        metric_surface(KA, "false")  # the flag must be a real bool, not a truthy str
    with pytest.raises(TypeError):
        metric_surface(KA, 1)


# ---------------------------------------------------------------------------
# Real manifest rows, end to end
# ---------------------------------------------------------------------------


def test_real_fleurs_row_in_both_modes():
    """fleurs_test_000416.wav: two dandas plus a doubled space."""
    sensitive = metric_surface(ROW_FLEURS_DANDA, True)
    insensitive = metric_surface(ROW_FLEURS_DANDA, False)

    assert "  " not in sensitive
    assert "  " not in insensitive
    assert sensitive.count(DANDA) == 2
    assert DANDA not in insensitive
    # Both dandas sit at a word end here, so the row keeps 16 words either way.
    assert len(ROW_FLEURS_DANDA.split()) == 16
    assert len(sensitive.split()) == 16
    assert len(insensitive.split()) == 16

    word = unicodedata.normalize("NFC", ROW_FLEURS_DANDA.split()[9])
    assert word.endswith(DANDA)
    assert sensitive.split()[9] == word
    assert insensitive.split()[9] == word[:-1]


def test_real_manifest_rows_match_the_hardcoded_literals():
    """Guards against the literals above drifting away from the actual data."""
    rows = {os.path.basename(r["audio_filepath"]): r["text"] for r in _manifest_rows()}
    assert rows["openslr_8b33332599.wav"] == ROW_C1_LEADING
    assert rows["openslr_2d1b3ef50b.wav"] == ROW_C1_STANDALONE
    assert rows["openslr_c4d86f2b51.wav"] == ROW_ZWNJ
    assert rows["openslr_f658c699d1.wav"] == ROW_ZWJ
    assert rows["openslr_58bcea8705.wav"] == ROW_NON_NFC
    assert rows["fleurs_test_000416.wav"] == ROW_FLEURS_DANDA


def test_group_row_counts_and_change_counts_are_frozen():
    """The published numbers depend on these exact counts."""
    rows = _manifest_rows()
    counts = dict.fromkeys(("openslr_", "fleurs_test_", "fleurs_train_", "OTHER"), 0)
    changed = dict.fromkeys(("openslr_", "fleurs_test_", "fleurs_train_", "OTHER"), 0)
    for row in rows:
        group = _group(row["audio_filepath"])
        counts[group] += 1
        if normalize_bn_v1(row["text"]) != row["text"]:
            changed[group] += 1

    assert counts == {"openslr_": 2908, "fleurs_test_": 920, "fleurs_train_": 46, "OTHER": 0}
    assert changed == {"openslr_": 23, "fleurs_test_": 26, "fleurs_train_": 2, "OTHER": 0}


def test_bare_fleurs_prefix_is_the_wrong_clean_predicate():
    """966 basenames start with 'fleurs'; only 920 are the clean official test set."""
    rows = _manifest_rows()
    bare = sum(1 for r in rows if os.path.basename(r["audio_filepath"]).startswith("fleurs"))
    exact = sum(1 for r in rows if os.path.basename(r["audio_filepath"]).startswith("fleurs_test_"))
    assert bare == 966
    assert exact == 920


def test_reference_word_counts_per_group_are_frozen():
    """WER is word-weighted; these are the denominators of every published number."""
    rows = _manifest_rows()
    words = dict.fromkeys(("openslr_", "fleurs_test_", "fleurs_train_"), 0)
    words_insensitive = dict(words)
    for row in rows:
        group = _group(row["audio_filepath"])
        words[group] += len(metric_surface(row["text"], True).split())
        words_insensitive[group] += len(metric_surface(row["text"], False).split())

    assert words == {"openslr_": 9237, "fleurs_test_": 17590, "fleurs_train_": 877}
    assert words_insensitive == {"openslr_": 9330, "fleurs_test_": 17764, "fleurs_train_": 881}
    # The clean 920 carry 63.5% of the punctuation-sensitive word denominator on
    # 23.7% of the utterances -- which is why the groups must never be pooled.
    assert 0.63 < words["fleurs_test_"] / sum(words.values()) < 0.64


def test_no_residual_control_or_format_characters_after_the_scoring_surface():
    rows = _manifest_rows()
    for row in rows:
        for score_punctuation in (True, False):
            residual = audit_residual(metric_surface(row["text"], score_punctuation))
            assert residual["control"] == 0, row["audio_filepath"]
            assert residual["format_other"] == 0, row["audio_filepath"]
            assert residual["surrogate"] == 0, row["audio_filepath"]
            assert residual["private_use"] == 0, row["audio_filepath"]


def test_the_joiners_are_the_only_format_characters_that_survive():
    """1 ZWNJ + 1 ZWJ in openslr_, 2 ZWNJ + 3 ZWJ in fleurs_test_, 0 in fleurs_train_."""
    rows = _manifest_rows()
    per_group = dict.fromkeys(("openslr_", "fleurs_test_", "fleurs_train_"), 0)
    for row in rows:
        per_group[_group(row["audio_filepath"])] += audit_residual(
            metric_surface(row["text"], False)
        )["format_joiner"]
    assert per_group == {"openslr_": 2, "fleurs_test_": 5, "fleurs_train_": 0}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_source_sha256_is_stable_and_hex():
    digest = source_sha256()
    assert len(digest) == 64
    assert digest == source_sha256()
    int(digest, 16)


def test_normalizer_version_is_v1():
    assert NORMALIZER_VERSION == "bn_normalize_v1"
