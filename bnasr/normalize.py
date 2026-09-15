"""Frozen, versioned Bengali text normalization for ASR scoring.

This module is the ONE text surface used by the scorer.  The identical function
must be applied to references and hypotheses; anything else reports label
typography as acoustics.

FREEZE CONTRACT
---------------
``normalize_bn_v1`` and ``metric_surface`` are frozen.  Once a WER/CER number
computed with them has been published, their behaviour must never change.  A
behaviour change means a NEW function (``normalize_bn_v2``) alongside this one,
never an edit in place.  Frozen against Python 3.12.13 / unicodedata 15.0.0;
``UNICODE_VERSION`` records what the running interpreter actually has, and
``tests/test_bn_normalize.py`` asserts the classifications this module depends
on so a Unicode upgrade fails loudly instead of silently moving the numbers.

WHY NFC AND NOT NFKC  (measured on this repo's manifests, 2026-08-22)
---------------------------------------------------------------------
1. On every row that is actually scored the two forms are identical, so the
   choice is free on the reference side:

       NFKC(t) != NFC(t) on   0 / 3,874 test rows
                              (0/2,908 openslr_, 0/920 fleurs_test_, 0/46 fleurs_train_)
                              0 / 1,488 dev rows
                             18 / 217,669 train rows

2. The tiebreaker is therefore the HYPOTHESIS side, and there NFKC is actively
   harmful.  SentencePiece decodes an out-of-vocabulary piece to U+2047 DOUBLE
   QUESTION MARK.  NFKC rewrites U+2047 into "??" (two Po code points), which a
   punctuation-insensitive surface then maps to whitespace -- silently deleting
   the model's own failure marker and flattering WER.  NFC leaves U+2047 intact,
   and this module additionally exempts it from punctuation mapping (see
   ``PUNCT_EXEMPT``).

3. NFKC buys nothing else here: it does NOT unify Bengali digits (U+09E6-U+09EF)
   with ASCII digits, and the compatibility spaces it would fold (NBSP, thin,
   ideographic) are already folded explicitly below.

4. NFC is not a no-op and is not optional.  It rewrites the pre-composed nukta
   letters U+09DF / U+09DC / U+09DD into base + U+09BC (they are Unicode
   composition exclusions, so NFC decomposes rather than composes them) on
   20 / 2,908 openslr_ test rows and 994 / 214,333 train rows.  The vocab-256
   SentencePiece model contains U+09BC but NOT U+09DF, so the model can only
   ever emit the decomposed form; without NFC those references are unmatchable
   by construction.  That is the measured reason NFC is applied at all.

CHARACTER POLICY
----------------
* Cc (C0 and C1 controls, incl. the U+0093 / U+0094 mojibake curly quotes that
  appear in 3 openslr_ test rows) -> single space.  Space, not deletion: these
  stand in for punctuation and must not silently weld two words together.  On
  the real data the two choices are identical (all 3 occurrences sit at a word
  edge), so this is a general-safety choice, not a scoring choice.
* Cf format characters -> DELETED, except:
    - U+200C ZWNJ and U+200D ZWJ are PRESERVED.  They are contrastive in Bengali
      conjuncts (ra + ZWNJ + hasanta + ya vs. ra + hasanta + ya) and appear in
      real references in every population.  They are never treated as
      whitespace and never stripped.  NOTE: U+200C is not in the vocab-256
      tokenizer, so a reference containing it costs at least one word error that
      the model cannot avoid.  That is honest, not a bug to normalize away.
    - U+200B ZWSP -> single space.  It is a space by name and by function and is
      a legitimate word boundary.
  U+200E LRM (required by the brief; 2 occurrences in train) and U+FEFF, U+00AD,
  bidi embeddings/isolates and every other Cf therefore disappear.  U+FEFF and
  U+2060 are deliberately DELETED rather than spaced: they are explicitly
  "no-break", so they must not create a word boundary.
* Zs / Zl / Zp and the C0 whitespace characters collapse via str.split(), which
  is Unicode-aware (NBSP, U+2028, U+2000-U+200A all split).
* Case folding: str.casefold() then a final NFC.  Measured effect on the
  references: 0 / 3,874 test rows, 1 / 1,488 dev rows, 15 / 217,669 train rows
  (there is not a single Lu character in the entire test manifest), so it cannot
  flatter the published reference.  It is kept because the vocab-256 tokenizer
  DOES contain 8 uppercase Latin pieces (B D F H K L O T), so the hypothesis
  side can produce case the reference does not have.
* Surrogates (Cs) and private-use (Co) are left untouched on purpose, so that
  ``audit_residual`` can surface them instead of this module hiding corruption.
  Measured: 0 occurrences in any manifest.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No number expansion, no Bengali/ASCII digit unification (FLEURS writes 18.3%
of its numerals in ASCII, training data 93% in Bengali -- that is a genuine
transcription difference, to be reported, not folded away), no transliteration,
no spelling equivalence, no symbol stripping (U+0024 "$", U+007E "~", U+00B0 are
S* not P* and survive both surfaces).

CER NOTE
--------
Any CER computed on this output is over Python code points (Unicode scalar
values), not extended grapheme clusters.  Bengali vowel signs, hasanta, nukta,
ZWNJ and ZWJ each count as one code point.  Say so in the model card; it is not
comparable to a grapheme-cluster metric.
"""

from __future__ import annotations

import hashlib
import unicodedata
from functools import lru_cache
from pathlib import Path

__all__ = [
    "NORMALIZER_VERSION",
    "UNICODE_VERSION",
    "EmptyNormalizationError",
    "normalize_bn_v1",
    "metric_surface",
    "audit_residual",
    "source_sha256",
    "ZWNJ",
    "ZWJ",
    "ZWSP",
    "UNK_MARKER",
    "DANDA",
    "DOUBLE_DANDA",
    "JOINERS_PRESERVED",
    "PUNCT_EXEMPT",
]

NORMALIZER_VERSION = "bn_normalize_v1"
FROZEN_AGAINST = {"python": "3.12.13", "unicodedata": "15.0.0"}
UNICODE_VERSION = unicodedata.unidata_version

ZWNJ = "\u200c"          # ZERO WIDTH NON-JOINER  -- preserved, contrastive
ZWJ = "\u200d"           # ZERO WIDTH JOINER      -- preserved, contrastive
ZWSP = "\u200b"          # ZERO WIDTH SPACE       -- becomes a word boundary
LRM = "\u200e"           # LEFT-TO-RIGHT MARK     -- deleted (Cf)
UNK_MARKER = "⁇"    # DOUBLE QUESTION MARK: SentencePiece <unk> surface
DANDA = "।"
DOUBLE_DANDA = "॥"
BENGALI_ABBREVIATION_SIGN = "৽"  # Po in Unicode 15.0 -> scored as punctuation

#: Format characters that survive normalization. Nothing else in Cf does.
JOINERS_PRESERVED = frozenset({ZWNJ, ZWJ})

#: P* characters that ``metric_surface`` refuses to treat as punctuation.
#: U+2047 is the model's out-of-vocabulary marker: mapping it to a space would
#: delete the evidence of a decoding failure and flatter the reported WER.
PUNCT_EXEMPT = frozenset({UNK_MARKER})

_KEEP = 0
_SPACE = 1
_DROP = 2


class EmptyNormalizationError(ValueError):
    """Raised when normalization would erase text that still had real content.

    "Real content" means at least one Unicode letter (L*) or number (N*).  A
    punctuation-only reference (a bare danda, say) legitimately normalizes to
    the empty string under the punctuation-insensitive surface and does NOT
    raise -- the scorer is expected to keep that row, count zero reference
    words, and report how many such rows exist.
    """


@lru_cache(maxsize=None)
def _char_action(ch: str) -> int:
    """Classify one character: keep it, replace it with a space, or delete it.

    Cached per code point; the first call for a character is the only one that
    touches the Unicode database.
    """
    if ch in JOINERS_PRESERVED:
        return _KEEP
    if ch == ZWSP:
        return _SPACE
    category = unicodedata.category(ch)
    if category == "Cc":
        # C0 + C1, including U+0093/U+0094 and \t \n \r \v \f.
        return _SPACE
    if category == "Cf":
        # U+200E LRM, U+200F, U+061C, bidi embeddings/isolates, U+00AD,
        # U+FEFF, U+2060, tag characters, ...
        return _DROP
    return _KEEP


@lru_cache(maxsize=None)
def _is_scoring_punct(ch: str) -> bool:
    """True if ``metric_surface`` should turn this character into a space."""
    if ch in PUNCT_EXEMPT:
        return False
    return unicodedata.category(ch).startswith("P")


def _has_scoreable_content(text: str) -> bool:
    """True if ``text`` contains at least one Unicode letter or number."""
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


def _describe(text: str, limit: int = 24) -> str:
    """Code-point listing for error messages (the inputs are invisible)."""
    points = " ".join(f"U+{ord(ch):04X}" for ch in text[:limit])
    if len(text) > limit:
        points += f" ... (+{len(text) - limit} more)"
    return points


def normalize_bn_v1(text: str) -> str:
    """Frozen Bengali scoring normalization. Apply to references AND hypotheses.

    Pipeline, in order:
      1. Cc -> space, Cf -> deleted (ZWNJ/ZWJ kept, ZWSP -> space).
      2. casefold().
      3. NFC.  Applied after step 1 so that removing an interposed format
         character lets a base and its combining mark compose, and after step 2
         because casefold() does not preserve NFC for every character.
      4. Whitespace collapsed to single U+0020, string stripped.

    Idempotent: ``normalize_bn_v1(normalize_bn_v1(x)) == normalize_bn_v1(x)``.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        EmptyNormalizationError: if the input contained a letter or a number and
            the output is empty. That can only happen if this function is
            broken, so it is a hard failure rather than a silently dropped row.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"{NORMALIZER_VERSION}: expected str, got {type(text).__name__!r}. "
            "Reference and hypothesis must both be decoded text."
        )

    out = []
    for ch in text:
        action = _char_action(ch)
        if action == _KEEP:
            out.append(ch)
        elif action == _SPACE:
            out.append(" ")
        # _DROP: append nothing.
    stripped = "".join(out)

    folded = unicodedata.normalize("NFC", stripped.casefold())
    result = " ".join(folded.split())

    if not result and _has_scoreable_content(text):
        raise EmptyNormalizationError(
            f"{NORMALIZER_VERSION} erased a non-empty input that contained "
            f"letters or digits: {text!r} [{_describe(text)}]"
        )
    return result


def metric_surface(text: str, score_punctuation: bool) -> str:
    """The exact string the scorer tokenizes. Apply to references AND hypotheses.

    Args:
        text: raw manifest reference or raw decoder hypothesis.
        score_punctuation: ``True`` keeps punctuation as part of the words
            (the punctuation-SENSITIVE surface).  ``False`` maps every Unicode
            P* character -- including U+0964 DANDA, U+0965 DOUBLE DANDA and
            U+09FD BENGALI ABBREVIATION SIGN -- to a space, so punctuation acts
            as a word boundary and is never scored as a lexical token (the
            punctuation-INSENSITIVE surface).

    U+2047, the SentencePiece out-of-vocabulary marker, is exempt from the
    punctuation mapping in both modes; see ``PUNCT_EXEMPT``.

    Both modes must be reported.  The two differ by roughly 4.3 pp WER on the
    920 clean fleurs_test_ rows (30.1% of which carry punctuation, against 6.4%
    of the openslr_ rows), so the choice is not cosmetic and must be disclosed.

    Idempotent for a fixed ``score_punctuation``.

    Returns the empty string only for input with no letters and no digits (a
    punctuation-only reference under ``score_punctuation=False``); the scorer
    must keep such a row, count zero reference words, and report the count.

    Raises:
        TypeError: if ``text`` is not a ``str``, or ``score_punctuation`` is not
            a ``bool`` -- passing the flag positionally by mistake is the single
            easiest way to publish two numbers under one label.
        EmptyNormalizationError: if the input contained a letter or a number and
            the output is empty.
    """
    if not isinstance(score_punctuation, bool):
        raise TypeError(
            "metric_surface(text, score_punctuation): score_punctuation must be "
            f"a bool, got {type(score_punctuation).__name__!r}"
        )

    base = normalize_bn_v1(text)
    if score_punctuation:
        return base

    mapped = "".join(" " if _is_scoring_punct(ch) else ch for ch in base)
    result = " ".join(mapped.split())

    if not result and _has_scoreable_content(text):
        raise EmptyNormalizationError(
            f"{NORMALIZER_VERSION} punctuation surface erased a non-empty input "
            f"that contained letters or digits: {text!r} [{_describe(text)}]"
        )
    return result


def audit_residual(text: str) -> dict[str, int]:
    """Count characters a scorer should refuse to score silently.

    Intended for a post-normalization assertion: everything here should be zero
    except ``format_joiner`` (U+200C / U+200D, which are preserved on purpose)
    and ``unk_marker`` (U+2047, the model's own failure marker).
    """
    counts = {
        "control": 0,        # Cc -- must be 0 after normalization
        "format_other": 0,   # Cf other than ZWNJ/ZWJ -- must be 0
        "format_joiner": 0,  # ZWNJ/ZWJ -- expected, preserved deliberately
        "surrogate": 0,      # Cs -- corruption, never normalized away
        "private_use": 0,    # Co -- corruption, never normalized away
        "unassigned": 0,     # Cn
        "unk_marker": 0,     # U+2047 -- decoder emitted an OOV piece
    }
    for ch in text:
        if ch == UNK_MARKER:
            counts["unk_marker"] += 1
            continue
        category = unicodedata.category(ch)
        if category == "Cc":
            counts["control"] += 1
        elif category == "Cf":
            if ch in JOINERS_PRESERVED:
                counts["format_joiner"] += 1
            else:
                counts["format_other"] += 1
        elif category == "Cs":
            counts["surrogate"] += 1
        elif category == "Co":
            counts["private_use"] += 1
        elif category == "Cn":
            counts["unassigned"] += 1
    return counts


def source_sha256() -> str:
    """SHA-256 of this file, for the evaluation metadata sidecar.

    The published WER is only reproducible if the exact normalizer bytes are
    identified alongside it.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(f"{NORMALIZER_VERSION}  unicodedata={UNICODE_VERSION}")
    print(f"sha256={source_sha256()}")
