"""Paragraph unwrapping and rule-based sentence segmentation for EN/HI legal text.

Pipeline: `text_clean.clean` -> `unwrap` (hard-wrap -> paragraph-like segments)
-> `split_sentences` (rule-based, abbreviation-aware).
"""
from __future__ import annotations

import re
import string
from typing import Literal

from .text_clean import clean

#: Leading list/paragraph markers: "1.", "(ii)"/"ii.", "(a)", "a)".
MARKER_RE = re.compile(
    r"^(?:\d{1,3}(?:\.\d{1,3})*\.(?:\s|$)"
    r"|\d{1,3}(?:\.\d{1,3})+(?:\s|$)"
    r"|\(?[ivxlc]{1,6}[.)](?:\s|$)"
    r"|\([a-zA-Z]\)(?:\s|$)"
    r"|[a-z]\)(?:\s|$))"
)

_TERMINAL_PUNCT = set('.?!:;।"”’')
_ASCII_LOWER = set(string.ascii_lowercase)

_EN_ABBR = {
    "no", "nos", "vs", "v", "mr", "mrs", "ms", "dr", "sr", "jr", "ltd", "pvt", "co", "corp",
    "inc", "st", "hon", "ors", "anr", "etc", "viz", "i.e", "e.g", "cf", "art", "arts", "sec",
    "secs", "s", "ss", "cl", "para", "paras", "p", "pp", "vol", "cr", "crl", "civ", "govt",
    "dept", "addl", "asst", "dist", "distt", "div", "sri", "smt", "shri", "kum", "j", "jj",
    "cj", "rs", "ch", "ibid", "approx", "sl", "nr", "regd", "misc", "spl", "ex", "u/s",
}

# Punctuation + optional closing quote(s)/bracket + required whitespace, with a
# lookahead requiring the next non-space char to look sentence-initial.
_EN_SPLIT_RE = re.compile(
    r'([.?!])(["”’)\]]*)(\s+)(?=[A-Z0-9"“‘\'(\[])'
)
_EN_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*)$")

_HI_ABBR = {
    "सं", "क्र", "नं", "डॉ", "श्री", "प्र", "पृ", "मा", "उ.प्र", "वि", "अनु", "पं",
}
_DEVANAGARI = r"ऀ-ॿ"
# A danda ends a sentence even when the next word follows without a space (common in the source files);
# `?`, `!` and `.` need trailing whitespace.
_HI_SPLIT_RE = re.compile(
    r"(।)([\"”’]*)(\s*)|([?!.])([\"”’]*)(\s+)"
)
_HI_WORD_BEFORE_RE = re.compile(
    rf"([{_DEVANAGARI}]+(?:\.[{_DEVANAGARI}]+)*)$"
)
_HI_ACRONYM_RE = re.compile(rf"(?:[{_DEVANAGARI}]{{1,3}}\.){{1,}}[{_DEVANAGARI}]{{1,3}}$")

_LEADING_NUM_RE = re.compile(r"^\s*(\d{1,3})\.(?:\s|$)")


def unwrap(text: str) -> list[str]:
    """Join hard-wrapped lines into paragraph-like segments.

    A new segment starts when a line matches `MARKER_RE`, or when there was
    at least one blank line before it and the previous non-empty line ends
    with terminal punctuation. Otherwise the line is joined to the current
    segment with a single space (or with no space, keeping the hyphen, when
    the previous line ends with ``-`` and this line starts with a lowercase
    ASCII letter). Lines consisting only of digits (page numbers) are
    dropped. Internal runs of whitespace are collapsed.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    blank_run = 0

    def flush() -> None:
        if current:
            segments.append(list(current))
            current.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            blank_run += 1
            continue
        if line.isdigit():
            # Page number: drop silently, don't affect blank/segment state.
            continue

        starts_new = bool(MARKER_RE.match(line)) or (
            blank_run >= 1 and bool(current) and current[-1][-1:] in _TERMINAL_PUNCT
        )
        if starts_new:
            flush()
            current.append(line)
        elif current and current[-1].endswith("-") and line[:1] in _ASCII_LOWER:
            current[-1] = current[-1] + line
        else:
            current.append(line)
        blank_run = 0
    flush()

    joined = (" ".join(seg) for seg in segments)
    return [re.sub(r"\s+", " ", s).strip() for s in joined if s.strip()]


def _is_en_abbrev(word: str) -> bool:
    if not word:
        return False
    if re.fullmatch(r"[A-Za-z]", word):
        return True
    if re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])+", word):
        return True
    return word.lower() in _EN_ABBR


def _split_en(segment: str) -> list[str]:
    marker_end = 0
    m = MARKER_RE.match(segment)
    if m:
        marker_end = m.end()

    boundaries: list[int] = []
    for cand in _EN_SPLIT_RE.finditer(segment):
        if cand.start() < marker_end:
            continue
        window = segment[max(0, cand.start() - 40):cand.start()]
        wm = _EN_WORD_BEFORE_RE.search(window)
        word = wm.group(1) if wm else ""
        if _is_en_abbrev(word):
            continue
        boundaries.append(cand.end())

    return _slice(segment, boundaries)


def _is_hi_abbrev(word: str) -> bool:
    if not word:
        return True
    if word in _HI_ABBR:
        return True
    if _HI_ACRONYM_RE.fullmatch(word):
        return True
    plain_len = len(word.replace(".", ""))
    return plain_len < 3


def _split_hi(segment: str) -> list[str]:
    marker_end = 0
    m = MARKER_RE.match(segment)
    if m:
        marker_end = m.end()

    boundaries: list[int] = []
    for cand in _HI_SPLIT_RE.finditer(segment):
        if cand.start() < marker_end:
            continue
        punct = cand.group(1) or cand.group(4)
        if punct == ".":
            window = segment[max(0, cand.start() - 40):cand.start()]
            wm = _HI_WORD_BEFORE_RE.search(window)
            word = wm.group(1) if wm else ""
            if _is_hi_abbrev(word):
                continue
        boundaries.append(cand.end())

    return _slice(segment, boundaries)


def _slice(segment: str, boundaries: list[int]) -> list[str]:
    sentences: list[str] = []
    start = 0
    for b in boundaries:
        sentences.append(segment[start:b])
        start = b
    sentences.append(segment[start:])
    return [s.strip() for s in sentences if s.strip()]


def split_sentences(segment: str, lang: Literal["en", "hi"]) -> list[str]:
    """Split a paragraph-like segment into sentences using EN/HI-specific rules.

    Never splits a leading list/paragraph marker off its sentence.
    """
    if lang == "en":
        return _split_en(segment)
    if lang == "hi":
        return _split_hi(segment)
    raise ValueError(f"unknown lang: {lang!r}")


def leading_para_number(sentence: str) -> int | None:
    """Return the leading paragraph number of `sentence` (e.g. 7 for "7. We...") or None."""
    m = _LEADING_NUM_RE.match(sentence)
    return int(m.group(1)) if m else None


def segment_document(text: str, lang: Literal["en", "hi"]) -> list[str]:
    """Clean, unwrap and sentence-split a whole document's raw text."""
    cleaned = clean(text, lang)
    sentences: list[str] = []
    for seg in unwrap(cleaned):
        sentences.extend(split_sentences(seg, lang))
    return sentences
