"""Unicode/OCR cleanup for the raw English and Hindi judgment text files.

The source files are OCR/PDF extractions with a few systematic quirks (BOM,
zero-width joiners in Hindi, mis-recognised roman numerals in English,
spaced-out dates and duplicated dandas in Hindi). These functions fix those
quirks without touching anything else.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

# Zero-width space/non-joiner/joiner, BOM, soft hyphen.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_NBSP_RE = re.compile("\u00a0")

_OCR_LI_RE = re.compile(r"^([ \t]*\(?)([li]+)([.)])", re.MULTILINE)

_HI_DATE_RE = re.compile(r"\b(\d{1,2})\.(\s*)(\d{1,2})\.(\s*)(\d{4})\b")
_HI_DANDA_DUP_RE = re.compile(r"।(?:\s*।)+")


def _normalize_unicode(text: str) -> str:
    """Normalise newlines/Unicode form and strip invisible characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _NBSP_RE.sub(" ", text)
    return text


def read_text(path: Path) -> str:
    """Read a UTF-8 (optionally BOM-prefixed) text file and normalise it.

    Strips a leading BOM, normalises `\\r\\n` to `\\n`, applies Unicode NFC,
    removes zero-width characters and replaces NBSP with a regular space.
    """
    raw = Path(path).read_text(encoding="utf-8-sig")
    return _normalize_unicode(raw)


def fix_ocr_en(text: str) -> str:
    """Fix line-initial roman-numeral OCR artefacts (``li.`` -> ``ii.`` etc).

    Only touches a token made purely of ``l``/``i`` characters that sits at
    the start of a line (optionally after spaces or an opening paren) and is
    immediately followed by ``.`` or ``)``. The token is normalised to a run
    of ``i`` characters, capped at three (``i``, ``ii``, ``iii``), since those
    are the only roman numerals this OCR artefact can stand for. Everything
    else — mid-line ``li``, or words like ``Delhi.`` that aren't purely
    ``l``/``i`` — is left untouched.
    """

    def _repl(match: re.Match[str]) -> str:
        prefix, token, term = match.group(1), match.group(2), match.group(3)
        fixed = "i" * min(len(token), 3)
        return f"{prefix}{fixed}{term}"

    return _OCR_LI_RE.sub(_repl, text)


def fix_ocr_hi(text: str) -> str:
    """Collapse OCR-inserted spaces inside Hindi dates and duplicated dandas.

    ``17. 10.1998`` / ``15. 06. 2006`` -> ``17.10.1998`` / ``15.06.2006``
    (only when at least one space was present; digits are never changed).
    ``। ।`` (and longer runs) collapse to a single ``।``.
    """

    def _date_repl(match: re.Match[str]) -> str:
        d1, sp1, d2, sp2, d3 = match.groups()
        if sp1 or sp2:
            return f"{d1}.{d2}.{d3}"
        return match.group(0)

    text = _HI_DATE_RE.sub(_date_repl, text)
    text = _HI_DANDA_DUP_RE.sub("।", text)
    return text


def clean(text: str, lang: Literal["en", "hi"]) -> str:
    """Normalise Unicode and fix language-specific OCR artefacts.

    Composes `_normalize_unicode` with `fix_ocr_en`/`fix_ocr_hi` depending on
    `lang`.
    """
    text = _normalize_unicode(text)
    if lang == "en":
        return fix_ocr_en(text)
    if lang == "hi":
        return fix_ocr_hi(text)
    raise ValueError(f"unknown lang: {lang!r}")
