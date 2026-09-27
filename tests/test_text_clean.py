"""Tests for adalat_mt.data.text_clean."""
from __future__ import annotations

from pathlib import Path

from adalat_mt.data.text_clean import clean, fix_ocr_en, fix_ocr_hi, read_text


def test_read_text_strips_bom(tmp_path: Path) -> None:
    p = tmp_path / "hi.txt"
    p.write_bytes("﻿1. नमस्ते\n".encode("utf-8"))
    text = read_text(p)
    assert not text.startswith("﻿")
    assert text.startswith("1. नमस्ते")


def test_read_text_normalises_crlf(tmp_path: Path) -> None:
    p = tmp_path / "en.txt"
    p.write_bytes(b"line one\r\nline two\r\n")
    text = read_text(p)
    assert "\r" not in text
    assert text == "line one\nline two\n"


def test_zwj_removed_from_hindi_word() -> None:
    # "क्‍या" ("क्‍या") -> "क्या" once the ZWJ (U+200D) is stripped.
    raw = "क्‍या"
    cleaned = clean(raw, "hi")
    assert cleaned == "क्या"
    assert "‍" not in cleaned


def test_fix_ocr_en_li_and_lili() -> None:
    text = "i. one\nli. two\nlili. three\n(li) four\n"
    fixed = fix_ocr_en(text)
    assert "i. one" in fixed
    assert "ii. two" in fixed
    assert "iii. three" in fixed
    assert "(ii) four" in fixed


def test_fix_ocr_en_leaves_unrelated_text_alone() -> None:
    text = "Delhi. is a city\nsome li text mid-line\n"
    fixed = fix_ocr_en(text)
    assert fixed == text


def test_fix_ocr_hi_date_space_collapse() -> None:
    assert fix_ocr_hi("17. 10.1998") == "17.10.1998"
    assert fix_ocr_hi("15. 06. 2006") == "15.06.2006"


def test_fix_ocr_hi_date_untouched_without_spaces() -> None:
    assert fix_ocr_hi("27.05.2003") == "27.05.2003"


def test_fix_ocr_hi_danda_dedupe() -> None:
    assert fix_ocr_hi("बात हुई। । अगली बात") == "बात हुई। अगली बात"
    assert fix_ocr_hi("बात। । ।") == "बात।"
