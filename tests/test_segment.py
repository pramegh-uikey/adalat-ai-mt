"""Tests for adalat_mt.data.segment."""
from __future__ import annotations

from adalat_mt.data.segment import (
    MARKER_RE,
    leading_para_number,
    split_sentences,
    unwrap,
)


def test_unwrap_joins_hard_wrapped_lines() -> None:
    text = "1. The Appellant worked as an officer\nof the Bank. He was served\nwith notice."
    segs = unwrap(text)
    assert segs == ["1. The Appellant worked as an officer of the Bank. He was served with notice."]


def test_unwrap_joins_spurious_blank_line_mid_sentence() -> None:
    text = "1. You did not comply the directions and regulations of\n\nHead Office.\n"
    segs = unwrap(text)
    assert segs == ["1. You did not comply the directions and regulations of Head Office."]


def test_unwrap_blank_line_after_terminal_punct_starts_new_segment() -> None:
    text = "First sentence ends here.\n\nSecond paragraph starts."
    segs = unwrap(text)
    assert segs == ["First sentence ends here.", "Second paragraph starts."]


def test_unwrap_marker_line_starts_new_segment() -> None:
    text = "1. First paragraph text.\n2. Second paragraph text."
    segs = unwrap(text)
    assert segs == ["1. First paragraph text.", "2. Second paragraph text."]


def test_unwrap_drops_page_number_lines() -> None:
    text = "1. Some text continues\n7\nacross a page break."
    segs = unwrap(text)
    assert segs == ["1. Some text continues across a page break."]


def test_unwrap_hyphen_join_no_space() -> None:
    text = "This is a self-\nemployed person."
    segs = unwrap(text)
    assert segs == ["This is a self-employed person."]


def test_marker_re_matches() -> None:
    assert MARKER_RE.match("1. Leave granted.")
    assert MARKER_RE.match("(ii) The accused")
    assert MARKER_RE.match("ii. The accused")
    assert MARKER_RE.match("(a) some clause")
    assert MARKER_RE.match("a) some clause")
    assert not MARKER_RE.match("Random text.")


def test_en_split_respects_abbreviations() -> None:
    sentences = split_sentences(
        "The case No. 55 of 2008 was filed. It cites U.P. Zamidari Abolition Act.", "en"
    )
    assert sentences == [
        "The case No. 55 of 2008 was filed.",
        "It cites U.P. Zamidari Abolition Act.",
    ]


def test_en_split_on_vs_and_initials() -> None:
    sentences = split_sentences(
        "Smt. Sharma vs. State of U.P. is the citation. R. K. Sharma appeared for the appellant.",
        "en",
    )
    assert sentences == [
        "Smt. Sharma vs. State of U.P. is the citation.",
        "R. K. Sharma appeared for the appellant.",
    ]


def test_en_split_does_not_split_leading_marker() -> None:
    assert split_sentences("1. Leave granted.", "en") == ["1. Leave granted."]
    assert split_sentences("(ii) The accused was present. He denied it.", "en") == [
        "(ii) The accused was present.",
        "He denied it.",
    ]


def test_hi_split_on_danda_and_question() -> None:
    sentences = split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है क्या? हाँ यह तीसरा है।", "hi")
    assert sentences == [
        "यह पहला वाक्य है।",
        "यह दूसरा वाक्य है क्या?",
        "हाँ यह तीसरा है।",
    ]


def test_hi_split_not_on_known_abbreviations() -> None:
    sentences = split_sentences("प्रतिवादी सं. 2 से 5 के पक्ष में विक्रय हुआ। मामला समाप्त हुआ।", "hi")
    assert sentences == [
        "प्रतिवादी सं. 2 से 5 के पक्ष में विक्रय हुआ।",
        "मामला समाप्त हुआ।",
    ]


def test_hi_split_not_on_dotted_acronym() -> None:
    sentences = split_sentences(
        "यह मामला ए.एल.आर. 517 में दर्ज है। अगला वाक्य यहाँ है।", "hi"
    )
    assert sentences == [
        "यह मामला ए.एल.आर. 517 में दर्ज है।",
        "अगला वाक्य यहाँ है।",
    ]


def test_leading_para_number() -> None:
    assert leading_para_number("7. We have heard the parties.") == 7
    assert leading_para_number("No number here.") is None
    assert leading_para_number("(ii) The accused") is None


def test_hi_danda_without_space_splits() -> None:
    out = split_sentences("यह पहला वाक्य है।यह दूसरा वाक्य है।", "hi")
    assert out == ["यह पहला वाक्य है।", "यह दूसरा वाक्य है।"]


def test_multilevel_para_marker_not_split_off() -> None:
    out = split_sentences("366.1. The expression is wide. It covers everything.", "en")
    assert out == ["366.1. The expression is wide.", "It covers everything."]
