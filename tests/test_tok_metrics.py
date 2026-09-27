"""Unit tests for `adalat_mt.tokenization.metrics` (no downloads)."""
from __future__ import annotations

import math

import pytest

from adalat_mt.tokenization.metrics import (
    LangStats,
    corpus_stats,
    count_chars,
    count_words,
    judgment_cost_usd,
    pct_over,
)


def test_count_words_hindi_matras():
    # Devanagari matras (े, ी) must not be treated as word boundaries by a
    # naive \w-regex splitter; whitespace splitting gets this right.
    assert count_words("न्यायालय ने अपील खारिज की।") == 5


def test_count_words_english():
    assert count_words("The court dismissed the appeal.") == 5


def test_count_words_empty_and_whitespace():
    assert count_words("") == 0
    assert count_words("   \n\t  ") == 0


def test_count_chars_ignores_whitespace():
    assert count_chars("ab cd") == 4
    assert count_chars("न्याय") == len("न्याय")  # all non-whitespace code points count
    assert count_chars("") == 0


class _FakeCounter:
    """Minimal TokenCounter stub: 1 token per non-whitespace char, with a fixed UNK id."""

    def __init__(self, unk_id=None, unk_chars=()):
        self.unk_id = unk_id
        self._unk_chars = set(unk_chars)

    def ids(self, text, lang):
        out = []
        for ch in text:
            if ch.isspace():
                continue
            out.append(self.unk_id if ch in self._unk_chars and self.unk_id is not None else 1)
        return out

    def count(self, text, lang):
        return len(self.ids(text, lang))


def test_lang_stats_fertility_and_chars_per_token():
    stats = LangStats(tokens=10, words=5, chars=20, unk=2)
    assert stats.fertility == 2.0
    assert stats.chars_per_token == 2.0
    assert stats.unk_rate == 0.2


def test_lang_stats_zero_division_is_nan():
    stats = LangStats(tokens=0, words=0, chars=0, unk=0)
    assert math.isnan(stats.fertility)
    assert math.isnan(stats.chars_per_token)
    assert math.isnan(stats.unk_rate)


def test_corpus_stats_counts_tokens_words_chars_unk():
    counter = _FakeCounter(unk_id=0, unk_chars={"x"})
    texts = ["ab cd", "xy"]  # 4 non-space chars first text, 2 second -> 6 tokens total
    stats = corpus_stats(counter, texts, "en")
    assert stats.tokens == 6
    assert stats.words == 3  # "ab","cd" + "xy"
    assert stats.chars == 6
    assert stats.unk == 1  # only the "x" in "xy"


def test_pct_over_basic():
    assert pct_over([10, 20, 30, 40], 20) == 50.0


def test_pct_over_all_over():
    assert pct_over([100, 200], 10) == 100.0


def test_pct_over_none_over():
    assert pct_over([1, 2, 3], 10) == 0.0


def test_pct_over_empty_list():
    assert pct_over([], 10) == 0.0


def test_pct_over_boundary_is_strict():
    # exactly at the limit does not count as "over"
    assert pct_over([128, 128], 128) == 0.0


def test_judgment_cost_usd_arithmetic():
    price = {"input": 2.0, "output": 10.0}
    cost = judgment_cost_usd(en_tokens=1_000_000, hi_tokens=500_000, price=price)
    assert cost["input_usd"] == pytest.approx(2.0)
    assert cost["output_usd"] == pytest.approx(5.0)
    assert cost["total_usd"] == pytest.approx(7.0)


def test_judgment_cost_usd_zero_tokens():
    cost = judgment_cost_usd(0, 0, {"input": 1.0, "output": 1.0})
    assert cost == {"input_usd": 0.0, "output_usd": 0.0, "total_usd": 0.0}
