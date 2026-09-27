"""Pure tokenizer-efficiency metrics (no downloads, fully unit-tested).

`count_words` deliberately splits on whitespace rather than Python's `\\w`
regex: Devanagari text encodes vowel sounds as combining "matra" marks (e.g.
the ``े`` in ``न्यायालय ने``), which `\\w` treats as separate word-boundary
characters, silently over-counting Hindi words. Whitespace-delimited
counting matches how the corpus is actually tokenised for word counts in
the tokenizer study (and matches what a human would call a "word").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


def count_words(text: str) -> int:
    """Whitespace-delimited word count.

    Splitting on whitespace (not `\\w`) matters for Hindi: Python's `\\w`
    regex splits Devanagari words at combining vowel signs (matras), which
    would over-count words. Whitespace splitting is script-agnostic and
    correct for both English and Hindi.
    """
    return len(text.split())


def count_chars(text: str) -> int:
    """Count of non-whitespace Unicode code points in `text`."""
    return sum(1 for ch in text if not ch.isspace())


@dataclass
class LangStats:
    """Aggregate tokenizer statistics for one language over a text sample."""

    tokens: int
    words: int
    chars: int
    unk: int

    @property
    def fertility(self) -> float:
        """Tokens per word."""
        return self.tokens / self.words if self.words else float("nan")

    @property
    def chars_per_token(self) -> float:
        """Non-whitespace characters per token."""
        return self.chars / self.tokens if self.tokens else float("nan")

    @property
    def unk_rate(self) -> float:
        """Fraction of tokens that are the unknown-token id."""
        return self.unk / self.tokens if self.tokens else float("nan")


class _CounterLike(Protocol):
    unk_id: int | None

    def count(self, text: str, lang: str) -> int: ...
    def ids(self, text: str, lang: str) -> list[int]: ...


def corpus_stats(counter: _CounterLike, texts: list[str], lang: str) -> LangStats:
    """Aggregate token/word/char/unk counts for `counter` over `texts`."""
    tokens = 0
    words = 0
    chars = 0
    unk = 0
    for text in texts:
        ids = counter.ids(text, lang)
        tokens += len(ids)
        words += count_words(text)
        chars += count_chars(text)
        if counter.unk_id is not None:
            unk += sum(1 for i in ids if i == counter.unk_id)
    return LangStats(tokens=tokens, words=words, chars=chars, unk=unk)


def pct_over(lengths: list[int], limit: int) -> float:
    """Percentage (0-100) of `lengths` strictly greater than `limit`.

    Returns 0.0 for an empty list.
    """
    if not lengths:
        return 0.0
    return 100.0 * sum(1 for n in lengths if n > limit) / len(lengths)


def judgment_cost_usd(en_tokens: float, hi_tokens: float, price: dict[str, float]) -> dict[str, float]:
    """Estimated USD cost of EN->HI translation of one judgment by an API LLM.

    `price` gives USD per 1M tokens as `{"input": ..., "output": ...}`. Input
    cost is `en_tokens` (the source judgment) at the input rate; output cost
    is `hi_tokens` (the generated Hindi translation) at the output rate.
    This ignores prompt/system-message overhead and any retries — it is a
    lower bound, not a full cost estimate.
    """
    input_cost = en_tokens * price["input"] / 1_000_000
    output_cost = hi_tokens * price["output"] / 1_000_000
    return {
        "input_usd": input_cost,
        "output_usd": output_cost,
        "total_usd": input_cost + output_cost,
    }
