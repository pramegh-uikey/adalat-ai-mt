"""Tests for adalat_mt.data.align, using fake (network-free) embedders."""
from __future__ import annotations

from typing import Callable

import numpy as np

from adalat_mt.data.align import AlignConfig, align_document


def make_concept_embedder(dim: int = 32, seed: int = 0) -> Callable[[list[str]], np.ndarray]:
    """Embed each text as the L2-normalised sum of per-word random unit vectors.

    Shared words across "EN" and "HI" test sentences yield high cosine
    similarity; unrelated words yield near-orthogonal (low) similarity.
    """
    rng = np.random.default_rng(seed)
    word_vecs: dict[str, np.ndarray] = {}

    def word_vec(word: str) -> np.ndarray:
        if word not in word_vecs:
            v = rng.normal(size=dim)
            word_vecs[word] = v / np.linalg.norm(v)
        return word_vecs[word]

    def embed(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            total = np.zeros(dim)
            for word in text.split():
                total += word_vec(word)
            norm = np.linalg.norm(total)
            rows.append(total / norm if norm > 0 else total)
        return np.asarray(rows)

    return embed


def make_controlled_embedder(mapping: dict[str, np.ndarray]) -> Callable[[list[str]], np.ndarray]:
    """A fake embedder returning exactly-specified vectors for known input strings."""

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray([mapping[t] for t in texts])

    return embed


def _covered(pairs, n_src_skipped, n_tgt_skipped, n_src, n_tgt) -> None:
    """Assert pairs are monotonic, contiguous and non-overlapping, and totals add up."""
    prev_src_end = 0
    prev_tgt_end = 0
    for p in pairs:
        assert p.src_idx == list(range(p.src_idx[0], p.src_idx[-1] + 1))
        assert p.tgt_idx == list(range(p.tgt_idx[0], p.tgt_idx[-1] + 1))
        assert p.src_idx[0] >= prev_src_end
        assert p.tgt_idx[0] >= prev_tgt_end
        prev_src_end = p.src_idx[-1] + 1
        prev_tgt_end = p.tgt_idx[-1] + 1
    total_src = sum(len(p.src_idx) for p in pairs) + n_src_skipped
    total_tgt = sum(len(p.tgt_idx) for p in pairs) + n_tgt_skipped
    assert total_src == n_src
    assert total_tgt == n_tgt


def test_identity_alignment_equal_length_docs() -> None:
    embed = make_concept_embedder()
    src = ["cat runs", "dog sleeps", "bird flies"]
    tgt = ["cat runs", "dog sleeps", "bird flies"]
    cfg = AlignConfig()
    result = align_document(src, tgt, embed, cfg)

    assert [p.align_type for p in result.pairs] == ["1-1", "1-1", "1-1"]
    assert all(p.kept for p in result.pairs)
    for p in result.pairs:
        assert p.score > 0.99
    _covered(result.pairs, result.n_src_skipped, result.n_tgt_skipped, result.n_src, result.n_tgt)


def test_target_sentence_split_in_two_gives_1_2_pair() -> None:
    embed = make_concept_embedder()
    src = ["alpha beta"]
    tgt = ["alpha", "beta"]
    cfg = AlignConfig()
    result = align_document(src, tgt, embed, cfg)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.align_type == "1-2"
    assert pair.src_idx == [0]
    assert pair.tgt_idx == [0, 1]
    assert pair.kept


def test_extra_junk_target_sentence_is_skipped() -> None:
    embed = make_concept_embedder()
    src = ["cat runs", "dog sleeps"]
    tgt = ["zzz junk qqq", "cat runs", "dog sleeps"]
    cfg = AlignConfig()
    result = align_document(src, tgt, embed, cfg)

    assert result.n_tgt_skipped == 1
    assert result.n_src_skipped == 0
    kept_pairs = [p for p in result.pairs if p.kept]
    assert [p.align_type for p in kept_pairs] == ["1-1", "1-1"]
    assert kept_pairs[0].tgt_idx == [1]
    assert kept_pairs[1].tgt_idx == [2]
    _covered(result.pairs, result.n_src_skipped, result.n_tgt_skipped, result.n_src, result.n_tgt)


def test_output_is_monotonic_contiguous_and_disjoint() -> None:
    embed = make_concept_embedder()
    src = ["cat runs", "dog sleeps fast", "bird flies", "fish swims"]
    tgt = ["junk qqq", "cat runs", "dog sleeps", "fast", "bird flies", "fish swims"]
    cfg = AlignConfig()
    result = align_document(src, tgt, embed, cfg)
    _covered(result.pairs, result.n_src_skipped, result.n_tgt_skipped, result.n_src, result.n_tgt)


def test_low_similarity_pair_dropped_with_low_score() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(0.75)])  # cos(v1, v2) == 0.5
    mapping = {"Alpha sentence.": v1, "Beta sentence.": v2}
    embed = make_controlled_embedder(mapping)

    cfg = AlignConfig(skip_threshold=0.40, min_score=0.60)
    result = align_document(["Alpha sentence."], ["Beta sentence."], embed, cfg)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert abs(pair.score - 0.5) < 1e-6
    assert not pair.kept
    assert pair.drop_reason == "low_score"


def test_len_ratio_filter_drops_mismatched_lengths() -> None:
    short_en = "short"
    long_hi = "x" * 300
    v = np.array([1.0, 0.0])
    mapping = {short_en: v, long_hi: v}
    embed = make_controlled_embedder(mapping)

    cfg = AlignConfig(min_score=0.60, len_ratio=(0.5, 2.5))
    result = align_document([short_en], [long_hi], embed, cfg)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.score > cfg.min_score
    assert not pair.kept
    assert pair.drop_reason == "len_ratio"


def test_wilson_interval_known_value() -> None:
    from adalat_mt.data.report import wilson_interval

    lo, hi = wilson_interval(19, 20)
    assert round(lo, 3) == 0.764 and round(hi, 3) == 0.991
    assert wilson_interval(0, 0) == (0.0, 0.0)
