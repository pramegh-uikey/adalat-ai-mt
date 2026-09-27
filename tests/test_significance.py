"""Tests for `adalat_mt.evaluation.significance` (no network / no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest
from sacrebleu.metrics import BLEU, CHRF

from adalat_mt.evaluation.significance import paired_bootstrap, paired_bootstrap_segments

REFS = [
    "The court granted bail to the accused.",
    "The appeal was dismissed by the bench.",
    "The petitioner filed a writ petition.",
    "Notice was issued to the respondent.",
    "The matter is listed for further hearing.",
    "The learned counsel appeared for the state.",
]
GOOD = REFS  # a "system" that reproduces the reference exactly
BAD = [
    "Xyz abc def.",
    "Completely unrelated text here.",
    "Nothing like the reference at all.",
    "Random words in a random order.",
    "This has no overlap with the reference.",
    "Another unrelated sentence entirely.",
]


@pytest.mark.parametrize("metric,factory", [("bleu", lambda: BLEU(tokenize="13a")), ("chrf", lambda: CHRF(word_order=2))])
def test_full_index_sufficient_stats_equals_corpus_score(metric, factory) -> None:
    m = factory()
    stats = np.array(m._extract_corpus_statistics(GOOD, [REFS]))
    full_idx = np.arange(stats.shape[0])
    recomputed = m._compute_score_from_stats(stats[full_idx].sum(0).tolist()).score
    corpus = m.corpus_score(GOOD, [REFS]).score
    assert recomputed == corpus


def test_identical_systems_zero_delta_and_p_one() -> None:
    result = paired_bootstrap(GOOD, GOOD, REFS, metric="bleu", n_samples=200, seed=13)
    assert result["delta"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0)


def test_clearly_better_system_low_p_value() -> None:
    # a = BAD, b = GOOD -> b is clearly better than a.
    result = paired_bootstrap(BAD, GOOD, REFS, metric="chrf", n_samples=1000, seed=13)
    assert result["b_score"] > result["a_score"]
    assert result["delta"] > 0
    assert result["p_value"] < 0.05


def test_determinism_for_seed() -> None:
    r1 = paired_bootstrap(BAD, GOOD, REFS, metric="bleu", n_samples=300, seed=42)
    r2 = paired_bootstrap(BAD, GOOD, REFS, metric="bleu", n_samples=300, seed=42)
    assert r1 == r2


def test_segment_version_identical_and_better() -> None:
    scores = [80.0, 82.0, 79.0, 85.0, 81.0, 83.0]
    identical = paired_bootstrap_segments(scores, scores, n_samples=200, seed=13)
    assert identical["delta"] == pytest.approx(0.0, abs=1e-9)
    assert identical["p_value"] == pytest.approx(1.0)

    worse = [40.0, 42.0, 39.0, 45.0, 41.0, 43.0]
    better = paired_bootstrap_segments(worse, scores, n_samples=1000, seed=13)
    assert better["delta"] > 0
    assert better["p_value"] < 0.05


def test_output_schema() -> None:
    result = paired_bootstrap(BAD, GOOD, REFS, metric="spbleu", n_samples=50, seed=1)
    assert set(result) == {"metric", "a_score", "b_score", "delta", "ci95", "p_value"}
    assert result["metric"] == "spbleu"
    assert len(result["ci95"]) == 2
