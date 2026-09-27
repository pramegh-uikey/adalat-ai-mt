"""Paired bootstrap resampling significance tests (Koehn, 2004)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

_METRIC_FACTORIES = {
    "bleu": lambda: BLEU(tokenize="13a"),
    "spbleu": lambda: BLEU(tokenize="flores200"),
    "chrf": lambda: CHRF(word_order=2),
}


def _score_from_stats(metric, stats: np.ndarray) -> float:
    return metric._compute_score_from_stats(stats.tolist()).score


def paired_bootstrap(
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    metric: Literal["bleu", "spbleu", "chrf"] = "bleu",
    n_samples: int = 1000,
    seed: int = 13,
) -> dict:
    """Koehn (2004) paired bootstrap test comparing system B against system A on a shared reference.

    Uses sacrebleu's per-segment sufficient statistics (``metric._extract_corpus_statistics``) so each of
    the `n_samples` resamples can be scored in O(1) via ``metric._compute_score_from_stats(stats[idx].sum(0))``
    instead of re-tokenizing. Segment indices are resampled with replacement using
    ``numpy.random.default_rng(seed)``.

    Returns ``{"metric", "a_score", "b_score", "delta", "ci95", "p_value"}`` where ``delta = b_score -
    a_score`` on the full corpus, ``ci95`` is the [2.5, 97.5] percentile interval of the bootstrap `delta`
    distribution, and ``p_value`` is the fraction of resamples with ``delta <= 0`` — a one-sided test of
    H1: system B is better than system A (smaller p means stronger evidence B > A).
    """
    if len(hyps_a) != len(refs) or len(hyps_b) != len(refs):
        raise ValueError("length mismatch between hyps_a/hyps_b/refs")

    m = _METRIC_FACTORIES[metric]()
    stats_a = np.array(m._extract_corpus_statistics(hyps_a, [refs]))
    stats_b = np.array(m._extract_corpus_statistics(hyps_b, [refs]))
    n = stats_a.shape[0]

    a_score = _score_from_stats(m, stats_a.sum(0))
    b_score = _score_from_stats(m, stats_b.sum(0))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _score_from_stats(m, stats_b[idx].sum(0)) - _score_from_stats(m, stats_a[idx].sum(0))

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float(np.mean(deltas <= 0))
    return {
        "metric": metric,
        "a_score": round(a_score, 2),
        "b_score": round(b_score, 2),
        "delta": round(b_score - a_score, 2),
        "ci95": [round(float(ci_lo), 2), round(float(ci_hi), 2)],
        "p_value": p_value,
    }


def paired_bootstrap_segments(
    scores_a: list[float],
    scores_b: list[float],
    n_samples: int = 1000,
    seed: int = 13,
    metric: str = "comet",
) -> dict:
    """Paired bootstrap test for pre-computed segment-level scores (e.g. COMET), mean aggregation.

    Same output schema as `paired_bootstrap`, but the corpus score is the arithmetic mean of the segment
    scores (rather than a sacrebleu sufficient-statistic aggregate), and each resample takes the mean of a
    with-replacement resample of segment indices. ``delta = mean(b) - mean(a)``; ``p_value`` is the fraction
    of resamples with ``delta <= 0`` (one-sided, H1: B better than A).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("length mismatch between scores_a and scores_b")
    n = len(a)

    a_score = float(a.mean())
    b_score = float(b.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = b[idx].mean() - a[idx].mean()

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float(np.mean(deltas <= 0))
    return {
        "metric": metric,
        "a_score": round(a_score, 4),
        "b_score": round(b_score, 4),
        "delta": round(b_score - a_score, 4),
        "ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "p_value": p_value,
    }
