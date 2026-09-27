"""Tests for `adalat_mt.evaluation.metrics` (no network / no model downloads)."""

from __future__ import annotations

import pytest
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from adalat_mt.evaluation.metrics import corpus_metrics, sentence_chrf

HYPS = ["The court granted bail.", "The appeal was dismissed."]
REFS = ["The court granted bail.", "The appeal was dismissed."]


def test_perfect_hyps_score_100() -> None:
    metrics = corpus_metrics(HYPS, REFS)
    assert metrics["bleu"] == pytest.approx(100.0, abs=1e-6)
    assert metrics["chrf"] == pytest.approx(100.0, abs=1e-6)


def test_matches_direct_sacrebleu_calls() -> None:
    hyps = ["This is a small test.", "It has two lines."]
    refs = ["This is a small test.", "It has a different second line."]
    metrics = corpus_metrics(hyps, refs)

    bleu_direct = BLEU(tokenize="13a").corpus_score(hyps, [refs])
    chrf_direct = CHRF(word_order=2).corpus_score(hyps, [refs])

    assert metrics["bleu"] == round(bleu_direct.score, 2)
    assert metrics["chrf"] == round(chrf_direct.score, 2)
    assert metrics["bleu_tokenize"] == "13a"
    assert metrics["chrf_word_order"] == 2


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        corpus_metrics(["one", "two"], ["only one"])
    with pytest.raises(ValueError):
        sentence_chrf(["one", "two"], ["only one"])


def test_signatures_present() -> None:
    metrics = corpus_metrics(HYPS, REFS)
    sigs = metrics["signatures"]
    assert set(sigs) == {"bleu", "spbleu", "chrf"}
    for sig in sigs.values():
        assert isinstance(sig, str) and len(sig) > 0
        assert "version:" in sig


def test_sentence_chrf_per_segment() -> None:
    scores = sentence_chrf(HYPS, REFS)
    assert len(scores) == len(HYPS)
    for s in scores:
        assert s == pytest.approx(100.0, abs=1e-6)

    scores_bad = sentence_chrf(["a completely different sentence"], ["The court granted bail."])
    assert scores_bad[0] < 50.0


@pytest.mark.slow
def test_spbleu_flores200_tokenizer() -> None:
    """spBLEU needs the flores200 SPM model; sacrebleu downloads/caches it under ~/.sacrebleu on first use."""
    metrics = corpus_metrics(HYPS, REFS)
    assert metrics["spbleu"] == pytest.approx(100.0, abs=1e-6)
    # sanity: sacrebleu's own flores200-tokenized BLEU agrees.
    direct = sacrebleu.BLEU(tokenize="flores200").corpus_score(HYPS, [REFS])
    assert metrics["spbleu"] == round(direct.score, 2)
