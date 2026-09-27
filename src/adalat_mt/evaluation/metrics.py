"""Corpus- and segment-level MT quality metrics (BLEU, spBLEU, chrF++) via sacrebleu."""

from __future__ import annotations

from sacrebleu.metrics import BLEU, CHRF

BLEU_TOKENIZE = "13a"
CHRF_WORD_ORDER = 2


def _check_lengths(hyps: list[str], refs: list[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(f"length mismatch: {len(hyps)} hypotheses vs {len(refs)} references")


def corpus_metrics(hyps: list[str], refs: list[str]) -> dict:
    """Corpus-level BLEU (tokenize=13a), spBLEU (tokenize=flores200) and chrF++ (word_order=2).

    Returns ``{"bleu", "spbleu", "chrf"}`` scores rounded to 2 dp, plus ``"bleu_tokenize"``,
    ``"chrf_word_order"`` and the sacrebleu ``"signatures"`` of the three metrics (for reproducibility).
    Raises `ValueError` if `hyps` and `refs` have different lengths.
    """
    _check_lengths(hyps, refs)
    bleu = BLEU(tokenize=BLEU_TOKENIZE)
    spbleu = BLEU(tokenize="flores200")
    chrf = CHRF(word_order=CHRF_WORD_ORDER)

    bleu_score = bleu.corpus_score(hyps, [refs])
    spbleu_score = spbleu.corpus_score(hyps, [refs])
    chrf_score = chrf.corpus_score(hyps, [refs])

    return {
        "bleu": round(bleu_score.score, 2),
        "spbleu": round(spbleu_score.score, 2),
        "chrf": round(chrf_score.score, 2),
        "bleu_tokenize": BLEU_TOKENIZE,
        "chrf_word_order": CHRF_WORD_ORDER,
        "signatures": {
            "bleu": str(bleu.get_signature()),
            "spbleu": str(spbleu.get_signature()),
            "chrf": str(chrf.get_signature()),
        },
    }


def sentence_chrf(hyps: list[str], refs: list[str]) -> list[float]:
    """Per-segment chrF++ (word_order=2) scores, rounded to 2 dp. Raises on length mismatch."""
    _check_lengths(hyps, refs)
    chrf = CHRF(word_order=CHRF_WORD_ORDER)
    return [round(chrf.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, refs)]
