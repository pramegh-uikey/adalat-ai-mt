"""Tests for `adalat_mt.tokenization.embedding_probe` (no Hub downloads)."""
from __future__ import annotations

import math

import pytest
import torch

from adalat_mt.tokenization.embedding_probe import (
    compute_metrics,
    encode_batch,
    forward_nll,
    score_corpus,
)


# --------------------------------------------------------------------------
# fakes / fixtures: tiny random LlamaForCausalLM (vocab 64, hidden 16, 1 layer)
# --------------------------------------------------------------------------


def _tiny_llama(vocab_size: int = 64):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class _FakeTokenizer:
    """Character-level stub tokenizer over a tiny fixed alphabet.

    ids 0..1 reserved for bos/pad; ids 2.. map 1:1 to characters in
    `alphabet`, in order, so tests can predict exact token ids/counts.
    """

    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 1

    def __init__(self, alphabet: str):
        self._char_to_id = {ch: 2 + i for i, ch in enumerate(alphabet)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._char_to_id[ch] for ch in text]


# --------------------------------------------------------------------------
# encode_batch
# --------------------------------------------------------------------------


def test_encode_batch_prepends_bos_and_right_pads():
    tok = _FakeTokenizer("abc")
    input_ids, attention_mask = encode_batch(tok, ["ab", "a"])
    assert input_ids.shape == (2, 3)  # bos + 2 chars is the longest sequence
    assert input_ids[0].tolist() == [0, 2, 3]  # bos, a, b
    assert input_ids[1].tolist() == [0, 2, 1]  # bos, a, pad
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]


# --------------------------------------------------------------------------
# forward_nll: batching/padding must not change the result vs. unpadded singles
# --------------------------------------------------------------------------


def test_forward_nll_batch_matches_sum_of_unbatched_singles():
    model = _tiny_llama(vocab_size=64)
    tok = _FakeTokenizer("abcdefgh")
    texts = ["abc", "de", "fgh"]

    batch_ids, batch_mask = encode_batch(tok, texts)
    batch_nll, batch_n, batch_new = forward_nll(model, batch_ids, batch_mask, base_vocab_size=60)

    total_nll = 0.0
    total_n = 0
    total_new = 0
    for t in texts:
        ids, mask = encode_batch(tok, [t])
        nll, n, new = forward_nll(model, ids, mask, base_vocab_size=60)
        total_nll += nll
        total_n += n
        total_new += new

    assert batch_n == total_n
    assert batch_new == total_new
    assert batch_nll == pytest.approx(total_nll, abs=1e-3)


def test_forward_nll_counts_only_non_pad_predicted_tokens():
    model = _tiny_llama(vocab_size=64)
    tok = _FakeTokenizer("abcdefgh")
    # "abc" -> 4 real tokens (bos,a,b,c); "de" -> 3 real tokens (bos,d,e), padded to len 4.
    ids, mask = encode_batch(tok, ["abc", "de"])
    _, n_tokens, _ = forward_nll(model, ids, mask, base_vocab_size=60)
    # predicted tokens = (len - 1) per real sequence: 3 for "abc", 2 for "de"
    assert n_tokens == 3 + 2


def test_forward_nll_new_token_share_uses_base_vocab_size_threshold():
    model = _tiny_llama(vocab_size=64)
    tok = _FakeTokenizer("abcdefgh")
    ids, mask = encode_batch(tok, ["abc"])
    # chars a,b,c -> ids 2,3,4 (all < 60): no "new" predicted tokens at that threshold.
    _, n_tokens, n_new = forward_nll(model, ids, mask, base_vocab_size=60)
    assert n_tokens == 3
    assert n_new == 0
    # lower the threshold so ids 2,3,4 all count as "new".
    _, _, n_new_low = forward_nll(model, ids, mask, base_vocab_size=2)
    assert n_new_low == 3


def test_forward_nll_matches_manual_cross_entropy_single_sequence():
    model = _tiny_llama(vocab_size=64)
    tok = _FakeTokenizer("abcdefgh")
    ids, mask = encode_batch(tok, ["abcd"])
    nll, n_tokens, _ = forward_nll(model, ids, mask, base_vocab_size=60)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    manual_nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="sum"
    )
    assert n_tokens == 4  # bos+4 chars -> 4 predicted tokens
    assert float(nll) == pytest.approx(float(manual_nll), rel=1e-4)


# --------------------------------------------------------------------------
# score_corpus: batches over the whole corpus and sums raw totals
# --------------------------------------------------------------------------


def test_score_corpus_aggregates_across_batches():
    model = _tiny_llama(vocab_size=64)
    tok = _FakeTokenizer("abcdefgh")
    texts = ["ab", "cd", "ef", "gh", "abcd"]

    raw = score_corpus(model, tok, texts, base_vocab_size=60, batch_size=2)
    assert raw["n_sentences"] == 5
    # predicted tokens per sentence: len(bos+chars) - 1 = len(chars)
    assert raw["total_tokens"] == 2 + 2 + 2 + 2 + 4
    assert raw["total_nll"] > 0.0
    assert raw["wall_clock_s"] >= 0.0
    assert raw["total_new_tokens"] == 0


# --------------------------------------------------------------------------
# compute_metrics: pure arithmetic, exact expected values
# --------------------------------------------------------------------------


def test_compute_metrics_arithmetic():
    raw = {"total_tokens": 10, "total_nll": 20.0, "total_new_tokens": 4, "n_sentences": 5, "wall_clock_s": 1.5}
    metrics = compute_metrics(raw, total_chars=40)
    assert metrics["total_tokens"] == 10
    assert metrics["mean_tokens_per_sentence"] == pytest.approx(2.0)
    assert metrics["total_nll_nats"] == pytest.approx(20.0)
    assert metrics["bits_per_char"] == pytest.approx(20.0 / math.log(2) / 40)
    assert metrics["perplexity"] == pytest.approx(math.exp(20.0 / 10))
    assert metrics["new_token_share"] == pytest.approx(0.4)
    assert metrics["wall_clock_s"] == pytest.approx(1.5)


def test_compute_metrics_handles_zero_tokens_and_chars():
    raw = {"total_tokens": 0, "total_nll": 0.0, "total_new_tokens": 0, "n_sentences": 0, "wall_clock_s": 0.0}
    metrics = compute_metrics(raw, total_chars=0)
    assert math.isnan(metrics["mean_tokens_per_sentence"])
    assert math.isnan(metrics["bits_per_char"])
    assert math.isnan(metrics["perplexity"])
    assert math.isnan(metrics["new_token_share"])
