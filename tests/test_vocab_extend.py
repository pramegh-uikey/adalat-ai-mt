"""Tests for `adalat_mt.tokenization.vocab_extend` (no Hub downloads)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adalat_mt.tokenization.vocab_extend import (
    init_new_embeddings,
    is_devanagari_piece,
    merge_spm,
    new_token_init_map,
    resize_and_init,
    train_hindi_spm,
)


# --------------------------------------------------------------------------
# init_new_embeddings
# --------------------------------------------------------------------------


def test_init_new_embeddings_new_row_is_mean_of_named_rows():
    weight = torch.tensor(
        [
            [1.0, 1.0],
            [3.0, 5.0],
            [0.0, 0.0],  # row 2 will become a new token, initialised from rows 0 and 1
            [9.0, 9.0],
        ]
    )
    init_new_embeddings(weight, {2: [0, 1]})
    assert torch.allclose(weight[2], torch.tensor([2.0, 3.0]))


def test_init_new_embeddings_old_rows_unchanged():
    weight = torch.tensor([[1.0, 1.0], [3.0, 5.0], [0.0, 0.0]])
    original_0 = weight[0].clone()
    original_1 = weight[1].clone()
    init_new_embeddings(weight, {2: [0, 1]})
    assert torch.equal(weight[0], original_0)
    assert torch.equal(weight[1], original_1)


def test_init_new_embeddings_multiple_entries():
    weight = torch.zeros(5, 3)
    weight[0] = torch.tensor([1.0, 2.0, 3.0])
    weight[1] = torch.tensor([3.0, 4.0, 5.0])
    init_new_embeddings(weight, {3: [0], 4: [0, 1]})
    assert torch.allclose(weight[3], weight[0])
    assert torch.allclose(weight[4], torch.tensor([2.0, 3.0, 4.0]))


# --------------------------------------------------------------------------
# resize_and_init, with fake tokenizers + a tiny random LlamaForCausalLM
# --------------------------------------------------------------------------


class _FakeOldTokenizer:
    """Old-vocab stub: one id per known character, `add_special_tokens` adds a BOS."""

    bos_token_id = 1

    def __init__(self, char_to_id: dict[str, int]):
        self._char_to_id = char_to_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._char_to_id[ch] for ch in text if ch in self._char_to_id]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids


class _FakeNewTokenizer:
    """New-vocab stub: base ids plus a few appended piece ids."""

    def __init__(self, base_size: int, added_pieces: list[str]):
        self._piece_to_id = {p: base_size + i for i, p in enumerate(added_pieces)}
        self._len = base_size + len(added_pieces)

    def __len__(self) -> int:
        return self._len

    def convert_tokens_to_ids(self, piece: str) -> int:
        return self._piece_to_id[piece]


def _tiny_llama(vocab_size: int, tie_word_embeddings: bool):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie_word_embeddings,
    )
    return LlamaForCausalLM(config)


def test_new_token_init_map_uses_surface_text_and_strips_bos():
    old_tok = _FakeOldTokenizer({"a": 5, "b": 6, "c": 7})
    new_tok = _FakeNewTokenizer(base_size=10, added_pieces=["▁ab", "c"])
    init_map = new_token_init_map(old_tok, new_tok, ["▁ab", "c"])
    # "▁ab" -> " ab" -> only known chars a,b are encodable (leading space unknown, dropped)
    assert init_map[10] == [5, 6]
    assert init_map[11] == [7]


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_resize_and_init_shapes_and_values(tie_word_embeddings):
    old_tok = _FakeOldTokenizer({"a": 5, "b": 6, "c": 7})
    added_pieces = ["▁ab", "c"]
    new_tok = _FakeNewTokenizer(base_size=64, added_pieces=added_pieces)

    model = _tiny_llama(vocab_size=64, tie_word_embeddings=tie_word_embeddings)
    input_weight_before = model.get_input_embeddings().weight.data.clone()

    stats = resize_and_init(model, old_tok, new_tok, added_pieces)

    assert stats["old_vocab_size"] == 64
    assert stats["new_vocab_size"] == 66
    assert stats["n_initialised"] == 2
    assert stats["mean_sub_tokens_per_new_token"] == pytest.approx(1.5)  # [5,6] and [7]

    input_weight_after = model.get_input_embeddings().weight.data
    assert input_weight_after.shape == (66, 16)
    # old rows (0..63) are untouched by resizing/init.
    assert torch.equal(input_weight_after[:64], input_weight_before)
    assert torch.allclose(input_weight_after[64], (input_weight_before[5] + input_weight_before[6]) / 2)
    assert torch.allclose(input_weight_after[65], input_weight_before[7])

    output_emb = model.get_output_embeddings()
    if tie_word_embeddings:
        # Tied: output embeddings are the same tensor as input embeddings.
        assert torch.equal(output_emb.weight.data, input_weight_after)
    else:
        assert output_emb.weight.data.shape == (66, 16)
        assert torch.allclose(
            output_emb.weight.data[64],
            (output_emb.weight.data[5] + output_emb.weight.data[6]) / 2,
        )


# --------------------------------------------------------------------------
# merge_spm, on two tiny SPM models trained from synthetic sentences
# --------------------------------------------------------------------------


def test_merge_spm_added_pieces_and_encodability(tmp_path: Path):
    base_sentences = ["aa bb aa bb", "aa cc aa cc", "bb cc bb cc"] * 5
    extra_sentences = ["dd ee dd ee", "ee ff ee ff", "dd ff dd ff"] * 5

    base_model = train_hindi_spm(base_sentences, vocab_size=24, out_prefix=tmp_path / "base" / "spm")
    extra_model = train_hindi_spm(extra_sentences, vocab_size=24, out_prefix=tmp_path / "extra" / "spm")

    out_model = tmp_path / "merged.model"
    added = merge_spm(base_model, extra_model, out_model, keep=lambda piece: True)

    assert added, "expected at least one piece unique to the extra model"

    from sentencepiece import sentencepiece_model_pb2 as pb2

    base_proto = pb2.ModelProto()
    base_proto.ParseFromString(base_model.read_bytes())
    base_piece_set = {p.piece for p in base_proto.pieces}

    extra_proto = pb2.ModelProto()
    extra_proto.ParseFromString(extra_model.read_bytes())
    normal_type = pb2.ModelProto.SentencePiece.Type.NORMAL
    expected_added = [p.piece for p in extra_proto.pieces if p.type == normal_type and p.piece not in base_piece_set]
    assert added == expected_added

    merged_proto = pb2.ModelProto()
    merged_proto.ParseFromString(out_model.read_bytes())
    assert len(merged_proto.pieces) == len(base_proto.pieces) + len(added)

    from sentencepiece import SentencePieceProcessor

    sp = SentencePieceProcessor(model_file=str(out_model))
    ids = sp.encode("dd ee", out_type=int)
    assert isinstance(ids, list) and len(ids) > 0
    # Round trip is at least non-empty and decodes back to something.
    assert isinstance(sp.decode(ids), str)


def test_is_devanagari_piece_excludes_digits_and_punctuation():
    assert is_devanagari_piece("\u2581न्यायालय")
    assert is_devanagari_piece("ों")
    assert not is_devanagari_piece("\u25812019")
    assert not is_devanagari_piece(".2022")
    assert not is_devanagari_piece("\u2581(सं")
    assert not is_devanagari_piece("\u2581")
