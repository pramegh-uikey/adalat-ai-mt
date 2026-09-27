"""Tests for `adalat_mt.inference.translate`.

Pure-helper tests run with no network / no model downloads. The single `@pytest.mark.slow` test
downloads/loads IndicTrans2-dist-200M and translates 2 sentences on CPU end-to-end.
"""

from __future__ import annotations

import json

import pytest
import torch

from adalat_mt.inference.translate import (
    count_input_tokens,
    count_output_tokens,
    load_split,
    pick_device,
    pick_dtype,
    restore_order,
    sort_by_length_indices,
    system_output_name,
)


def test_sort_by_length_indices_ascending() -> None:
    sentences = ["a b c d", "a", "a b", "a b c"]
    order = sort_by_length_indices(sentences)
    lengths = [len(sentences[i].split()) for i in order]
    assert lengths == sorted(lengths)
    assert set(order) == set(range(len(sentences)))


def test_restore_order_roundtrip() -> None:
    sentences = ["ddd dd", "a", "bb b", "ccc"]
    order = sort_by_length_indices(sentences)
    sorted_sents = [sentences[i] for i in order]
    restored = restore_order(sorted_sents, order)
    assert restored == sentences


def test_restore_order_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        restore_order(["a", "b"], [0])


def test_pick_device_auto_resolves_to_cpu_or_cuda() -> None:
    resolved = pick_device("auto")
    assert resolved in {"cpu", "cuda"}
    assert resolved == ("cuda" if torch.cuda.is_available() else "cpu")


def test_pick_device_passthrough() -> None:
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"


def test_pick_dtype() -> None:
    assert pick_dtype("cpu") == torch.float32
    assert pick_dtype("cuda") == torch.float16
    assert pick_dtype(torch.device("cpu")) == torch.float32


def test_count_input_tokens_with_fake_tensor() -> None:
    pad_id = 0
    input_ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 8, 0]])
    assert count_input_tokens(input_ids, pad_id) == 3 + 4


def test_count_output_tokens_excludes_special_ids() -> None:
    pad_id, bos_id, eos_id = 0, 1, 2
    output_ids = torch.tensor(
        [
            [bos_id, 10, 11, eos_id, pad_id],
            [bos_id, 10, 11, 12, eos_id],
        ]
    )
    special_ids = {pad_id, bos_id, eos_id}
    assert count_output_tokens(output_ids, pad_id, special_ids) == 2 + 3


def test_count_output_tokens_with_forced_language_token() -> None:
    pad_id, bos_id, eos_id, forced_id = 0, 1, 2, 99
    output_ids = torch.tensor([[bos_id, forced_id, 10, 11, eos_id]])
    special_ids = {pad_id, bos_id, eos_id, forced_id}
    assert count_output_tokens(output_ids, pad_id, special_ids) == 2


def test_system_output_name_no_limit() -> None:
    assert system_output_name("it2-1b", None) == "it2-1b"


def test_system_output_name_with_limit() -> None:
    assert system_output_name("it2-1b", 10) == "it2-1b.limit10"


def test_load_split(tmp_path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    rows = [
        {"doc_id": "1", "pair_id": "1-0", "en": "Hello.", "hi": "नमस्ते।", "align_score": 0.9, "align_type": "1-1"},
        {"doc_id": "1", "pair_id": "1-1", "en": "Bye.", "hi": "अलविदा।", "align_score": 0.8, "align_type": "1-1"},
    ]
    (data_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    loaded = load_split(data_dir, "test")
    assert loaded == rows


@pytest.mark.slow
def test_indictrans2_200m_end_to_end_cpu() -> None:
    """Real end-to-end translation of 2 sentences with IT2-dist-200M on CPU (no full test-set inference)."""
    from adalat_mt.inference.translate import IndicTrans2Translator

    translator = IndicTrans2Translator(
        model_name="ai4bharat/indictrans2-en-indic-dist-200M",
        device="cpu",
        batch_size=2,
        num_beams=1,
        max_length=64,
    )
    sentences = [
        "The court granted bail to the accused.",
        "The appeal was dismissed by the bench.",
    ]
    result = translator.translate(sentences)
    assert len(result.hyps) == 2
    assert all(isinstance(h, str) and len(h.strip()) > 0 for h in result.hyps)
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert result.seconds > 0
    print("IT2-200M outputs:", result.hyps)
