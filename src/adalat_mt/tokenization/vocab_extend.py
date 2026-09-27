"""P2 experiment: Chinese-LLaMA-style Hindi SentencePiece merge into TinyLlama's tokenizer.

    python -m adalat_mt.tokenization.vocab_extend --config configs/vocab_extend.yaml

Trains a Hindi BPE SentencePiece model on train-split Hindi text, merges its
pieces into the base (Llama-2) tokenizer used by TinyLlama-1.1B, resizes the
saved tokenizer, and measures the fertility reduction on train/test. Does
NOT download model weights (tokenizer files only); embedding resize/init is
exercised separately in tests on a tiny random model.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SPM_SPACE = "▁"  # SentencePiece's "▁" leading-space meta symbol.


def train_hindi_spm(sentences: list[str], vocab_size: int, out_prefix: Path) -> Path:
    """Train a BPE SentencePiece model on `sentences`; returns the `.model` path.

    Deterministic given a fixed `sentences` order: `shuffle_input_sentence=False`
    and a single thread avoid any run-to-run nondeterminism from the trainer.
    `character_coverage=1.0` so no Devanagari code point is dropped.
    """
    import sentencepiece as spm

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    input_path = out_prefix.with_suffix(".txt")
    input_path.write_text("\n".join(sentences), encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        shuffle_input_sentence=False,
        num_threads=1,
        hard_vocab_limit=False,
    )
    return out_prefix.with_suffix(".model")


def is_devanagari_piece(piece: str) -> bool:
    """True if the piece (ignoring the ``▁`` word-start marker) is non-empty and entirely Devanagari.

    The Hindi SPM also learns digit/punctuation pieces (``▁19``, ``.2022``); merging those would change how the base
    tokenizer splits English and numbers (Llama-2 deliberately uses one token per digit), so they are excluded.
    """
    core = piece.replace("\u2581", "")
    return bool(core) and all("\u0900" <= ch <= "\u097f" for ch in core)


def merge_spm(base_model: Path, extra_model: Path, out_model: Path, keep=is_devanagari_piece) -> list[str]:
    """Append `extra_model`'s NORMAL pieces that pass `keep` and are not already in `base_model`.

    Added pieces get `type=NORMAL, score=0.0` (Chinese-LLaMA-style merge). Returns the list of added piece
    strings (in the order they appear in `extra_model`). Pass ``keep=lambda p: True`` to merge everything.
    """
    from sentencepiece import sentencepiece_model_pb2 as pb2

    base = pb2.ModelProto()
    base.ParseFromString(base_model.read_bytes())
    extra = pb2.ModelProto()
    extra.ParseFromString(extra_model.read_bytes())

    base_pieces = {p.piece for p in base.pieces}
    added: list[str] = []
    normal_type = pb2.ModelProto.SentencePiece.Type.NORMAL
    for p in extra.pieces:
        if p.type != normal_type or p.piece in base_pieces or not keep(p.piece):
            continue
        new_piece = base.pieces.add()
        new_piece.piece = p.piece
        new_piece.score = 0.0
        new_piece.type = normal_type
        base_pieces.add(p.piece)
        added.append(p.piece)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_bytes(base.SerializeToString())
    return added


def build_extended_tokenizer(merged_model: Path, base_ident: str, out_dir: Path) -> Any:
    """Build a `LlamaTokenizer` from `merged_model`, copy special tokens from `base_ident`, save it."""
    from transformers import AutoTokenizer, LlamaTokenizer

    base_tok = AutoTokenizer.from_pretrained(base_ident, use_fast=False)
    new_tok = LlamaTokenizer(
        vocab_file=str(merged_model),
        unk_token=base_tok.unk_token,
        bos_token=base_tok.bos_token,
        eos_token=base_tok.eos_token,
        pad_token=base_tok.pad_token or base_tok.eos_token,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    new_tok.save_pretrained(str(out_dir))
    return new_tok


def _encode_surface(old_tok: Any, text: str) -> list[int]:
    """Ids `old_tok` assigns to `text`, with no BOS/EOS token mixed in."""
    ids = old_tok.encode(text, add_special_tokens=False)
    bos_id = getattr(old_tok, "bos_token_id", None)
    if ids and bos_id is not None and ids[0] == bos_id:
        ids = ids[1:]
    return ids


def new_token_init_map(old_tok: Any, new_tok: Any, added_pieces: list[str]) -> dict[int, list[int]]:
    """For each added SPM piece: its new-vocab id -> old-tokenizer ids of its surface string.

    `▁` in an SPM piece denotes a leading space, so `piece.replace("▁", " ")`
    recovers the surface text the piece represents. `old_tok.encode(...,
    add_special_tokens=False)` is used (and any leftover BOS id stripped) so
    the returned ids represent exactly that string, not "<bos> + string".
    """
    init_map: dict[int, list[int]] = {}
    for piece in added_pieces:
        text = piece.replace(_SPM_SPACE, " ")
        if not text.strip():
            continue
        new_id = new_tok.convert_tokens_to_ids(piece)
        old_ids = _encode_surface(old_tok, text)
        if old_ids:
            init_map[new_id] = old_ids
    return init_map


def init_new_embeddings(weight: Any, init_map: dict[int, list[int]]) -> None:
    """In place: `weight[new_id] = mean(weight[old_ids])` for each entry of `init_map`."""
    import torch

    with torch.no_grad():
        for new_id, old_ids in init_map.items():
            if not old_ids:
                continue
            weight[new_id] = weight[old_ids].mean(dim=0)


def resize_and_init(model: Any, old_tok: Any, new_tok: Any, added_pieces: list[str]) -> dict[str, Any]:
    """Resize `model`'s embeddings to `len(new_tok)` and init new rows from old sub-tokens.

    Initialises `get_input_embeddings().weight` and, if the model does not
    tie input/output embeddings, `get_output_embeddings().weight` too.
    Returns `{old_vocab_size, new_vocab_size, n_initialised, mean_sub_tokens_per_new_token}`.
    """
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    init_map = new_token_init_map(old_tok, new_tok, added_pieces)

    model.resize_token_embeddings(len(new_tok))
    init_new_embeddings(model.get_input_embeddings().weight.data, init_map)

    tied = bool(getattr(model.config, "tie_word_embeddings", True))
    output_emb = model.get_output_embeddings()
    if not tied and output_emb is not None:
        init_new_embeddings(output_emb.weight.data, init_map)

    mean_sub = statistics.mean(len(v) for v in init_map.values()) if init_map else 0.0
    return {
        "old_vocab_size": int(old_vocab_size),
        "new_vocab_size": len(new_tok),
        "n_initialised": len(init_map),
        "mean_sub_tokens_per_new_token": mean_sub,
    }


def _fertility(tok: Any, texts: list[str]) -> float:
    from .metrics import count_words

    tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
    words = sum(count_words(t) for t in texts)
    return tokens / words if words else float("nan")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_vocab_extend(config: dict, data_dir: Path, out_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the SPM-merge sweep over `config["vocab_sizes"]`; returns the results dict."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    base_ident = config["base_ident"]
    vocab_sizes = config.get("vocab_sizes", [2000, 4000, 8000, 16000])

    train_rows = _read_jsonl(data_dir / "train.jsonl")
    test_rows = _read_jsonl(data_dir / "test.jsonl")
    hi_train = [r["hi"] for r in train_rows]
    # Fixed seed: makes SPM training order-independent to reproduce, even
    # though BPE training itself is deterministic given a fixed sentence list.
    random.Random(config.get("seed", 13)).shuffle(hi_train)
    hi_test = [r["hi"] for r in test_rows]
    en_test = [r["en"] for r in test_rows]

    base_tok = AutoTokenizer.from_pretrained(base_ident, use_fast=False)
    base_model_path = Path(hf_hub_download(repo_id=base_ident, filename="tokenizer.model"))

    hi_fertility_test_base = _fertility(base_tok, hi_test)
    en_fertility_test_base = _fertility(base_tok, en_test)

    runs = []
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for vocab_size in vocab_sizes:
        prefix = artifacts_dir / f"hi_spm_{vocab_size}" / "hi_spm"
        hi_spm_path = train_hindi_spm(hi_train, vocab_size, prefix)
        merged_path = prefix.parent / "merged.model"
        added = merge_spm(base_model_path, hi_spm_path, merged_path)

        out_tok_dir = artifacts_dir / f"tinyllama-hi{vocab_size}"
        new_tok = build_extended_tokenizer(merged_path, base_ident, out_tok_dir)

        hi_fertility_train = _fertility(new_tok, hi_train)
        hi_fertility_test = _fertility(new_tok, hi_test)
        en_fertility_test = _fertility(new_tok, en_test)
        reduction = (
            100.0 * (hi_fertility_test_base - hi_fertility_test) / hi_fertility_test_base
            if hi_fertility_test_base
            else float("nan")
        )
        runs.append(
            {
                "hi_spm_vocab": vocab_size,
                "n_added": len(added),
                "new_vocab_size": len(new_tok),
                "hi_fertility_train": round(hi_fertility_train, 4),
                "hi_fertility_test": round(hi_fertility_test, 4),
                "en_fertility_test": round(en_fertility_test, 4),
                "hi_fertility_test_base": round(hi_fertility_test_base, 4),
                "en_fertility_test_base": round(en_fertility_test_base, 4),
                "hi_reduction_test_pct": round(reduction, 4),
            }
        )
        logger.info("vocab_size=%d n_added=%d hi_fertility_test=%.3f", vocab_size, len(added), hi_fertility_test)

    results = {
        "base": "tinyllama(llama-2)",
        "runs": runs,
        "embedding_init": "mean of old sub-token embeddings (input and lm_head)",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenization").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenization" / "vocab_extension.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_vocab_extension(runs, hi_fertility_test_base, out_dir / "figures" / "vocab_extension.png")
    return results


def _plot_vocab_extension(runs: list[dict[str, Any]], base_fertility: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_added = [r["n_added"] for r in runs]
    hi_test = [r["hi_fertility_test"] for r in runs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_added, hi_test, marker="o", label="HI test fertility (extended)")
    ax.axhline(base_fertility, linestyle="--", color="gray", label="HI test fertility (base TinyLlama)")
    ax.set_xlabel("number of added Hindi SPM pieces")
    ax.set_ylabel("tokens per word (fertility)")
    ax.set_title("Vocabulary extension: Hindi fertility vs added tokens")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vocab_extend.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/tokenizers"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_vocab_extend(config, args.data_dir, args.out_dir, args.artifacts_dir)


if __name__ == "__main__":
    main()
