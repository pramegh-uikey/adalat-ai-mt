"""P2 experiment: embedding-initialisation probe on REAL TinyLlama weights.

    python -m adalat_mt.tokenization.embedding_probe --config configs/embedding_probe.yaml

No training happens here. This measures, on the same 64 Hindi test sentences,
what vocabulary extension + embedding resize buys for free (sequence-length
reduction) and what it costs (language-model quality) under two ways of
initialising the new embedding rows:

- ``base``: unmodified base tokenizer + unmodified base model.
- ``extended_mean_init``: extended tokenizer; new rows initialised as the
  mean of the base model's sub-token embeddings for the piece's surface text
  (reuses ``adalat_mt.tokenization.vocab_extend.new_token_init_map`` /
  ``resize_and_init``).
- ``extended_random_init``: extended tokenizer; new rows left at HF's default
  ``resize_token_embeddings`` init (normal-distribution draw), no mean init.

Each sentence is scored independently (BOS prepended, no special prompt) as a
causal-LM next-token-prediction task; NLL is summed in nats over predicted
tokens (padding excluded). Memory is tight (~7GB box), so only one model is
resident at a time: each condition loads the base model fresh from disk,
mutates it in place if needed, scores, and is torch.cuda-free/CPU-deleted +
garbage-collected before the next condition loads.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import yaml

from .metrics import count_chars
from .vocab_extend import resize_and_init

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_batch(tokenizer: Any, texts: list[str]) -> tuple[Any, Any]:
    """BOS-prepended, right-padded `(input_ids, attention_mask)` for `texts`.

    No special prompt; only the tokenizer's own BOS token is added. Padding
    uses ``tokenizer.pad_token_id`` (falls back to the eos id, matching how
    `vocab_extend.build_extended_tokenizer` sets ``pad_token``).
    """
    import torch

    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = [[bos_id] + tokenizer.encode(t, add_special_tokens=False) for t in texts]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = 1
    return input_ids, attention_mask


def forward_nll(model: Any, input_ids: Any, attention_mask: Any, base_vocab_size: int) -> tuple[float, int, int]:
    """One forward pass: `(total_nll_nats, n_predicted_tokens, n_new_predicted_tokens)`.

    Next-token prediction: position `t`'s logits predict the token at `t+1`.
    A predicted token counts only if both it and the position predicting it
    are real (non-pad) tokens, per `attention_mask`. Log-softmax is computed
    in float32 (the model may run in bfloat16) for numerically stable NLL.
    "New" means the predicted token id is `>= base_vocab_size`.
    """
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    valid_nll = token_nll[shift_mask]
    valid_labels = shift_labels[shift_mask]
    total_nll = float(valid_nll.sum().item())
    n_tokens = int(shift_mask.sum().item())
    n_new = int((valid_labels >= base_vocab_size).sum().item())
    return total_nll, n_tokens, n_new


def score_corpus(model: Any, tokenizer: Any, texts: list[str], base_vocab_size: int, batch_size: int = 8) -> dict[str, Any]:
    """Score all of `texts` in batches of `batch_size`; returns raw (unnormalised) totals.

    Returns ``{total_tokens, total_nll, total_new_tokens, n_sentences, wall_clock_s}``.
    Wall-clock covers only the forward passes (encoding/padding excluded).
    """
    model.eval()
    total_tokens = 0
    total_nll = 0.0
    total_new = 0
    wall_clock = 0.0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        input_ids, attention_mask = encode_batch(tokenizer, batch)
        t0 = time.perf_counter()
        nll, n_tok, n_new = forward_nll(model, input_ids, attention_mask, base_vocab_size)
        wall_clock += time.perf_counter() - t0
        total_tokens += n_tok
        total_nll += nll
        total_new += n_new
    return {
        "total_tokens": total_tokens,
        "total_nll": total_nll,
        "total_new_tokens": total_new,
        "n_sentences": len(texts),
        "wall_clock_s": wall_clock,
    }


def compute_metrics(raw: dict[str, Any], total_chars: int) -> dict[str, Any]:
    """Pure arithmetic: turn `score_corpus`'s raw totals into reportable metrics.

    `total_chars` is the sum of `count_chars` (non-whitespace code points)
    over the *original* sentences, independent of how they were tokenized,
    so bits-per-char is comparable across conditions with different vocabs.
    """
    total_tokens = raw["total_tokens"]
    total_nll = raw["total_nll"]
    n_sentences = raw["n_sentences"]
    return {
        "total_tokens": total_tokens,
        "mean_tokens_per_sentence": total_tokens / n_sentences if n_sentences else float("nan"),
        "total_nll_nats": total_nll,
        "bits_per_char": total_nll / math.log(2) / total_chars if total_chars else float("nan"),
        "perplexity": math.exp(total_nll / total_tokens) if total_tokens else float("nan"),
        "new_token_share": raw["total_new_tokens"] / total_tokens if total_tokens else float("nan"),
        "wall_clock_s": raw["wall_clock_s"],
    }


def _added_pieces(base_tok: Any, new_tok: Any) -> list[str]:
    """Pieces of `new_tok`'s vocab with id >= `len(base_tok)` (the added pieces)."""
    base_size = len(base_tok)
    return [new_tok.convert_ids_to_tokens(i) for i in range(base_size, len(new_tok))]


def _load_base_model(base_ident: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(base_ident, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)


def run_condition_base(base_ident: str, hi_sentences: list[str], base_vocab_size: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_ident, use_fast=False)
    model = _load_base_model(base_ident)
    raw = score_corpus(model, tok, hi_sentences, base_vocab_size)
    del model
    gc.collect()
    return raw


def run_condition_extended_mean_init(
    base_ident: str, extended_tokenizer: str, hi_sentences: list[str], base_vocab_size: int
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(base_ident, use_fast=False)
    new_tok = AutoTokenizer.from_pretrained(extended_tokenizer, use_fast=False)
    model = _load_base_model(base_ident)
    added_pieces = _added_pieces(base_tok, new_tok)
    resize_and_init(model, base_tok, new_tok, added_pieces)
    raw = score_corpus(model, new_tok, hi_sentences, base_vocab_size)
    del model
    gc.collect()
    return raw


def run_condition_extended_random_init(
    base_ident: str, extended_tokenizer: str, hi_sentences: list[str], base_vocab_size: int, seed: int
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    new_tok = AutoTokenizer.from_pretrained(extended_tokenizer, use_fast=False)
    model = _load_base_model(base_ident)
    torch.manual_seed(seed)
    model.resize_token_embeddings(len(new_tok))
    raw = score_corpus(model, new_tok, hi_sentences, base_vocab_size)
    del model
    gc.collect()
    return raw


def run_embedding_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Run all three conditions per `config` (see `configs/embedding_probe.yaml`); returns the results dict."""
    from transformers import AutoTokenizer

    base_ident = config["base_ident"]
    extended_tokenizer = config["extended_tokenizer"]
    data_path = Path(config["data"])
    n_sentences = int(config["n_sentences"])
    seed = int(config.get("seed", 13))

    rows = _read_jsonl(data_path)[:n_sentences]
    hi_sentences = [r["hi"] for r in rows]
    total_chars = sum(count_chars(t) for t in hi_sentences)

    base_tok = AutoTokenizer.from_pretrained(base_ident, use_fast=False)
    base_vocab_size = len(base_tok)
    del base_tok
    gc.collect()

    conditions: dict[str, Any] = {}

    logger.info("condition=base")
    raw = run_condition_base(base_ident, hi_sentences, base_vocab_size)
    conditions["base"] = compute_metrics(raw, total_chars)

    logger.info("condition=extended_mean_init")
    raw = run_condition_extended_mean_init(base_ident, extended_tokenizer, hi_sentences, base_vocab_size)
    conditions["extended_mean_init"] = compute_metrics(raw, total_chars)

    logger.info("condition=extended_random_init")
    raw = run_condition_extended_random_init(base_ident, extended_tokenizer, hi_sentences, base_vocab_size, seed)
    conditions["extended_random_init"] = compute_metrics(raw, total_chars)

    return {
        "base_ident": base_ident,
        "extended_tokenizer": extended_tokenizer,
        "n_sentences": len(hi_sentences),
        "conditions": conditions,
        "notes": "no training; new rows are untrained apart from initialisation",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/embedding_probe.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = run_embedding_probe(config)

    out_path = Path(config["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
