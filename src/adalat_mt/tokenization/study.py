"""CLI: tokenizer efficiency study across `configs/tokenizers.yaml`.

    python -m adalat_mt.tokenization.study --config configs/tokenizers.yaml

Computes, for every tokenizer in the config, fertility/parity/cost metrics on
the train and test splits produced by Phase 1 (`data/processed/*.jsonl`), and
writes `results/tokenization/summary.json`, `results/tokenization/cost_table.md`
and two figures under `results/figures/`.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import yaml

from ..data.segment import segment_document
from ..data.text_clean import read_text
from ..data.sources import load_data_config, source_path
from .figures import plot_fertility, plot_hindi_example
from .metrics import corpus_stats, count_words, judgment_cost_usd, pct_over
from .registry import TokenizerSpec, load_counter

logger = logging.getLogger(__name__)

#: The six tokenizers shown in the Hindi-tokenization illustration figure.
EXAMPLE_TOKENIZER_NAMES = [
    "o200k_base",
    "llama-3.2",
    "tinyllama(llama-2)",
    "sarvam-1",
    "indictrans2",
    "nllb-200",
]

DEFINITIONS = {
    "en_fertility": "English tokens / English whitespace words on the train split pairs.",
    "hi_fertility": "Hindi tokens / Hindi whitespace words on the train split pairs.",
    "en_chars_per_token": "Non-whitespace English characters / English tokens on the train split.",
    "hi_chars_per_token": "Non-whitespace Hindi characters / Hindi tokens on the train split.",
    "hi_en_token_ratio": "Total Hindi tokens / total English tokens over the same train-split pairs.",
    "hi_en_ratio_median_sentence": "Median, over train pairs, of (Hindi tokens / English tokens) for that pair.",
    "hi_unk_rate": "Fraction of Hindi train tokens equal to the tokenizer's unknown-token id.",
    "en_unk_rate": "Fraction of English train tokens equal to the tokenizer's unknown-token id.",
    "vocab_size": "Tokenizer vocabulary size (target/Hindi side for spm_pair tokenizers).",
    "pct_test_en_over_L": "Percentage of test-split English sentences with more than L tokens.",
    "pct_test_hi_over_L": "Percentage of test-split Hindi sentences with more than L tokens.",
    "pct_test_hi_over_model_max": "Percentage of test-split Hindi sentences exceeding the model's max sequence length.",
    "en_tokens_per_judgment": "Mean English tokens over the full segmented text of each train-split judgment.",
    "hi_tokens_per_judgment": "Mean Hindi tokens over the full segmented text of each train-split judgment.",
    "cost_per_judgment_usd": "Estimated API cost (input=EN tokens, output=HI tokens) to translate one judgment; no prompt overhead.",
    "cost_per_1000_judgments_usd": "cost_per_judgment_usd * 1000.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _load_specs(config: dict) -> list[TokenizerSpec]:
    specs = []
    for entry in config["tokenizers"]:
        specs.append(
            TokenizerSpec(
                name=entry["name"],
                kind=entry["kind"],
                ident=entry["ident"],
                family=entry.get("family", entry["name"]),
                price=entry.get("price"),
                model_max_len=entry.get("model_max_len"),
                src_file=entry.get("src_file"),
                tgt_file=entry.get("tgt_file"),
            )
        )
    return specs


def _judgment_token_counts(counter, doc_ids: list[str], dataset_dir: Path) -> tuple[list[int], list[int]]:
    """Per-document (EN tokens, HI tokens) totals over the FULL segmented text.

    Hindi text of the OCR-recovered documents is read from ``data/recovered`` (see ``configs/data.yaml``).
    """
    data_cfg = {**load_data_config(), "dataset_dir": str(dataset_dir)}
    en_totals: list[int] = []
    hi_totals: list[int] = []
    for doc_id in doc_ids:
        en_path = source_path(doc_id, "en", data_cfg)
        hi_path = source_path(doc_id, "hi", data_cfg)
        en_sentences = segment_document(read_text(en_path), "en")
        hi_sentences = segment_document(read_text(hi_path), "hi")
        en_totals.append(sum(counter.count(s, "en") for s in en_sentences))
        hi_totals.append(sum(counter.count(s, "hi") for s in hi_sentences))
    return en_totals, hi_totals


def compute_tokenizer_metrics(
    spec: TokenizerSpec,
    counter,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_doc_ids: list[str],
    dataset_dir: Path,
    length_limits: list[int],
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """All summary.json fields for one tokenizer."""
    en_texts = [r["en"] for r in train_rows]
    hi_texts = [r["hi"] for r in train_rows]
    en_stats = corpus_stats(counter, en_texts, "en")
    hi_stats = corpus_stats(counter, hi_texts, "hi")

    per_pair_ratios = []
    for r in train_rows:
        en_n = counter.count(r["en"], "en")
        hi_n = counter.count(r["hi"], "hi")
        if en_n:
            per_pair_ratios.append(hi_n / en_n)
    median_ratio = statistics.median(per_pair_ratios) if per_pair_ratios else float("nan")

    metrics: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "en_fertility": en_stats.fertility,
        "hi_fertility": hi_stats.fertility,
        "en_chars_per_token": en_stats.chars_per_token,
        "hi_chars_per_token": hi_stats.chars_per_token,
        "hi_en_token_ratio": hi_stats.tokens / en_stats.tokens if en_stats.tokens else float("nan"),
        "hi_en_ratio_median_sentence": median_ratio,
        "hi_unk_rate": hi_stats.unk_rate,
        "en_unk_rate": en_stats.unk_rate,
        "vocab_size": getattr(counter, "vocab_size", None),
    }

    en_test_lengths = [counter.count(r["en"], "en") for r in test_rows]
    hi_test_lengths = [counter.count(r["hi"], "hi") for r in test_rows]
    for limit in length_limits:
        metrics[f"pct_test_en_over_{limit}"] = pct_over(en_test_lengths, limit)
        metrics[f"pct_test_hi_over_{limit}"] = pct_over(hi_test_lengths, limit)
    if spec.model_max_len is not None:
        metrics["pct_test_hi_over_model_max"] = pct_over(hi_test_lengths, spec.model_max_len)

    en_judg, hi_judg = _judgment_token_counts(counter, train_doc_ids, dataset_dir)
    en_per_judgment = statistics.mean(en_judg) if en_judg else float("nan")
    hi_per_judgment = statistics.mean(hi_judg) if hi_judg else float("nan")
    metrics["en_tokens_per_judgment"] = en_per_judgment
    metrics["hi_tokens_per_judgment"] = hi_per_judgment

    if spec.price and spec.price in prices:
        price = prices[spec.price]
        cost = judgment_cost_usd(en_per_judgment, hi_per_judgment, price)
        metrics["cost_per_judgment_usd"] = cost["total_usd"]
        metrics["cost_per_1000_judgments_usd"] = cost["total_usd"] * 1000

    return {k: _round(v) for k, v in metrics.items()}


def _pick_example_sentence(hi_texts: list[str]) -> str | None:
    """A short (10-14 word) Hindi sentence containing a common legal term."""
    keywords = ("अपीलार्थी", "न्यायालय")
    candidates = [
        s for s in hi_texts if any(k in s for k in keywords) and 10 <= count_words(s) <= 14
    ]
    if candidates:
        return min(candidates, key=count_words)
    # Relax the word-count window before giving up.
    candidates = [s for s in hi_texts if any(k in s for k in keywords) and 6 <= count_words(s) <= 20]
    if candidates:
        return min(candidates, key=lambda s: abs(count_words(s) - 12))
    return max(hi_texts, key=count_words, default=None)


def write_cost_table(summary: dict[str, Any], out_path: Path) -> None:
    """Markdown cost/latency table from an already-built summary dict."""
    lines = [
        "| tokenizer | EN tok/judgment | HI tok/judgment | HI/EN premium | % test HI > 256 | cost/judgment (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for t in summary["tokenizers"]:
        cost = t.get("cost_per_judgment_usd")
        cost_str = f"{cost:.4f}" if cost is not None else "—"
        lines.append(
            "| {name} | {en:.2f} | {hi:.2f} | {ratio:.2f} | {pct:.1f} | {cost} |".format(
                name=t["name"],
                en=t["en_tokens_per_judgment"],
                hi=t["hi_tokens_per_judgment"],
                ratio=t["hi_en_token_ratio"],
                pct=t.get("pct_test_hi_over_256", float("nan")),
                cost=cost_str,
            )
        )
    lines.append("")
    lines.append(
        "Prices are illustrative 2024 list prices (see `configs/tokenizers.yaml`), an ASSUMPTION not a verified "
        "live quote; costs ignore prompt/system-message overhead. Decoder latency scales with output length: "
        "HI tok/judgment = sequential decoding steps for an autoregressive LLM generating the Hindi translation."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    config_path: Path,
    split_config_path: Path,
    data_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    font_path: Path,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full tokenizer study and write results under `out_dir`. Returns the summary dict.

    `names`, if given, restricts the study to tokenizers whose `name` is in the list (smoke
    tests: e.g. ``["cl100k_base", "o200k_base"]``); raises `ValueError` for any unknown name.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_config = yaml.safe_load(split_config_path.read_text(encoding="utf-8"))
    length_limits = config.get("length_limits", [128, 256, 512])
    prices = config.get("prices", {})

    train_rows = _read_jsonl(data_dir / "train.jsonl")
    test_rows = _read_jsonl(data_dir / "test.jsonl")
    train_doc_ids = split_config.get("docs", {}).get("train", [])

    specs = _load_specs(config)
    if names:
        wanted = set(names)
        unknown = sorted(wanted - {s.name for s in specs})
        if unknown:
            raise ValueError(f"unknown tokenizer names: {unknown}")
        specs = [s for s in specs if s.name in wanted]

    tokenizer_metrics = []
    example_pieces: dict[str, list[str]] = {}
    example_counts: dict[str, int] = {}
    hi_train_texts = [r["hi"] for r in train_rows]
    example_sentence = _pick_example_sentence(hi_train_texts)

    for spec in specs:
        logger.info("loading tokenizer %s (%s)", spec.name, spec.kind)
        counter = load_counter(spec)
        tokenizer_metrics.append(
            compute_tokenizer_metrics(
                spec, counter, train_rows, test_rows, train_doc_ids, dataset_dir, length_limits, prices
            )
        )
        if example_sentence and spec.name in EXAMPLE_TOKENIZER_NAMES:
            example_pieces[spec.name] = counter.tokenize(example_sentence, "hi")
            # `ids()` is the definition used everywhere else in the study;
            # pin the figure's "(N tok)" label to it rather than trusting
            # `tokenize()` to always return one piece per id.
            example_counts[spec.name] = len(counter.ids(example_sentence, "hi"))

    tokenizer_metrics.sort(key=lambda m: m["hi_fertility"])

    summary = {
        "text": {
            "split": "train",
            "n_pairs": len(train_rows),
            "n_en_words": sum(count_words(r["en"]) for r in train_rows),
            "n_hi_words": sum(count_words(r["hi"]) for r in train_rows),
            "n_train_docs": len(train_doc_ids),
        },
        "definitions": DEFINITIONS,
        "tokenizers": tokenizer_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenization").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenization" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_cost_table(summary, out_dir / "tokenization" / "cost_table.md")

    plot_fertility(
        [t["name"] for t in tokenizer_metrics],
        [t["en_fertility"] for t in tokenizer_metrics],
        [t["hi_fertility"] for t in tokenizer_metrics],
        out_dir / "figures" / "fertility.png",
    )
    if example_sentence and example_pieces:
        ordered = {name: example_pieces[name] for name in EXAMPLE_TOKENIZER_NAMES if name in example_pieces}
        plot_hindi_example(
            example_sentence,
            ordered,
            font_path,
            out_dir / "figures" / "hindi_tokenization_example.png",
            token_counts=example_counts,
        )
    else:
        logger.warning("no example sentence/tokenizer pieces available; skipping hindi_tokenization_example.png")

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizers.yaml"))
    parser.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--font", type=Path, default=Path("assets/fonts/NotoSansDevanagari-Regular.ttf"))
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="restrict the study to these tokenizer names from --config (smoke tests, e.g. "
        "--names cl100k_base o200k_base)",
    )
    args = parser.parse_args()

    summary = run_study(
        args.config, args.split_config, args.data_dir, args.dataset_dir, args.out_dir, args.font, names=args.names
    )
    for t in summary["tokenizers"]:
        logger.info(
            "%-20s hi_fertility=%.3f en_fertility=%.3f hi_en_token_ratio=%.3f",
            t["name"],
            t["hi_fertility"],
            t["en_fertility"],
            t["hi_en_token_ratio"],
        )


if __name__ == "__main__":
    main()
