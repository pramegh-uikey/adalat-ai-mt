"""Markdown table generators for `REPORT.md`, reading only from `results/` and `configs/`.

Every function in `TABLES` takes no arguments (it reads its source file(s) at the default,
real-repo path) and returns a Markdown string ending with an italic ``*Source: ...*`` line
naming the file(s) it read. If a source file is missing, the function returns a one-line
``_(not available: <path> missing)_`` placeholder instead of raising, so `build_report` can run
before every phase has produced its outputs. Each function also accepts the same paths as
optional keyword arguments so it can be exercised on fixtures in tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import yaml

# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _fmt2(x: Any) -> str:
    """2 dp (fertility, ratios, chars/token, BLEU/chrF)."""
    return "—" if x is None else f"{float(x):.2f}"


def _fmt4(x: Any) -> str:
    """4 dp (COMET, costs)."""
    return "—" if x is None else f"{float(x):.4f}"


def _fmt_pct(x: Any, already_pct: bool = True) -> str:
    """1 dp percentage. `x` is already a 0-100 number unless `already_pct=False` (then a 0-1 fraction)."""
    if x is None:
        return "—"
    value = float(x) if already_pct else float(x) * 100
    return f"{value:.1f}%"


def _fmt_int(x: Any) -> str:
    """Integer with thousands separators (token counts, vocab sizes, param counts)."""
    if x is None:
        return "—"
    return f"{round(float(x)):,}"


def _fmt_plain_int(x: Any) -> str:
    """Plain integer, no separators (small counts: docs, segments)."""
    return "—" if x is None else str(int(x))


def _fmt_signed(x: Any, dp: int) -> str:
    return "—" if x is None else f"{float(x):+.{dp}f}"


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _source(*paths: Path) -> str:
    names = ", ".join(f"`{p}`" for p in paths)
    return f"*Source: {names}*"


def _not_available(path: Path) -> str:
    return f"_(not available: {path} missing)_"


def _first_missing(paths: list[Path]) -> Path | None:
    for p in paths:
        if not p.exists():
            return p
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def table_data(
    alignment_stats_path: Path = Path("results/data/alignment_stats.json"),
    split_config_path: Path = Path("configs/split.yaml"),
    alignment_report_path: Path = Path("results/data/alignment_report.md"),
) -> str:
    """Docs/pairs per split, drop/merge rates and audit precision lines."""
    missing = _first_missing([alignment_stats_path, split_config_path, alignment_report_path])
    if missing is not None:
        return _not_available(missing)

    stats = _load_json(alignment_stats_path)
    split_cfg = _load_yaml(split_config_path) or {}
    report_text = alignment_report_path.read_text(encoding="utf-8")

    docs = split_cfg.get("docs", {})
    pairs_per_split = stats.get("pairs_per_split", {})

    lines = ["| split | docs | pairs |", "|---|---|---|"]
    total_docs = 0
    total_pairs = 0
    for split_name in ("train", "dev", "test"):
        ids = docs.get(split_name, [])
        pairs = pairs_per_split.get(split_name, 0)
        total_docs += len(ids)
        total_pairs += pairs
        id_list = ", ".join(str(i) for i in ids)
        lines.append(f"| {split_name} | {id_list} ({len(ids)}) | {_fmt_int(pairs)} |")
    lines.append(f"| **total** | **{total_docs}** | **{_fmt_int(total_pairs)}** |")
    lines.append("")

    drop_rate_match = re.search(r"Drop rate:.*", report_text)
    merge_rate_match = re.search(r"Merge rate.*", report_text)
    if drop_rate_match:
        lines.append(f"- {drop_rate_match.group(0)}")
    if merge_rate_match:
        lines.append(f"- {merge_rate_match.group(0)}")
    # The alignment report has one audit section for the random sample, then one for the OCR-recovered docs.
    audit_labels = ["Alignment audit, random sample", "Alignment audit, OCR-recovered documents"]
    for k, m in enumerate(re.finditer(r"Judged by reading:.*", report_text)):
        label = audit_labels[k] if k < len(audit_labels) else "Alignment audit"
        lines.append(f"- {label}: {m.group(0)}")
    lines.append("")
    lines.append(_source(alignment_stats_path, split_config_path, alignment_report_path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# tokenizers
# --------------------------------------------------------------------------


def table_tokenizers(summary_path: Path = Path("results/tokenization/summary.json")) -> str:
    """Fertility/ratio/cost table, one row per tokenizer in the study."""
    if not summary_path.exists():
        return _not_available(summary_path)
    summary = _load_json(summary_path)

    lines = [
        "| tokenizer | family | HI fertility | EN fertility | HI/EN ratio | HI chars/token "
        "| HI tok/judgment | % test HI > 256 | cost/judgment USD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in summary.get("tokenizers", []):
        lines.append(
            "| {name} | {family} | {hi_f} | {en_f} | {ratio} | {cpt} | {tpj} | {pct} | {cost} |".format(
                name=t.get("name", ""),
                family=t.get("family", ""),
                hi_f=_fmt2(t.get("hi_fertility")),
                en_f=_fmt2(t.get("en_fertility")),
                ratio=_fmt2(t.get("hi_en_token_ratio")),
                cpt=_fmt2(t.get("hi_chars_per_token")),
                tpj=_fmt_int(t.get("hi_tokens_per_judgment")),
                pct=_fmt_pct(t.get("pct_test_hi_over_256")),
                cost=_fmt4(t.get("cost_per_judgment_usd")),
            )
        )
    lines.append("")
    lines.append(_source(summary_path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# vocab_extension
# --------------------------------------------------------------------------


def table_vocab_extension(path: Path = Path("results/tokenization/vocab_extension.json")) -> str:
    """Before/after Hindi fertility for each Hindi-SPM vocab size merged into the base tokenizer."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)

    lines = [f"Base tokenizer: `{data.get('base', '')}`", ""]
    lines.append(
        "| HI SPM vocab | tokens added | new vocab | HI fert (test) base -> extended | reduction % | EN fert (test) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in data.get("runs", []):
        lines.append(
            "| {vocab} | {added} | {new_vocab} | {base_f} -> {ext_f} | {reduction} | {en_f} |".format(
                vocab=_fmt_int(r.get("hi_spm_vocab")),
                added=_fmt_int(r.get("n_added")),
                new_vocab=_fmt_int(r.get("new_vocab_size")),
                base_f=_fmt2(r.get("hi_fertility_test_base")),
                ext_f=_fmt2(r.get("hi_fertility_test")),
                reduction=_fmt_pct(r.get("hi_reduction_test_pct")),
                en_f=_fmt2(r.get("en_fertility_test")),
            )
        )
    lines.append("")
    lines.append(_source(path))
    return "\n".join(lines)


def table_embedding_probe(path: Path = Path("results/tokenization/embedding_init_probe.json")) -> str:
    """Sequence length and untrained LM quality (bits/char) for base vs extended-vocabulary TinyLlama."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)
    lines = [
        f"{_fmt_plain_int(data.get('n_sentences'))} Hindi test sentences, `{data.get('base_ident', '')}`, no training.",
        "",
        "| condition | tokens / sentence | bits per char | token perplexity | share of new tokens |",
        "|---|---|---|---|---|",
    ]
    for name, c in data.get("conditions", {}).items():
        lines.append(
            f"| {name} | {_fmt2(c.get('mean_tokens_per_sentence'))} | {_fmt2(c.get('bits_per_char'))} | "
            f"{c.get('perplexity', float('nan')):,.1f} | {_fmt_pct(c.get('new_token_share'), already_pct=False)} |"
        )
    lines += ["", _source(path)]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def table_training(path: Path = Path("results/training/train_summary_it2_1b.json")) -> str:
    """Per-epoch dev history (best epoch bolded), param/time summary and the LoRA hyper-parameters."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)

    best_epoch = data.get("best_epoch")
    lines = ["| epoch | step | dev loss | dev chrF++ | dev BLEU |", "|---|---|---|---|---|"]
    for h in data.get("history", []):
        cells = [
            str(h.get("epoch", "—")),
            _fmt_plain_int(h.get("step")) if h.get("step") is not None else "—",
            _fmt4(h.get("dev_loss")),
            _fmt2(h.get("dev_chrf")),
            _fmt2(h.get("dev_bleu")),
        ]
        if h.get("epoch") == best_epoch:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    trainable = data.get("trainable_params")
    total = data.get("total_params")
    train_seconds = data.get("train_seconds")
    gpu = data.get("gpu", "—")
    seconds_str = _fmt2(train_seconds) if train_seconds is not None else "—"
    lines.append(
        f"Trainable / total params: {_fmt_int(trainable)} / {_fmt_int(total)}; "
        f"train time: {seconds_str}s on {gpu}."
    )
    lines.append("")

    cfg = data.get("config", {}) or {}
    lora_cfg = cfg.get("lora", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    batch = train_cfg.get("batch_size")
    grad_accum = train_cfg.get("grad_accum")
    effective_batch = batch * grad_accum if batch is not None and grad_accum is not None else None
    target_modules = ", ".join(lora_cfg.get("target_modules", []) or [])

    lines.append("| r | alpha | dropout | target modules | lr | epochs | effective batch | label smoothing | max length | seed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        "| {r} | {alpha} | {dropout} | {targets} | {lr} | {epochs} | {eff} | {label_smoothing} | {max_len} | {seed} |".format(
            r=lora_cfg.get("r", "—"),
            alpha=lora_cfg.get("alpha", "—"),
            dropout=lora_cfg.get("dropout", "—"),
            targets=target_modules or "—",
            lr=train_cfg.get("lr", "—"),
            epochs=train_cfg.get("epochs", "—"),
            eff=effective_batch if effective_batch is not None else "—",
            label_smoothing=train_cfg.get("label_smoothing", "—"),
            max_len=cfg.get("max_length", "—"),
            seed=cfg.get("seed", "—"),
        )
    )
    lines.append("")
    lines.append(_source(path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main_results
# --------------------------------------------------------------------------


def table_main_results(path: Path = Path("results/metrics.json")) -> str:
    """Corpus-level BLEU/spBLEU/chrF++/COMET, token usage and throughput per system."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)

    lines = [
        "| system | BLEU (13a) | spBLEU (flores200) | chrF++ | COMET | tokens in | tokens out | seconds | sents/s | device |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in data.get("systems", []):
        lines.append(
            "| {label} | {bleu} | {spbleu} | {chrf} | {comet} | {tin} | {tout} | {secs} | {sps} | {device} |".format(
                label=_escape_cell(s.get("label", s.get("name", ""))),
                bleu=_fmt2(s.get("bleu")),
                spbleu=_fmt2(s.get("spbleu")),
                chrf=_fmt2(s.get("chrf")),
                comet=_fmt4(s.get("comet")),
                tin=_fmt_int(s.get("tokens_in")),
                tout=_fmt_int(s.get("tokens_out")),
                secs=_fmt2(s.get("seconds")),
                sps=_fmt2(s.get("sents_per_sec")),
                device=s.get("device") or "—",
            )
        )
    lines.append("")

    signatures = data.get("signatures", {})
    if signatures:
        lines.append("```")
        for k, v in signatures.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    lines.append(_source(path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------


def table_significance(path: Path = Path("results/metrics.json")) -> str:
    """Paired-bootstrap significance rows (`results/metrics.json["significance"]`)."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)
    significance = data.get("significance", [])

    lines = ["| A | B | metric | A | B | Δ (B−A) | 95% CI | p (one-sided) |", "|---|---|---|---|---|---|---|---|"]
    for s in significance:
        dp = 4 if s.get("metric") == "comet" else 2
        fmt = _fmt2 if dp == 2 else _fmt4
        ci = s.get("ci95") or [None, None]
        ci_str = f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        p_value = s.get("p_value")
        # With n bootstrap resamples the smallest resolvable p-value is 1/n.
        n_samples = data.get("bootstrap", {}).get("n_samples", 1000)
        if p_value is None:
            p_str = "—"
        elif float(p_value) < 1 / n_samples:
            p_str = f"<{1 / n_samples:.3f}"
        else:
            p_str = f"{float(p_value):.3f}"
        lines.append(
            f"| {s.get('a', '')} | {s.get('b', '')} | {s.get('metric', '')} | {fmt(s.get('a_score'))} | "
            f"{fmt(s.get('b_score'))} | {_fmt_signed(s.get('delta'), dp)} | {ci_str} | {p_str} |"
        )
    lines.append("")
    lines.append(_source(path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cpu_latency
# --------------------------------------------------------------------------


def table_cpu_latency(pred_dir: Path = Path("results/predictions_cpu")) -> str:
    """Per-system CPU latency/throughput from `<pred_dir>/*.test.meta.json`."""
    pattern = pred_dir / "*.test.meta.json"
    metas = sorted(pred_dir.glob("*.test.meta.json")) if pred_dir.exists() else []
    if not metas:
        return _not_available(pattern)

    lines = ["| system | device | n | seconds | sents/s | tokens out/s |", "|---|---|---|---|---|---|"]
    for meta_path in metas:
        meta = _load_json(meta_path)
        lines.append(
            "| {system} | {device} | {n} | {secs} | {sps} | {tops} |".format(
                system=meta.get("label", meta.get("system", meta_path.stem)),
                device=meta.get("device", "—"),
                n=_fmt_plain_int(meta.get("n")),
                secs=_fmt2(meta.get("seconds")),
                sps=_fmt2(meta.get("sents_per_sec")),
                tops=_fmt2(meta.get("tokens_out_per_sec")),
            )
        )
    lines.append("")
    lines.append(_source(pattern))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# error_analysis
# --------------------------------------------------------------------------


def table_error_analysis(path: Path = Path("results/eval/error_analysis.json")) -> str:
    """Per-system fidelity/degeneration counts and per-comparison win/tie/loss (overall + by length)."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)

    lines = [
        "| system | numbers | citations | term hit-rate (any) | term hit-rate (ref) | repetition | omission | addition |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for system, stats in data.get("per_system", {}).items():
        numbers = stats.get("numbers", {})
        citations = stats.get("citations", {})
        terms = stats.get("terms", {})
        lines.append(
            "| {system} | {num} | {cit} | {any_r} | {ref_r} | {rep} | {om} | {add} |".format(
                system=system,
                num=_fmt_pct(numbers.get("rate"), already_pct=False),
                cit=_fmt_pct(citations.get("rate"), already_pct=False),
                any_r=_fmt_pct(terms.get("hit_any_rate"), already_pct=False),
                ref_r=_fmt_pct(terms.get("hit_ref_rate"), already_pct=False),
                rep=_fmt_plain_int(stats.get("repetition_segments")),
                om=_fmt_plain_int(stats.get("omission_segments")),
                add=_fmt_plain_int(stats.get("addition_segments")),
            )
        )
    lines.append("")

    for cmp_name, cmp_stats in data.get("comparisons", {}).items():
        a, b = cmp_name.rsplit("_vs_", 1)
        overall = cmp_stats.get("overall", {})
        lines.append(
            f"**{b} vs {a}** — overall: win={overall.get('win', 0)} tie={overall.get('tie', 0)} "
            f"loss={overall.get('loss', 0)}"
        )
        lines.append("")
        lines.append("| source length | win | tie | loss |")
        lines.append("|---|---|---|---|")
        for bucket, counts in cmp_stats.get("by_source_length", {}).items():
            lines.append(f"| {bucket} | {counts.get('win', 0)} | {counts.get('tie', 0)} | {counts.get('loss', 0)} |")
        lines.append("")

    lines.append(_source(path))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# term_categories
# --------------------------------------------------------------------------


def table_term_categories(path: Path = Path("results/eval/error_analysis.json")) -> str:
    """Per comparison, per glossary category: n and hit_ref rate for each of the two systems."""
    if not path.exists():
        return _not_available(path)
    data = _load_json(path)
    per_system = data.get("per_system", {})
    comparisons = data.get("comparisons", {})

    if not comparisons:
        return "_(no comparisons available)_\n\n" + _source(path)

    lines: list[str] = []
    for cmp_name in comparisons:
        a, b = cmp_name.rsplit("_vs_", 1)
        cats_a = (per_system.get(a, {}).get("terms", {}) or {}).get("by_category", {}) or {}
        cats_b = (per_system.get(b, {}).get("terms", {}) or {}).get("by_category", {}) or {}
        lines.append(f"**{b} vs {a}**")
        lines.append("")
        lines.append(f"| category | n | hit_ref {a} | hit_ref {b} |")
        lines.append("|---|---|---|---|")
        for cat in sorted(set(cats_a) | set(cats_b)):
            n = cats_a.get(cat, {}).get("n", cats_b.get(cat, {}).get("n"))
            rate_a = cats_a.get(cat, {}).get("hit_ref_rate")
            rate_b = cats_b.get(cat, {}).get("hit_ref_rate")
            lines.append(
                f"| {cat} | {_fmt_plain_int(n)} | {_fmt_pct(rate_a, already_pct=False)} | "
                f"{_fmt_pct(rate_b, already_pct=False)} |"
            )
        lines.append("")

    lines.append(_source(path))
    return "\n".join(lines)


def table_robustness(path: Path = Path("results/eval/error_analysis.json")) -> str:
    """Corpus chrF++ per system on subsets of segments with at least N source words."""
    if not path.exists():
        return _not_available(path)
    rows = _load_json(path).get("robustness", [])
    if not rows:
        return "_(no robustness data)_\n\n" + _source(path)
    systems = list(rows[0]["chrf"])
    lines = [
        "| min. source words | segments | " + " | ".join(systems) + " |",
        "|---|---|" + "---|" * len(systems),
    ]
    for r in rows:
        cells = " | ".join(_fmt2(r["chrf"].get(x)) for x in systems)
        lines.append(f"| {r['min_src_words']} | {_fmt_plain_int(r['n_segments'])} | {cells} |")
    lines += ["", _source(path)]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

TABLES: dict[str, Callable[[], str]] = {
    "data": table_data,
    "tokenizers": table_tokenizers,
    "vocab_extension": table_vocab_extension,
    "embedding_probe": table_embedding_probe,
    "robustness": table_robustness,
    "training": table_training,
    "training_200m": lambda: table_training(Path("results/training/train_summary_it2_200m.json")),
    "main_results": table_main_results,
    "significance": table_significance,
    "cpu_latency": table_cpu_latency,
    "error_analysis": table_error_analysis,
    "term_categories": table_term_categories,
}
