"""Tests for `adalat_mt.reporting` (marker substitution + individual table functions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from adalat_mt.reporting import tables as T
from adalat_mt.reporting.build_report import build_report, render_markers

# --------------------------------------------------------------------------
# render_markers / build_report
# --------------------------------------------------------------------------


def test_render_markers_replaces_only_the_marked_block():
    text = (
        "# Title\n\nSome prose.\n\n"
        "<!-- BEGIN:foo -->\nold content\n<!-- END:foo -->\n\n"
        "More prose that must survive untouched.\n"
    )
    out = render_markers(text, {"foo": lambda: "NEW TABLE"})
    assert "Some prose." in out
    assert "More prose that must survive untouched." in out
    assert "old content" not in out
    assert "<!-- BEGIN:foo -->\nNEW TABLE\n<!-- END:foo -->" in out


def test_render_markers_handles_multiple_distinct_markers():
    text = "<!-- BEGIN:a -->\nx\n<!-- END:a -->\n<!-- BEGIN:b -->\ny\n<!-- END:b -->\n"
    out = render_markers(text, {"a": lambda: "A!", "b": lambda: "B!"})
    assert "A!" in out and "B!" in out
    assert "x" not in out and "y" not in out


def test_render_markers_unknown_name_raises():
    text = "<!-- BEGIN:mystery -->\nold\n<!-- END:mystery -->\n"
    with pytest.raises(ValueError):
        render_markers(text, {"foo": lambda: "irrelevant"})


def test_render_markers_is_idempotent():
    text = "<!-- BEGIN:foo -->\nold\n<!-- END:foo -->\n"
    calls = {"n": 0}

    def render():
        calls["n"] += 1
        return "STABLE CONTENT"

    once = render_markers(text, {"foo": render})
    twice = render_markers(once, {"foo": render})
    assert once == twice


def test_build_report_writes_file_in_place(tmp_path: Path):
    report_path = tmp_path / "REPORT.md"
    report_path.write_text(
        "# Report\n\nintro\n\n<!-- BEGIN:foo -->\nold\n<!-- END:foo -->\n\noutro\n", encoding="utf-8"
    )
    result = build_report(report_path, tables={"foo": lambda: "rendered"})
    assert "rendered" in result
    assert "old" not in result
    assert result == report_path.read_text(encoding="utf-8")
    assert "intro" in result and "outro" in result


def test_build_report_unknown_marker_raises_and_does_not_write(tmp_path: Path):
    report_path = tmp_path / "REPORT.md"
    original = "<!-- BEGIN:nope -->\nold\n<!-- END:nope -->\n"
    report_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        build_report(report_path, tables={})
    assert report_path.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------
# missing-source placeholders
# --------------------------------------------------------------------------


def test_all_tables_report_missing_source_as_placeholder(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    assert "not available" in T.table_tokenizers(missing)
    assert "not available" in T.table_vocab_extension(missing)
    assert "not available" in T.table_training(missing)
    assert "not available" in T.table_main_results(missing)
    assert "not available" in T.table_significance(missing)
    assert "not available" in T.table_error_analysis(missing)
    assert "not available" in T.table_term_categories(missing)
    assert "not available" in T.table_cpu_latency(tmp_path / "no_predictions_cpu_dir")
    assert "not available" in T.table_data(
        missing, tmp_path / "split.yaml", tmp_path / "alignment_report.md"
    )


# --------------------------------------------------------------------------
# table_data
# --------------------------------------------------------------------------


def test_table_data(tmp_path: Path):
    stats_path = tmp_path / "alignment_stats.json"
    stats_path.write_text(
        json.dumps({"pairs_per_split": {"train": 1149, "dev": 162, "test": 120}}), encoding="utf-8"
    )
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        yaml.safe_dump({"docs": {"train": ["1", "2"], "dev": ["3"], "test": ["4"]}}), encoding="utf-8"
    )
    report_path = tmp_path / "alignment_report.md"
    report_path.write_text(
        "## Drop / merge rates\n\n"
        "- Drop rate: 0.0076 (by reason: {'low_score': 7})\n"
        "- Merge rate (non 1-1 pairs): 0.0513 (by type: {'1-1': 1368})\n\n"
        "## Audit\n\nJudged by reading: 20/20; precision (correct) = 19/20 = 0.950\n",
        encoding="utf-8",
    )

    out = T.table_data(stats_path, split_path, report_path)
    assert "| train | 1, 2 (2) | 1,149 |" in out
    assert "| dev | 3 (1) | 162 |" in out
    assert "| test | 4 (1) | 120 |" in out
    assert "**total**" in out and "**4**" in out and "**1,431**" in out
    assert "Drop rate: 0.0076" in out
    assert "Merge rate (non 1-1 pairs): 0.0513" in out
    assert "Judged by reading: 20/20; precision (correct) = 19/20 = 0.950" in out
    assert "*Source:" in out
    assert str(stats_path) in out


# --------------------------------------------------------------------------
# table_tokenizers
# --------------------------------------------------------------------------


def test_table_tokenizers(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tokenizers": [
                    {
                        "name": "indictrans2",
                        "family": "IndicTrans2",
                        "hi_fertility": 1.2805,
                        "en_fertility": 1.3441,
                        "hi_en_token_ratio": 1.0177,
                        "hi_chars_per_token": 3.3531,
                        "hi_tokens_per_judgment": 2079.5,
                        "pct_test_hi_over_256": 0.0,
                        "cost_per_judgment_usd": None,
                    },
                    {
                        "name": "cl100k_base",
                        "family": "GPT-4",
                        "hi_fertility": 5.0922,
                        "en_fertility": 1.3317,
                        "hi_en_token_ratio": 4.0851,
                        "hi_chars_per_token": 0.8432,
                        "hi_tokens_per_judgment": 8242.1667,
                        "pct_test_hi_over_256": 15.8333,
                        "cost_per_judgment_usd": 0.0873,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out = T.table_tokenizers(summary_path)
    assert "| indictrans2 | IndicTrans2 | 1.28 | 1.34 | 1.02 | 3.35 | 2,080 | 0.0% | —" in out
    assert "| cl100k_base | GPT-4 | 5.09 | 1.33 | 4.09 | 0.84 | 8,242 | 15.8% | 0.0873 |" in out
    assert out.strip().endswith(f"*Source: `{summary_path}`*")


# --------------------------------------------------------------------------
# table_vocab_extension
# --------------------------------------------------------------------------


def test_table_vocab_extension(tmp_path: Path):
    path = tmp_path / "vocab_extension.json"
    path.write_text(
        json.dumps(
            {
                "base": "tinyllama(llama-2)",
                "runs": [
                    {
                        "hi_spm_vocab": 8000,
                        "n_added": 6475,
                        "new_vocab_size": 38475,
                        "hi_fertility_test_base": 5.4482,
                        "hi_fertility_test": 1.4003,
                        "hi_reduction_test_pct": 74.2979,
                        "en_fertility_test": 1.4952,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = T.table_vocab_extension(path)
    assert "Base tokenizer: `tinyllama(llama-2)`" in out
    assert "| 8,000 | 6,475 | 38,475 | 5.45 -> 1.40 | 74.3% | 1.50 |" in out


# --------------------------------------------------------------------------
# table_training
# --------------------------------------------------------------------------


def test_table_training(tmp_path: Path):
    path = tmp_path / "train_summary_it2_1b.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    "max_length": 256,
                    "seed": 13,
                    "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "v_proj"]},
                    "train": {"lr": 3e-4, "epochs": 6, "batch_size": 8, "grad_accum": 2, "label_smoothing": 0.1},
                },
                "trainable_params": 1_000_000,
                "total_params": 100_000_000,
                "best_epoch": 2,
                "train_seconds": 543.21,
                "gpu": "Tesla T4",
                "history": [
                    {"epoch": 0, "dev_loss": 2.0, "dev_chrf": 10.0, "dev_bleu": 5.0},
                    {"epoch": 1, "step": 10, "dev_loss": 1.5, "dev_chrf": 20.0, "dev_bleu": 10.0},
                    {"epoch": 2, "step": 20, "dev_loss": 1.2, "dev_chrf": 30.0, "dev_bleu": 15.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = T.table_training(path)
    assert "| **2** | **20** | **1.2000** | **30.00** | **15.00** |" in out
    assert "| 1 | 10 | 1.5000 | 20.00 | 10.00 |" in out
    assert "Trainable / total params: 1,000,000 / 100,000,000" in out
    assert "train time: 543.21s on Tesla T4" in out
    assert "| 16 | 32 | 0.1 | q_proj, v_proj | 0.0003 | 6 | 16 | 0.1 | 256 | 13 |" in out


# --------------------------------------------------------------------------
# table_main_results / table_significance
# --------------------------------------------------------------------------


def _metrics_fixture() -> dict:
    return {
        "signatures": {"bleu": "nrefs:1|case:mixed"},
        "systems": [
            {
                "name": "it2-1b",
                "label": "IndicTrans2-1B zero-shot",
                "bleu": 25.5,
                "spbleu": 30.1,
                "chrf": 55.25,
                "comet": None,
                "tokens_in": 12345,
                "tokens_out": 23456,
                "seconds": 12.345,
                "sents_per_sec": 8.5,
                "device": "cpu",
            },
            {
                "name": "it2-1b-lora",
                "label": "IndicTrans2-1B + LoRA",
                "bleu": 30.2,
                "spbleu": 35.6,
                "chrf": 60.1,
                "comet": 0.812345,
                "tokens_in": 12345,
                "tokens_out": 23456,
                "seconds": 12.0,
                "sents_per_sec": 8.6,
                "device": "cuda",
            },
        ],
        "significance": [
            {
                "a": "it2-1b",
                "b": "it2-1b-lora",
                "metric": "bleu",
                "a_score": 25.5,
                "b_score": 30.2,
                "delta": 4.7,
                "ci95": [1.2, 8.1],
                "p_value": 0.012,
            },
            {
                "a": "it2-1b",
                "b": "it2-1b-lora",
                "metric": "comet",
                "a_score": 0.75,
                "b_score": 0.8123,
                "delta": 0.0623,
                "ci95": [0.01, 0.11],
                "p_value": 0.03,
            },
        ],
    }


def test_table_main_results(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics_fixture()), encoding="utf-8")
    out = T.table_main_results(path)
    assert "| IndicTrans2-1B zero-shot | 25.50 | 30.10 | 55.25 | — | 12,345 | 23,456 | 12.35 | 8.50 | cpu |" in out
    assert "| IndicTrans2-1B + LoRA | 30.20 | 35.60 | 60.10 | 0.8123 | 12,345 | 23,456 | 12.00 | 8.60 | cuda |" in out
    assert "bleu: nrefs:1|case:mixed" in out


def test_table_significance(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics_fixture()), encoding="utf-8")
    out = T.table_significance(path)
    assert "| it2-1b | it2-1b-lora | bleu | 25.50 | 30.20 | +4.70 | [1.20, 8.10] | 0.012 |" in out
    assert "| it2-1b | it2-1b-lora | comet | 0.7500 | 0.8123 | +0.0623 | [0.0100, 0.1100] | 0.030 |" in out


# --------------------------------------------------------------------------
# table_cpu_latency
# --------------------------------------------------------------------------


def test_table_cpu_latency(tmp_path: Path):
    pred_dir = tmp_path / "predictions_cpu"
    pred_dir.mkdir()
    (pred_dir / "it2-200m.test.meta.json").write_text(
        json.dumps(
            {
                "system": "it2-200m",
                "label": "IndicTrans2-dist-200M zero-shot",
                "device": "cpu",
                "n": 120,
                "seconds": 45.6789,
                "sents_per_sec": 2.6296,
                "tokens_out_per_sec": 30.1,
            }
        ),
        encoding="utf-8",
    )
    out = T.table_cpu_latency(pred_dir)
    assert "| IndicTrans2-dist-200M zero-shot | cpu | 120 | 45.68 | 2.63 | 30.10 |" in out


# --------------------------------------------------------------------------
# table_error_analysis / table_term_categories
# --------------------------------------------------------------------------


def _error_analysis_fixture() -> dict:
    return {
        "per_system": {
            "baseline": {
                "numbers": {"rate": 0.8},
                "citations": {"rate": 1.0},
                "terms": {
                    "hit_any_rate": 0.9,
                    "hit_ref_rate": 0.7,
                    "by_category": {
                        "legal-term": {"n": 10, "hit_any_rate": 0.9, "hit_ref_rate": 0.6},
                        "named-entity": {"n": 4, "hit_any_rate": 1.0, "hit_ref_rate": 0.75},
                    },
                },
                "repetition_segments": 1,
                "omission_segments": 2,
                "addition_segments": 0,
            },
            "adapted": {
                "numbers": {"rate": 0.95},
                "citations": {"rate": 1.0},
                "terms": {
                    "hit_any_rate": 0.95,
                    "hit_ref_rate": 0.85,
                    "by_category": {
                        "legal-term": {"n": 10, "hit_any_rate": 0.95, "hit_ref_rate": 0.8},
                        "named-entity": {"n": 4, "hit_any_rate": 1.0, "hit_ref_rate": 1.0},
                    },
                },
                "repetition_segments": 0,
                "omission_segments": 1,
                "addition_segments": 0,
            },
        },
        "comparisons": {
            "baseline_vs_adapted": {
                "overall": {"win": 8, "tie": 5, "loss": 1},
                "by_source_length": {
                    "<=20": {"win": 3, "tie": 2, "loss": 0},
                    "21-40": {"win": 5, "tie": 3, "loss": 1},
                },
            }
        },
    }


def test_table_error_analysis(tmp_path: Path):
    path = tmp_path / "error_analysis.json"
    path.write_text(json.dumps(_error_analysis_fixture()), encoding="utf-8")
    out = T.table_error_analysis(path)
    assert "| baseline | 80.0% | 100.0% | 90.0% | 70.0% | 1 | 2 | 0 |" in out
    assert "| adapted | 95.0% | 100.0% | 95.0% | 85.0% | 0 | 1 | 0 |" in out
    assert "**adapted vs baseline** — overall: win=8 tie=5 loss=1" in out
    assert "| <=20 | 3 | 2 | 0 |" in out


def test_table_term_categories(tmp_path: Path):
    path = tmp_path / "error_analysis.json"
    path.write_text(json.dumps(_error_analysis_fixture()), encoding="utf-8")
    out = T.table_term_categories(path)
    assert "**adapted vs baseline**" in out
    assert "| legal-term | 10 | 60.0% | 80.0% |" in out
    assert "| named-entity | 4 | 75.0% | 100.0% |" in out


def test_table_term_categories_no_comparisons(tmp_path: Path):
    path = tmp_path / "error_analysis.json"
    path.write_text(json.dumps({"per_system": {}, "comparisons": {}}), encoding="utf-8")
    out = T.table_term_categories(path)
    assert "no comparisons available" in out


def test_table_embedding_probe_renders(tmp_path) -> None:
    import json

    from adalat_mt.reporting.tables import table_embedding_probe

    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"n_sentences": 2, "base_ident": "x", "conditions": {"base": {
        "mean_tokens_per_sentence": 10.0, "bits_per_char": 2.5, "perplexity": 3.7, "new_token_share": 0.0}}}))
    out = table_embedding_probe(p)
    assert "| base | 10.00 | 2.50 | 3.7 | 0.0% |" in out
    assert table_embedding_probe(tmp_path / "missing.json").startswith("_(not available")


def test_table_robustness_renders(tmp_path) -> None:
    import json

    from adalat_mt.reporting.tables import table_robustness

    p = tmp_path / "ea.json"
    p.write_text(json.dumps({"robustness": [{"min_src_words": 10, "n_segments": 3, "chrf": {"a": 50.0, "b": 51.234}}]}))
    out = table_robustness(p)
    assert "| 10 | 3 | 50.00 | 51.23 |" in out


def test_render_markers_rejects_duplicate_names() -> None:
    import pytest

    from adalat_mt.reporting.build_report import render_markers

    text = "<!-- BEGIN:a -->\n<!-- END:a -->\n<!-- BEGIN:a -->\n<!-- END:a -->"
    with pytest.raises(ValueError, match="duplicate"):
        render_markers(text, {"a": lambda: "x"})
