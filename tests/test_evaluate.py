"""End-to-end test for `adalat_mt.evaluation.evaluate` using fake prediction files (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from adalat_mt.evaluation.evaluate import run_evaluate

REFS = [
    "The court granted bail to the accused.",
    "The appeal was dismissed by the bench.",
    "The petitioner filed a writ petition.",
    "Notice was issued to the respondent.",
]
SRCS = [
    "The court granted bail to the accused.",
    "The appeal was dismissed by the bench.",
    "The petitioner filed a writ petition.",
    "Notice was issued to the respondent.",
]
SYS_A_HYPS = REFS  # perfect system
SYS_B_HYPS = [
    "Totally different words here now.",
    "Nothing at all like the reference.",
    "This does not match either.",
    "Random unrelated output text.",
]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_pred_dir(tmp_path: Path) -> Path:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_lines(pred_dir / "test.ref.hi", REFS)
    _write_lines(pred_dir / "test.src.en", SRCS)
    _write_lines(pred_dir / "sys-a.test.hi", SYS_A_HYPS)
    _write_lines(pred_dir / "sys-b.test.hi", SYS_B_HYPS)
    (pred_dir / "sys-a.test.meta.json").write_text(
        json.dumps(
            {
                "system": "sys-a",
                "label": "System A",
                "tokens_in": 40,
                "tokens_out": 38,
                "seconds": 2.0,
                "sents_per_sec": 2.0,
                "tokens_out_per_sec": 19.0,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    _write_lines_meta = pred_dir / "sys-b.test.meta.json"
    _write_lines_meta.write_text(
        json.dumps(
            {
                "system": "sys-b",
                "label": "System B",
                "tokens_in": 40,
                "tokens_out": 30,
                "seconds": 1.0,
                "sents_per_sec": 4.0,
                "tokens_out_per_sec": 30.0,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    return pred_dir


def _write_eval_config(tmp_path: Path, pred_dir: Path) -> Path:
    cfg = {
        "split": "test",
        "pred_dir": str(pred_dir),
        "comet_dir": str(tmp_path / "comet"),
        "baseline": "sys-a",
        "systems": ["sys-a", "sys-b", "sys-missing"],
        "comparisons": [["sys-a", "sys-b"]],
        "bootstrap": {"n_samples": 100, "seed": 13, "metrics": ["bleu", "chrf"]},
    }
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_evaluate_end_to_end(tmp_path: Path) -> None:
    pred_dir = _setup_pred_dir(tmp_path)
    cfg_path = _write_eval_config(tmp_path, pred_dir)
    systems_config_path = tmp_path / "systems.yaml"  # deliberately absent -> labels fall back to meta/name

    metrics_out = tmp_path / "results" / "metrics.json"
    per_segment_out = tmp_path / "results" / "eval" / "per_segment.jsonl"

    result = run_evaluate(cfg_path, systems_config_path, metrics_out, per_segment_out)

    # Returned dict and on-disk metrics.json agree.
    assert metrics_out.exists()
    on_disk = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert on_disk == result

    assert result["split"] == "test"
    assert result["n_segments"] == 4
    assert set(result["signatures"]) == {"bleu", "spbleu", "chrf"}
    assert result["bootstrap"]["n_samples"] == 100
    assert result["bootstrap"]["seed"] == 13
    assert "p_value" in result["bootstrap"]

    names = {s["name"] for s in result["systems"]}
    assert names == {"sys-a", "sys-b"}  # sys-missing is skipped
    for entry in result["systems"]:
        for field in [
            "name",
            "label",
            "split",
            "bleu",
            "bleu_tokenize",
            "spbleu",
            "chrf",
            "chrf_word_order",
            "hyp_path",
            "ref_path",
            "tokens_in",
            "tokens_out",
            "seconds",
            "sents_per_sec",
            "tokens_out_per_sec",
            "device",
        ]:
            assert field in entry, f"missing field {field!r} in system entry"
        assert "comet" not in entry  # no comet files were written

    sys_a = next(s for s in result["systems"] if s["name"] == "sys-a")
    sys_b = next(s for s in result["systems"] if s["name"] == "sys-b")
    assert sys_a["bleu"] > sys_b["bleu"]
    assert sys_a["label"] == "System A"  # no systems.yaml -> fallback to the label in meta.json
    assert sys_a["device"] == "cpu"

    assert len(result["significance"]) == 2  # bleu + chrf comparisons for sys-a vs sys-b
    for sig in result["significance"]:
        assert sig["a"] == "sys-a" and sig["b"] == "sys-b"
        assert set(sig) == {"a", "b", "metric", "a_score", "b_score", "delta", "ci95", "p_value"}

    # per_segment.jsonl
    assert per_segment_out.exists()
    seg_lines = [json.loads(line) for line in per_segment_out.read_text(encoding="utf-8").splitlines()]
    assert len(seg_lines) == 4
    for seg in seg_lines:
        assert set(seg) == {"pair_id", "doc_id", "src_len_words", "chrf"}
        assert set(seg["chrf"]) == {"sys-a", "sys-b"}
        assert seg["chrf"]["sys-a"] == 100.0


def test_evaluate_uses_systems_yaml_labels(tmp_path: Path) -> None:
    pred_dir = _setup_pred_dir(tmp_path)
    cfg_path = _write_eval_config(tmp_path, pred_dir)
    systems_config_path = tmp_path / "systems.yaml"
    systems_config_path.write_text(
        yaml.safe_dump({"systems": {"sys-a": {"label": "Pretty Label A"}, "sys-b": {"label": "Pretty Label B"}}}),
        encoding="utf-8",
    )
    metrics_out = tmp_path / "results2" / "metrics.json"
    per_segment_out = tmp_path / "results2" / "eval" / "per_segment.jsonl"

    result = run_evaluate(cfg_path, systems_config_path, metrics_out, per_segment_out)
    labels = {s["name"]: s["label"] for s in result["systems"]}
    assert labels == {"sys-a": "Pretty Label A", "sys-b": "Pretty Label B"}
