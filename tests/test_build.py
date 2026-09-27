"""Tests for the `--doc-ids`/`--out-root` smoke-test support in `adalat_mt.data.build`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from adalat_mt.data.build import build_all

REAL_SPLIT_CONFIG = Path("configs/split.yaml")


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic, network-free stand-in for LaBSE: hash-based unit vectors."""
    dim = 16
    rows = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.normal(size=dim)
        rows.append(v / np.linalg.norm(v))
    return np.asarray(rows)


def _base_cfg(out_root: Path) -> dict:
    return {
        "dataset_dir": "dataset",
        "interim_dir": str(out_root / "data" / "interim"),
        "processed_dir": str(out_root / "data" / "processed"),
        "results_dir": str(out_root / "results" / "data"),
        "audit_dir": str(out_root / "data" / "audit"),
        "split_config": str(out_root / "split.yaml"),
        "audit_sample_size": 5,
        "audit_seed": 13,
        "align": {
            "max_block": 3, "skip_threshold": 0.40, "merge_penalty": 0.05,
            "anchor_bonus": 0.05, "min_score": 0.60, "len_ratio": [0.5, 2.5],
        },
    }


@pytest.mark.skipif(not Path("dataset/english/clean/1.txt").exists(), reason="raw dataset not present")
def test_doc_ids_and_out_root_round_robin_split(tmp_path: Path):
    out_root = tmp_path / "smoke"
    cfg = _base_cfg(out_root)

    before = REAL_SPLIT_CONFIG.read_text(encoding="utf-8")

    result = build_all(cfg, _fake_embed, doc_ids=["1", "2", "3"])

    assert REAL_SPLIT_CONFIG.read_text(encoding="utf-8") == before  # real config untouched

    docs = result["docs"]
    assert docs == {"train": ["1"], "dev": ["2"], "test": ["3"]}

    split_written = yaml.safe_load((out_root / "split.yaml").read_text(encoding="utf-8"))
    assert split_written["docs"] == docs

    for split_name in ("train", "dev", "test"):
        path = out_root / "data" / "processed" / f"{split_name}.jsonl"
        assert path.exists()

    assert (out_root / "results" / "data" / "alignment_stats.json").exists()
    assert (out_root / "results" / "data" / "alignment_report.md").exists()


@pytest.mark.skipif(not Path("dataset/english/clean/1.txt").exists(), reason="raw dataset not present")
def test_unknown_doc_id_raises(tmp_path: Path):
    out_root = tmp_path / "smoke"
    cfg = _base_cfg(out_root)
    with pytest.raises(ValueError):
        build_all(cfg, _fake_embed, doc_ids=["1", "999999"])


@pytest.mark.skipif(not Path("dataset/english/clean/1.txt").exists(), reason="raw dataset not present")
def test_no_doc_ids_filter_uses_every_document(tmp_path: Path):
    out_root = tmp_path / "smoke_full"
    cfg = _base_cfg(out_root)
    cfg["split_config"] = str(out_root / "split.yaml")
    # Provide an explicit tiny split so the (slow) 30-doc ratio path isn't exercised here.
    (out_root).mkdir(parents=True, exist_ok=True)
    split_path = out_root / "split.yaml"
    split_path.write_text(
        yaml.safe_dump({"seed": 13, "docs": {"train": ["1"], "dev": ["2"], "test": ["3"]}}), encoding="utf-8"
    )
    result = build_all(cfg, _fake_embed)
    assert set(result["doc_ids"]) >= {"1", "2", "3"}  # ran over the full dataset, not just 3 docs
    assert result["docs"] == {"train": ["1"], "dev": ["2"], "test": ["3"]}
