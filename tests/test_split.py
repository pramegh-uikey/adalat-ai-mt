"""Tests for adalat_mt.data.split."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from adalat_mt.data.split import assert_disjoint, make_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def test_make_split_sizes() -> None:
    doc_ids = [str(i) for i in range(1, 31)]
    split = make_split(doc_ids, seed=13, ratios=_RATIOS)
    assert len(split["train"]) == 24
    assert len(split["dev"]) == 3
    assert len(split["test"]) == 3


def test_make_split_deterministic_for_seed() -> None:
    doc_ids = [str(i) for i in range(1, 31)]
    split1 = make_split(doc_ids, seed=13, ratios=_RATIOS)
    split2 = make_split(doc_ids, seed=13, ratios=_RATIOS)
    assert split1 == split2


def test_make_split_disjoint_and_sorted() -> None:
    doc_ids = [str(i) for i in range(1, 31)]
    split = make_split(doc_ids, seed=13, ratios=_RATIOS)
    assert_disjoint(split)  # must not raise
    for ids in split.values():
        assert ids == sorted(ids, key=int)
    all_ids = set(split["train"]) | set(split["dev"]) | set(split["test"])
    assert all_ids == set(doc_ids)


def test_assert_disjoint_raises_on_overlap() -> None:
    bad = {"train": ["1", "2"], "dev": ["2", "3"], "test": ["4"]}
    with pytest.raises(ValueError):
        assert_disjoint(bad)


def test_processed_splits_share_no_doc_id_and_match_config() -> None:
    processed_dir = _REPO_ROOT / "data" / "processed"
    split_config_path = _REPO_ROOT / "configs" / "split.yaml"
    if not processed_dir.exists() or not split_config_path.exists():
        pytest.skip("data/processed/ not built yet")

    split_cfg = yaml.safe_load(split_config_path.read_text(encoding="utf-8")) or {}
    docs = split_cfg.get("docs")
    if not docs:
        pytest.skip("configs/split.yaml has no resolved docs yet")

    assert_disjoint(docs)

    doc_ids_by_split: dict[str, set[str]] = {}
    for name in ("train", "dev", "test"):
        path = processed_dir / f"{name}.jsonl"
        if not path.exists():
            pytest.skip(f"data/processed/{name}.jsonl not built yet")
        ids: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids.add(row["doc_id"])
        doc_ids_by_split[name] = ids
        assert ids == set(docs[name])

    assert not (doc_ids_by_split["train"] & doc_ids_by_split["dev"])
    assert not (doc_ids_by_split["train"] & doc_ids_by_split["test"])
    assert not (doc_ids_by_split["dev"] & doc_ids_by_split["test"])
