"""Deterministic document-level train/dev/test split."""
from __future__ import annotations

import random


def make_split(doc_ids: list[str], seed: int, ratios: dict[str, float]) -> dict[str, list[str]]:
    """Shuffle `doc_ids` with `random.Random(seed)` and split by `ratios`.

    Dev and test sizes are `round(n * ratio)`; train gets the remainder. Each
    returned list is sorted numerically (by `int(doc_id)`).
    """
    ids = list(doc_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_dev = round(n * ratios.get("dev", 0.0))
    n_test = round(n * ratios.get("test", 0.0))
    n_train = n - n_dev - n_test

    train = ids[:n_train]
    dev = ids[n_train : n_train + n_dev]
    test = ids[n_train + n_dev : n_train + n_dev + n_test]

    return {
        "train": sorted(train, key=int),
        "dev": sorted(dev, key=int),
        "test": sorted(test, key=int),
    }


def assert_disjoint(split: dict[str, list[str]]) -> None:
    """Raise `ValueError` if any doc id appears in more than one split list."""
    seen: dict[str, str] = {}
    for name, ids in split.items():
        for doc_id in ids:
            if doc_id in seen:
                raise ValueError(
                    f"doc id {doc_id!r} appears in both {seen[doc_id]!r} and {name!r} splits"
                )
            seen[doc_id] = name
