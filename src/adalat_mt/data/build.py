"""CLI: build the Phase 1 sentence-aligned EN-HI dataset.

    python -m adalat_mt.data.build --config configs/data.yaml

Reads `dataset/{english,hindi}/clean/{id}.txt`, segments and aligns every
document, splits by document into train/dev/test, and writes
`data/interim/alignments.jsonl`, `data/processed/{train,dev,test}.jsonl`,
`results/data/alignment_stats.json` and `results/data/alignment_report.md`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .align import AlignConfig, DocAlignment, EmbedFn, align_document, labse_embedder
from .report import write_report
from .segment import segment_document
from .sources import source_path
from .split import assert_disjoint, make_split
from .text_clean import read_text


def _doc_ids(dataset_dir: Path) -> list[str]:
    en_dir = dataset_dir / "english" / "clean"
    return sorted((p.stem for p in en_dir.glob("*.txt")), key=int)


def _align_config_from(cfg: dict[str, Any]) -> AlignConfig:
    a = cfg.get("align", {})
    len_ratio = tuple(a.get("len_ratio", [0.5, 2.5]))
    return AlignConfig(
        max_block=a.get("max_block", 3),
        skip_threshold=a.get("skip_threshold", 0.40),
        merge_penalty=a.get("merge_penalty", 0.05),
        anchor_bonus=a.get("anchor_bonus", 0.05),
        min_score=a.get("min_score", 0.60),
        len_ratio=len_ratio,
    )


def _compute_stats(
    doc_ids: list[str],
    doc_alignments: dict[str, DocAlignment],
    doc_to_split: dict[str, str],
) -> dict[str, Any]:
    per_doc: dict[str, Any] = {}
    all_scores: list[float] = []
    merge_counts: dict[str, int] = {}
    total_kept = 0
    total_dropped_by_reason: dict[str, int] = {}
    pairs_per_split: dict[str, int] = {}

    for doc_id in doc_ids:
        da = doc_alignments[doc_id]
        n_kept = 0
        dropped: dict[str, int] = {}
        doc_merge_counts: dict[str, int] = {}
        for p in da.pairs:
            all_scores.append(p.score)
            doc_merge_counts[p.align_type] = doc_merge_counts.get(p.align_type, 0) + 1
            merge_counts[p.align_type] = merge_counts.get(p.align_type, 0) + 1
            if p.kept:
                n_kept += 1
                total_kept += 1
                split_name = doc_to_split.get(doc_id)
                if split_name:
                    pairs_per_split[split_name] = pairs_per_split.get(split_name, 0) + 1
            else:
                reason = p.drop_reason or "unknown"
                dropped[reason] = dropped.get(reason, 0) + 1
                total_dropped_by_reason[reason] = total_dropped_by_reason.get(reason, 0) + 1
        per_doc[doc_id] = {
            "n_en_sents": da.n_src,
            "n_hi_sents": da.n_tgt,
            "n_pairs_kept": n_kept,
            "n_dropped": dropped,
            "n_src_skipped": da.n_src_skipped,
            "n_tgt_skipped": da.n_tgt_skipped,
            "merge_counts": doc_merge_counts,
            "split": doc_to_split.get(doc_id),
        }

    quantiles: dict[str, float] = {}
    if all_scores:
        arr = np.asarray(all_scores)
        quantiles = {f"p{q}": float(np.percentile(arr, q)) for q in (5, 25, 50, 75, 95)}

    return {
        "per_doc": per_doc,
        "total_pairs": len(all_scores),
        "total_kept": total_kept,
        "total_dropped_by_reason": total_dropped_by_reason,
        "merge_counts": merge_counts,
        "score_quantiles": quantiles,
        "pairs_per_split": pairs_per_split,
    }


def build_all(cfg: dict[str, Any], embed: EmbedFn, doc_ids: list[str] | None = None) -> dict[str, Any]:
    """Run the full pipeline with an injectable `embed` (for tests: a fake embedder).

    Writes all Phase 1 outputs to the directories named in `cfg` and returns a
    summary dict (doc ids, stats, resolved split).

    `doc_ids`, if given, restricts the run to exactly these document ids (in the given order)
    instead of every id under `cfg["dataset_dir"]`; every id must exist there. This is used by
    `--doc-ids` (smoke tests / quick iteration): when the split config (`cfg["split_config"]`)
    has no `docs` yet, the restricted ids are assigned round-robin to train/dev/test (in that
    fixed order) instead of via the ratio-based `make_split`, so e.g. 3 ids give one doc per
    split.
    """
    dataset_dir = Path(cfg["dataset_dir"])
    interim_dir = Path(cfg["interim_dir"])
    processed_dir = Path(cfg["processed_dir"])
    results_dir = Path(cfg["results_dir"])
    audit_dir = Path(cfg["audit_dir"])
    for d in (interim_dir, processed_dir, results_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    align_cfg = _align_config_from(cfg)
    recovered = cfg.get("recovered_hindi", {})
    recovered_ids = {str(d) for d in recovered.get("ids", [])}
    all_doc_ids = _doc_ids(dataset_dir)
    restricted = doc_ids is not None
    if restricted:
        unknown = sorted(set(doc_ids) - set(all_doc_ids), key=str)
        if unknown:
            raise ValueError(f"unknown doc ids: {unknown}")
        doc_ids = list(doc_ids)
    else:
        doc_ids = all_doc_ids

    doc_alignments: dict[str, DocAlignment] = {}
    for doc_id in doc_ids:
        en_text = read_text(source_path(doc_id, "en", cfg))
        hi_text = read_text(source_path(doc_id, "hi", cfg))
        en_sents = segment_document(en_text, "en")
        hi_sents = segment_document(hi_text, "hi")
        doc_alignments[doc_id] = align_document(en_sents, hi_sents, embed, align_cfg)

    # 2. data/interim/alignments.jsonl - every pair, including dropped ones.
    with open(interim_dir / "alignments.jsonl", "w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            for p in doc_alignments[doc_id].pairs:
                row = {
                    "doc_id": doc_id,
                    "src_idx": p.src_idx,
                    "tgt_idx": p.tgt_idx,
                    "en": p.en,
                    "hi": p.hi,
                    "score": p.score,
                    "kept": p.kept,
                    "drop_reason": p.drop_reason,
                    "align_type": p.align_type,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3. Split.
    split_path = Path(cfg["split_config"])
    split_cfg = yaml.safe_load(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    split_cfg = split_cfg or {}
    seed = split_cfg.get("seed", 13)
    ratios = split_cfg.get("ratios", {"train": 0.8, "dev": 0.1, "test": 0.1})
    docs = split_cfg.get("docs")
    if not docs:
        if restricted:
            # Deterministic round-robin over the caller-given order (smoke tests): with exactly
            # 3 restricted doc ids this gives one doc per split.
            split_names = ("train", "dev", "test")
            docs = {name: [] for name in split_names}
            for i, doc_id in enumerate(doc_ids):
                docs[split_names[i % len(split_names)]].append(doc_id)
        else:
            docs = make_split(doc_ids, seed, ratios)
        assert_disjoint(docs)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            f.write(f"# Document-level split (seed={seed}). Filled in by `python -m adalat_mt.data.build`.\n")
            yaml.safe_dump(
                {"seed": seed, "ratios": ratios, "docs": docs}, f, sort_keys=False, allow_unicode=True
            )
    else:
        assert_disjoint(docs)

    # 4. data/processed/{train,dev,test}.jsonl - kept pairs only.
    doc_to_split = {doc_id: name for name, ids in docs.items() for doc_id in ids}
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in docs}
    for doc_id in doc_ids:
        split_name = doc_to_split.get(doc_id)
        if split_name is None:
            continue
        idx = 0
        for p in doc_alignments[doc_id].pairs:
            if not p.kept:
                continue
            split_rows[split_name].append(
                {
                    "doc_id": doc_id,
                    "pair_id": f"{doc_id}-{idx:04d}",
                    "en": p.en,
                    "hi": p.hi,
                    "align_score": p.score,
                    "align_type": p.align_type,
                }
            )
            idx += 1
    for name, rows in split_rows.items():
        with open(processed_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5. results/data/alignment_stats.json
    stats = _compute_stats(doc_ids, doc_alignments, doc_to_split)
    stats["recovered_hindi_docs"] = {"ids": sorted(recovered_ids, key=int), "note": recovered.get("note", "")}
    stats["align_config"] = asdict(align_cfg)
    (results_dir / "alignment_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 6. results/data/alignment_report.md + data/audit/alignment_sample.jsonl
    write_report(
        doc_ids=doc_ids,
        doc_alignments=doc_alignments,
        doc_to_split=doc_to_split,
        stats=stats,
        results_dir=results_dir,
        audit_dir=audit_dir,
        audit_sample_size=cfg.get("audit_sample_size", 20),
        audit_seed=cfg.get("audit_seed", 13),
    )

    return {"doc_ids": doc_ids, "stats": stats, "docs": docs}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python -m adalat_mt.data.build --config configs/data.yaml`."""
    parser = argparse.ArgumentParser(description="Build the Phase 1 aligned EN-HI dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        default=None,
        help="restrict the run to these document ids, in order (smoke tests); with no docs yet "
        "in the split config, they are assigned round-robin to train/dev/test",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="rewrite interim/processed/results/audit dirs under this root and write the split "
        "to <out-root>/split.yaml instead of configs/split.yaml (smoke tests; leaves the real "
        "configs/split.yaml untouched)",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.out_root:
        root = args.out_root
        cfg["interim_dir"] = str(root / cfg["interim_dir"])
        cfg["processed_dir"] = str(root / cfg["processed_dir"])
        cfg["results_dir"] = str(root / cfg["results_dir"])
        cfg["audit_dir"] = str(root / cfg["audit_dir"])
        cfg["split_config"] = str(root / "split.yaml")

    embed = labse_embedder(
        model_name=cfg.get("labse_model", "sentence-transformers/LaBSE"),
        cache_dir=Path(cfg.get("labse_cache_dir", ".cache/labse")),
    )
    build_all(cfg, embed, doc_ids=args.doc_ids)


if __name__ == "__main__":
    main()
