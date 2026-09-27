"""CLI: aggregate MT system predictions into `results/metrics.json` and `results/eval/per_segment.jsonl`.

``python -m adalat_mt.evaluation.evaluate --config configs/eval.yaml``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from adalat_mt.evaluation.metrics import corpus_metrics, sentence_chrf
from adalat_mt.evaluation.significance import paired_bootstrap, paired_bootstrap_segments

DEFAULT_BOOTSTRAP = {"n_samples": 1000, "seed": 13, "metrics": ["bleu", "spbleu", "chrf"]}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _load_labels(systems_config_path: Path) -> dict[str, str]:
    """Map system name -> label from `configs/systems.yaml`; empty dict if the file is missing."""
    if not systems_config_path.exists():
        return {}
    cfg = _load_yaml(systems_config_path) or {}
    return {name: sub.get("label", name) for name, sub in cfg.get("systems", {}).items()}


def _load_meta(pred_dir: Path, name: str, split: str) -> dict:
    meta_path = pred_dir / f"{name}.{split}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _load_rows(data_dir: Path, split: str, n: int) -> list[dict]:
    """Read `<data_dir>/<split>.jsonl` for `pair_id`/`doc_id`; fall back to synthetic ids if unavailable."""
    path = data_dir / f"{split}.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) == n:
            return rows
    return [{"pair_id": f"{split}-{i}", "doc_id": None} for i in range(n)]


def run_evaluate(
    config_path: Path,
    systems_config_path: Path = Path("configs/systems.yaml"),
    metrics_out_path: Path = Path("results/metrics.json"),
    per_segment_out_path: Path = Path("results/eval/per_segment.jsonl"),
) -> dict:
    """Run the full evaluation pipeline described by `config_path` (see `configs/eval.yaml`).

    Systems listed in the config whose hyp file is missing (or has a mismatched line count) are skipped
    with a printed warning, so this can run before all systems/LoRA training are done. Writes
    `metrics_out_path` and `per_segment_out_path` and also returns the metrics dict.
    """
    cfg = _load_yaml(config_path)
    split = cfg["split"]
    pred_dir = Path(cfg["pred_dir"])
    comet_dir = Path(cfg.get("comet_dir", "results/comet"))
    data_dir = Path(cfg.get("data_dir", "data/processed"))
    wanted_systems: list[str] = cfg["systems"]
    comparisons: list[list[str]] = cfg.get("comparisons", [])
    bootstrap_cfg = cfg.get("bootstrap", DEFAULT_BOOTSTRAP)
    n_samples = bootstrap_cfg.get("n_samples", 1000)
    seed = bootstrap_cfg.get("seed", 13)
    bootstrap_metrics = bootstrap_cfg.get("metrics", ["bleu", "spbleu", "chrf"])

    labels = _load_labels(systems_config_path)

    ref_path = pred_dir / f"{split}.ref.hi"
    if not ref_path.exists():
        raise FileNotFoundError(f"missing reference file: {ref_path}")
    refs = _read_lines(ref_path)
    n = len(refs)

    src_path = pred_dir / f"{split}.src.en"
    srcs = _read_lines(src_path) if src_path.exists() else [""] * n
    rows = _load_rows(data_dir, split, n)

    systems: list[dict] = []
    hyps_by_system: dict[str, list[str]] = {}
    chrf_by_system: dict[str, list[float]] = {}
    comet_segments_by_system: dict[str, list[float]] = {}
    signatures: dict[str, str] = {}

    for name in wanted_systems:
        hyp_path = pred_dir / f"{name}.{split}.hi"
        if not hyp_path.exists():
            print(f"[evaluate] warning: skipping {name!r}: missing {hyp_path}")
            continue
        hyps = _read_lines(hyp_path)
        if len(hyps) != n:
            print(f"[evaluate] warning: skipping {name!r}: {len(hyps)} hyps != {n} refs")
            continue

        metrics = corpus_metrics(hyps, refs)
        signatures = metrics["signatures"]
        meta = _load_meta(pred_dir, name, split)

        entry = {
            "name": name,
            "label": labels.get(name, meta.get("label", name)),
            "split": split,
            "bleu": metrics["bleu"],
            "bleu_tokenize": metrics["bleu_tokenize"],
            "spbleu": metrics["spbleu"],
            "chrf": metrics["chrf"],
            "chrf_word_order": metrics["chrf_word_order"],
            "hyp_path": str(hyp_path),
            "ref_path": str(ref_path),
            "tokens_in": meta.get("tokens_in"),
            "tokens_out": meta.get("tokens_out"),
            "seconds": meta.get("seconds"),
            "sents_per_sec": meta.get("sents_per_sec"),
            "tokens_out_per_sec": meta.get("tokens_out_per_sec"),
            "device": meta.get("device"),
        }

        comet_path = comet_dir / f"{name}.{split}.json"
        if comet_path.exists():
            comet_data = json.loads(comet_path.read_text(encoding="utf-8"))
            entry["comet"] = round(comet_data["system_score"], 4)
            comet_segments_by_system[name] = comet_data["segment_scores"]

        systems.append(entry)
        hyps_by_system[name] = hyps
        chrf_by_system[name] = sentence_chrf(hyps, refs)

    present = {e["name"] for e in systems}

    significance: list[dict] = []
    for a, b in comparisons:
        if a not in present or b not in present:
            continue
        for metric in bootstrap_metrics:
            result = paired_bootstrap(
                hyps_by_system[a], hyps_by_system[b], refs, metric=metric, n_samples=n_samples, seed=seed
            )
            significance.append({"a": a, "b": b, **result})
        if a in comet_segments_by_system and b in comet_segments_by_system:
            comet_result = paired_bootstrap_segments(
                comet_segments_by_system[a],
                comet_segments_by_system[b],
                n_samples=n_samples,
                seed=seed,
                metric="comet",
            )
            significance.append({"a": a, "b": b, **comet_result})

    metrics_json = {
        "split": split,
        "n_segments": n,
        "signatures": signatures,
        "bootstrap": {
            "n_samples": n_samples,
            "seed": seed,
            "p_value": "one-sided, fraction of resamples with delta<=0",
        },
        "systems": systems,
        "significance": significance,
    }

    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.write_text(json.dumps(metrics_json, indent=2) + "\n", encoding="utf-8")

    per_segment_out_path.parent.mkdir(parents=True, exist_ok=True)
    with per_segment_out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = rows[i]
            src_text = srcs[i] if i < len(srcs) else ""
            seg = {
                "pair_id": row.get("pair_id"),
                "doc_id": row.get("doc_id"),
                "src_len_words": len(src_text.split()),
                "chrf": {name: chrf_by_system[name][i] for name in present},
            }
            comet_seg = {
                name: comet_segments_by_system[name][i]
                for name in comet_segments_by_system
                if i < len(comet_segments_by_system[name])
            }
            if comet_seg:
                seg["comet"] = comet_seg
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    return metrics_json


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Aggregate MT evaluation results.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--systems-config", default="configs/systems.yaml")
    parser.add_argument("--metrics-out", default="results/metrics.json")
    parser.add_argument("--per-segment-out", default="results/eval/per_segment.jsonl")
    args = parser.parse_args(argv)
    run_evaluate(
        Path(args.config),
        Path(args.systems_config),
        Path(args.metrics_out),
        Path(args.per_segment_out),
    )


if __name__ == "__main__":
    main()
