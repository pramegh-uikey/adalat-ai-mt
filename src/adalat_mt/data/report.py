"""Render `results/data/alignment_report.md` and the alignment audit sample."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

from .align import AlignedPair, DocAlignment


def _histogram(scores: list[float], n_bins: int = 10) -> str:
    """A 10-bin text histogram of scores in [0, 1] (scores outside are clamped)."""
    if not scores:
        return "(no scores)"
    width = 1.0 / n_bins
    counts = [0] * n_bins
    for s in scores:
        idx = int(min(max(s, 0.0), 0.999999) / width)
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo, hi = i * width, (i + 1) * width
        bar = "#" * max(1, round(40 * c / max_count)) if c else ""
        lines.append(f"[{lo:.1f},{hi:.1f}) {c:5d} {bar}")
    return "\n".join(lines)


def write_report(
    doc_ids: list[str],
    doc_alignments: dict[str, DocAlignment],
    doc_to_split: dict[str, str],
    stats: dict[str, Any],
    results_dir: Path,
    audit_dir: Path,
    audit_sample_size: int,
    audit_seed: int,
) -> None:
    """Write `alignment_report.md` (results_dir) and `alignment_sample.jsonl` (audit_dir)."""
    lines: list[str] = ["# Alignment report", ""]
    recovered = stats.get("recovered_hindi_docs", {})
    lines += [
        "## Method",
        "",
        "Hard-wrapped text is unwrapped into segments, cleaned (BOM/zero-width removal, EN `li.`→`ii.` OCR fix, HI date-space"
        " and double-danda fixes) and split into sentences with rule-based EN/HI splitters. Each document is then aligned"
        " with a monotonic dynamic programme over LaBSE embeddings of 1–3-sentence blocks (moves 1-1, 1-2, 2-1, 1-3, 3-1"
        " and skips), with a soft paragraph-number anchor bonus. Pairs below the score threshold or outside the length-ratio"
        f" band are dropped. Config: `{stats.get('align_config', {})}`.",
        "",
    ]
    if recovered.get("ids"):
        lines += [f"**Hindi text recovered from PDF (OCR + text-layer numbers):** docs "
                  f"{', '.join(map(str, recovered['ids']))} — {recovered.get('note', '')}", ""]

    lines += ["## Pairs per document", "", "| doc_id | split | n_en | n_hi | kept | dropped |", "|---|---|---|---|---|---|"]
    for doc_id in doc_ids:
        d = stats["per_doc"][doc_id]
        n_dropped = sum(d["n_dropped"].values())
        lines.append(
            f"| {doc_id} | {d['split']} | {d['n_en_sents']} | {d['n_hi_sents']} "
            f"| {d['n_pairs_kept']} | {n_dropped} |"
        )
    lines.append("")

    all_scores = [p.score for doc_id in doc_ids for p in doc_alignments[doc_id].pairs]
    quantiles_line = ", ".join(f"{k}={v:.4f}" for k, v in stats["score_quantiles"].items())
    lines += [
        "## Score distribution",
        "",
        f"Quantiles: {quantiles_line}" if quantiles_line else "Quantiles: (no pairs)",
        "",
        "```",
        _histogram(all_scores),
        "```",
        "",
    ]

    total_pairs = stats["total_pairs"] or 1
    drop_rate = sum(stats["total_dropped_by_reason"].values()) / total_pairs
    merge_rate = sum(c for t, c in stats["merge_counts"].items() if t != "1-1") / total_pairs
    lines += [
        "## Drop / merge rates",
        "",
        f"- Drop rate: {drop_rate:.4f} (by reason: {stats['total_dropped_by_reason']})",
        f"- Merge rate (non 1-1 pairs): {merge_rate:.4f} (by type: {stats['merge_counts']})",
        f"- Pairs per split: {stats['pairs_per_split']}",
        "",
    ]

    kept_pairs = _kept_pairs_with_ids(doc_ids, doc_alignments)

    sample = random.Random(audit_seed).sample(kept_pairs, min(audit_sample_size, len(kept_pairs)))

    audit_dir.mkdir(parents=True, exist_ok=True)
    sample_path = audit_dir / "alignment_sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for pair_id, p in sample:
            f.write(
                json.dumps(
                    {"pair_id": pair_id, "score": p.score, "align_type": p.align_type, "en": p.en, "hi": p.hi},
                    ensure_ascii=False,
                )
                + "\n"
            )

    judgements_path = audit_dir / "alignment_judgements.yaml"
    judgements: dict[str, Any] = {}
    if judgements_path.exists():
        judgements = yaml.safe_load(judgements_path.read_text(encoding="utf-8")) or {}

    lines += ["## Audit sample (random kept pairs, all splits)", ""]
    lines += _audit_section(sample, judgements)
    sampled = {pair_id for pair_id, _ in sample}
    by_id = dict(kept_pairs)
    extra = [(pid, by_id[pid]) for pid in judgements if pid not in sampled and pid in by_id]
    if extra:
        lines += ["## Supplementary audit (OCR-recovered Hindi documents)", ""]
        lines += _audit_section(extra, judgements)

    (results_dir / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (sensible for small n such as a 20-pair audit)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _kept_pairs_with_ids(doc_ids: list[str], doc_alignments: dict[str, DocAlignment]) -> list[tuple[str, AlignedPair]]:
    """Kept pairs with their ``<doc>-<nnnn>`` pair ids (same numbering as the processed jsonl)."""
    out: list[tuple[str, AlignedPair]] = []
    for doc_id in doc_ids:
        kept = [p for p in doc_alignments[doc_id].pairs if p.kept]
        out += [(f"{doc_id}-{idx:04d}", p) for idx, p in enumerate(kept)]
    return out


def _audit_section(items: list[tuple[str, AlignedPair]], judgements: dict[str, Any]) -> list[str]:
    """Precision summary plus a table of judged pairs (verdict and note per pair)."""
    judged = [judgements[pid]["verdict"] for pid, _ in items if pid in judgements]
    if judged:
        correct = sum(v == "correct" for v in judged)
        lenient = sum(v in ("correct", "partial") for v in judged)
        lo, hi = wilson_interval(correct, len(judged))
        out = [f"Judged by reading: {len(judged)}/{len(items)}; precision (correct) = {correct}/{len(judged)} = "
               f"{correct / len(judged):.3f} (95% Wilson CI {lo:.3f}–{hi:.3f}); lenient (correct + partial) = "
               f"{lenient / len(judged):.3f}", ""]
    else:
        out = ["Not yet judged.", ""]
    out += ["| pair_id | score | verdict | note | en | hi |", "|---|---|---|---|---|---|"]
    for pid, p in items:
        j = judgements.get(pid, {})
        cells = [c.replace("|", "\\|") for c in (j.get("note", ""), p.en, p.hi)]
        out.append(f"| {pid} | {p.score:.4f} | {j.get('verdict', '-')} | {cells[0]} | {cells[1]} | {cells[2]} |")
    out.append("")
    return out
