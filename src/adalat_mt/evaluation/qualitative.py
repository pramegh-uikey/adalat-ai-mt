"""Qualitative tooling: category tagging, candidate mining and curated-example
rendering for EN->HI legal MT outputs. Pure CPU, no model downloads.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from adalat_mt.evaluation.analysis import (
    addition_flag,
    find_citations,
    omission_flag,
    read_jsonl,
    read_lines,
    repetition_flag,
    sentence_chrf,
)

#: Fixed, canonically ordered qualitative category ids.
CATEGORIES: tuple[str, ...] = (
    "legal-term",
    "archaic/formulaic",
    "long-sentence",
    "citation/section/date",
    "named-entity",
    "omission/hallucination",
)

_ARCHAIC_RE = re.compile(
    r"hereinafter|the said|notwithstanding|aforesaid|thereof|therein|whereby|hereby|inter alia|"
    r"learned counsel|leave granted|it is ordered|mutatis mutandis|in the facts and circumstances",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
_SECTION_RE = re.compile(r"\b(?:Section|Article|Order|Rule)s?\.?\s*\d+", re.IGNORECASE)
_CASE_NAME_RE = re.compile(r"\b[A-Z][A-Za-z.&'()]*\s+v\.?s?\.?\s+[A-Z][A-Za-z.&'()]*\b")

_NE_STOPLIST = {"The", "This", "Court", "High", "Supreme", "Act", "Section", "State"}
_CAP_WORD_RE = re.compile(r"^[A-Z][a-zA-Z'.]*$")
_SENTENCE_END_RE = re.compile(r"[.!?]$")

LONG_SENTENCE_WORDS = 45


def _has_legal_term(src: str, glossary: dict[str, Any]) -> bool:
    for term in glossary.get("terms", []):
        if term.get("category") != "legal-term":
            continue
        if re.search(rf"\b(?:{term['en']})\b", src, re.IGNORECASE):
            return True
    return False


def _has_named_entity(src: str) -> bool:
    if _CASE_NAME_RE.search(src):
        return True
    tokens = src.split()
    for i in range(len(tokens) - 1):
        # A token is a "sentence start" if it is the first token, or the
        # previous token ends a sentence; consecutive capitalised words
        # starting there don't count as a named entity.
        is_sentence_start = i == 0 or bool(_SENTENCE_END_RE.search(tokens[i - 1]))
        if is_sentence_start:
            continue
        w0, w1 = tokens[i].strip("(),;:\""), tokens[i + 1].strip("(),;:\"")
        if (
            _CAP_WORD_RE.match(w0)
            and _CAP_WORD_RE.match(w1)
            and w0 not in _NE_STOPLIST
            and w1 not in _NE_STOPLIST
        ):
            return True
    return False


def _has_citation_section_date(src: str) -> bool:
    if _DATE_RE.search(src):
        return True
    if _SECTION_RE.search(src):
        return True
    if find_citations(src):
        return True
    return False


def tag_segment(
    src: str, ref: str, hyps: dict[str, str], glossary: dict[str, Any]
) -> list[str]:
    """Tag a segment with the qualitative categories it exercises.

    ``hyps`` maps system id -> hypothesis text; used only for the
    omission/hallucination tag (any system showing omission, addition or
    repetition on this segment).
    """
    tags: list[str] = []

    if _has_legal_term(src, glossary):
        tags.append("legal-term")
    if _ARCHAIC_RE.search(src):
        tags.append("archaic/formulaic")
    if len(src.split()) > LONG_SENTENCE_WORDS:
        tags.append("long-sentence")
    if _has_citation_section_date(src):
        tags.append("citation/section/date")
    if _has_named_entity(src):
        tags.append("named-entity")
    if any(
        repetition_flag(hyp) or omission_flag(hyp, ref) or addition_flag(hyp, ref)
        for hyp in hyps.values()
    ):
        tags.append("omission/hallucination")

    return tags


# --------------------------------------------------------------------------
# candidates: mine every test segment, tagged and scored
# --------------------------------------------------------------------------


def _load_system_hyps(pred_dir: Path, system: str, n: int, split: str = "test") -> list[str] | None:
    path = pred_dir / f"{system}.{split}.hi"
    if not path.exists():
        return None
    lines = read_lines(path)
    if len(lines) != n:
        return None
    return lines


def _load_rows(data_dir: Path, split: str, n: int) -> list[dict[str, Any]]:
    """Read ``<data_dir>/<split>.jsonl``; fall back to synthetic ids if unavailable."""
    path = data_dir / f"{split}.jsonl"
    if path.exists():
        rows = read_jsonl(path)
        if len(rows) == n:
            return rows
    return [{"pair_id": f"{split}-{i}", "doc_id": None} for i in range(n)]


def build_candidates(
    data_dir: str | Path,
    pred_dir: str | Path,
    baseline: str,
    adapted: str,
    glossary: dict[str, Any],
    other_systems: list[str] | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Build the per-segment candidate list, sorted by ``|delta|`` descending."""
    data_dir = Path(data_dir)
    pred_dir = Path(pred_dir)

    srcs = read_lines(pred_dir / f"{split}.src.en")
    refs = read_lines(pred_dir / f"{split}.ref.hi")
    n = len(refs)
    rows = _load_rows(data_dir, split, n)
    baseline_hyps = _load_system_hyps(pred_dir, baseline, n, split)
    adapted_hyps = _load_system_hyps(pred_dir, adapted, n, split)
    if baseline_hyps is None:
        raise FileNotFoundError(f"missing/misaligned baseline hypotheses for {baseline!r}")
    if adapted_hyps is None:
        raise FileNotFoundError(f"missing/misaligned adapted hypotheses for {adapted!r}")

    others = {}
    for system in other_systems or []:
        hyps = _load_system_hyps(pred_dir, system, n, split)
        if hyps is not None:
            others[system] = hyps

    candidates = []
    for i, row in enumerate(rows):
        src, ref = srcs[i], refs[i]
        base_hyp, adapt_hyp = baseline_hyps[i], adapted_hyps[i]
        chrf_baseline = sentence_chrf(base_hyp, ref)
        chrf_adapted = sentence_chrf(adapt_hyp, ref)
        delta = chrf_adapted - chrf_baseline
        hyps_for_tags = {baseline: base_hyp, adapted: adapt_hyp}
        hyps_for_tags.update({sys_id: hyp[i] for sys_id, hyp in others.items()})
        tags = tag_segment(src, ref, hyps_for_tags, glossary)
        candidates.append(
            {
                "pair_id": row.get("pair_id"),
                "doc_id": row.get("doc_id"),
                "tags": tags,
                "source_en": src,
                "reference_hi": ref,
                "baseline_hi": base_hyp,
                "adapted_hi": adapt_hyp,
                "chrf_baseline": chrf_baseline,
                "chrf_adapted": chrf_adapted,
                "delta": delta,
                "others": {sys_id: hyp[i] for sys_id, hyp in others.items()},
            }
        )

    candidates.sort(key=lambda c: abs(c["delta"]), reverse=True)
    return candidates


def _discover_other_systems(pred_dir: Path, exclude: set[str], split: str = "test") -> list[str]:
    suffix = f".{split}.hi"
    others = []
    for path in sorted(pred_dir.glob(f"*{suffix}")):
        system = path.name[: -len(suffix)]
        if system not in exclude:
            others.append(system)
    return others


def cmd_candidates(args: argparse.Namespace) -> None:
    """CLI subcommand: mine and write ``results/qualitative/candidates.jsonl``."""
    glossary = yaml.safe_load(Path(args.glossary).read_text(encoding="utf-8")) or {}
    pred_dir = Path(args.pred_dir)
    other_systems = _discover_other_systems(pred_dir, {args.baseline, args.adapted}, args.split)
    candidates = build_candidates(
        args.data_dir, pred_dir, args.baseline, args.adapted, glossary, other_systems, args.split
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# render: curated.yaml -> examples.md + curated_examples.jsonl
# --------------------------------------------------------------------------


def _load_records(
    data_dir: Path, pred_dir: Path, baseline: str, adapted: str, split: str = "test"
) -> dict[str, dict[str, Any]]:
    """Load one record per ``pair_id`` with source/reference/hyp texts and scores."""
    srcs = read_lines(pred_dir / f"{split}.src.en")
    refs = read_lines(pred_dir / f"{split}.ref.hi")
    n = len(refs)
    rows = _load_rows(data_dir, split, n)
    baseline_hyps = _load_system_hyps(pred_dir, baseline, n, split)
    adapted_hyps = _load_system_hyps(pred_dir, adapted, n, split)
    if baseline_hyps is None:
        raise FileNotFoundError(f"missing/misaligned baseline hypotheses for {baseline!r}")
    if adapted_hyps is None:
        raise FileNotFoundError(f"missing/misaligned adapted hypotheses for {adapted!r}")

    records = {}
    for i, row in enumerate(rows):
        pair_id = row.get("pair_id")
        records[pair_id] = {
            "pair_id": pair_id,
            "doc_id": row.get("doc_id"),
            "source_en": srcs[i],
            "reference_hi": refs[i],
            "baseline_hi": baseline_hyps[i],
            "adapted_hi": adapted_hyps[i],
            "chrf_baseline": sentence_chrf(baseline_hyps[i], refs[i]),
            "chrf_adapted": sentence_chrf(adapted_hyps[i], refs[i]),
        }
    return records


def render_examples(
    curated: dict[str, Any],
    records: dict[str, dict[str, Any]],
    baseline_label: str,
    adapted_label: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Render curated examples to Markdown.

    Raises ``ValueError`` if fewer than 12 examples are curated, or if any of
    the six fixed categories has no example. Returns ``(markdown, jsonl_rows)``.
    """
    examples = curated.get("examples", [])
    if len(examples) < 12:
        raise ValueError(f"need >= 12 curated examples, got {len(examples)}")

    coverage: dict[str, list[int]] = {cat: [] for cat in CATEGORIES}
    for k, ex in enumerate(examples, start=1):
        for cat in ex.get("categories", []):
            if cat in coverage:
                coverage[cat].append(k)

    missing = [cat for cat, nums in coverage.items() if not nums]
    if missing:
        raise ValueError(f"no curated example for categories: {missing}")

    lines: list[str] = []
    intro = curated.get("intro")
    if intro:
        lines.append(str(intro))
        lines.append("")

    lines.append("## Category coverage")
    lines.append("")
    lines.append("| Category | Examples |")
    lines.append("|---|---|")
    for cat in CATEGORIES:
        nums = ", ".join(f"#{n}" for n in coverage[cat])
        lines.append(f"| {cat} | {nums} |")
    lines.append("")

    jsonl_rows: list[dict[str, Any]] = []
    for k, ex in enumerate(examples, start=1):
        pair_id = ex["pair_id"]
        record = records[pair_id]
        cats = ", ".join(ex.get("categories", []))
        lines.append(f"### Example {k} — {cats}")
        lines.append("")
        lines.append(f"pair_id: `{record['pair_id']}` doc_id: `{record['doc_id']}`")
        lines.append("")
        lines.append("| | Text | chrF++ |")
        lines.append("|---|---|---|")
        lines.append(f"| **Source (EN)** | {record['source_en']} | |")
        lines.append(f"| **Reference (HI)** | {record['reference_hi']} | |")
        lines.append(
            f"| **Baseline ({baseline_label})** | {record['baseline_hi']} | {record['chrf_baseline']:.1f} |"
        )
        lines.append(
            f"| **Adapted ({adapted_label})** | {record['adapted_hi']} | {record['chrf_adapted']:.1f} |"
        )
        lines.append("")
        lines.append("**Analysis.** " + str(ex.get("analysis", "")).strip())
        lines.append("")

        jsonl_rows.append(
            {
                "id": pair_id,
                "source_en": record["source_en"],
                "reference_hi": record["reference_hi"],
                "baseline_hi": record["baseline_hi"],
                "adapted_hi": record["adapted_hi"],
            }
        )

    critic_summary = curated.get("critic_summary")
    if critic_summary:
        lines.append("## Second opinion (LLM judgement — not a metric)")
        lines.append("")
        lines.append(str(critic_summary))
        lines.append("")

    return "\n".join(lines), jsonl_rows


def _resolve_baseline_adapted(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve baseline/adapted system ids from CLI flags or ``configs/eval.yaml``.

    The adapted system defaults to the ``b`` of the first ``comparisons`` pair
    whose ``a`` is the baseline (the primary baseline-vs-LoRA comparison).
    """
    if args.baseline and args.adapted:
        return args.baseline, args.adapted
    eval_cfg_path = Path(args.eval_config)
    eval_cfg = yaml.safe_load(eval_cfg_path.read_text(encoding="utf-8")) if eval_cfg_path.exists() else {}
    eval_cfg = eval_cfg or {}
    baseline = args.baseline or eval_cfg.get("baseline")
    adapted = args.adapted
    if adapted is None:
        for pair in eval_cfg.get("comparisons", []):
            if len(pair) == 2 and pair[0] == baseline:
                adapted = pair[1]
                break
    if not baseline or not adapted:
        raise ValueError("could not resolve --baseline/--adapted from CLI flags or --eval-config")
    return baseline, adapted


def _label(systems_cfg: dict[str, Any], system: str) -> str:
    """Look up a system's display label from a ``configs/systems.yaml``-shaped dict."""
    systems = (systems_cfg or {}).get("systems", {})
    return systems.get(system, {}).get("label", system)


def cmd_render(args: argparse.Namespace) -> None:
    """CLI subcommand: render curated examples to Markdown + JSONL."""
    curated = yaml.safe_load(Path(args.curated).read_text(encoding="utf-8")) or {}
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    systems_cfg_path = Path(args.systems_config)
    systems_cfg = (
        yaml.safe_load(systems_cfg_path.read_text(encoding="utf-8")) if systems_cfg_path.exists() else {}
    ) or {}

    baseline, adapted = _resolve_baseline_adapted(args)
    records = _load_records(data_dir, pred_dir, baseline, adapted, args.split)
    markdown, jsonl_rows = render_examples(
        curated,
        records,
        _label(systems_cfg, baseline),
        _label(systems_cfg, adapted),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.md").write_text(markdown, encoding="utf-8")
    with (out_dir / "curated_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``candidates`` and ``render`` subcommands."""
    parser = argparse.ArgumentParser(description="Qualitative tooling for EN->HI MT outputs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cand = sub.add_parser("candidates", help="mine tagged, scored candidate segments")
    p_cand.add_argument("--baseline", required=True)
    p_cand.add_argument("--adapted", required=True)
    p_cand.add_argument("--split", default="test")
    p_cand.add_argument("--data-dir", default="data/processed")
    p_cand.add_argument("--pred-dir", default="results/predictions")
    p_cand.add_argument("--glossary", default="configs/glossary.yaml")
    p_cand.add_argument("--out-dir", default="results/qualitative")
    p_cand.set_defaults(func=cmd_candidates)

    p_render = sub.add_parser("render", help="render curated examples to Markdown")
    p_render.add_argument("--curated", required=True)
    p_render.add_argument("--baseline", default=None, help="defaults to configs/eval.yaml's baseline")
    p_render.add_argument(
        "--adapted", default=None, help="defaults to the first comparisons pair vs. the baseline"
    )
    p_render.add_argument("--eval-config", default="configs/eval.yaml")
    p_render.add_argument("--split", default="test")
    p_render.add_argument("--data-dir", default="data/processed")
    p_render.add_argument("--pred-dir", default="results/predictions")
    p_render.add_argument("--systems-config", default="configs/systems.yaml")
    p_render.add_argument("--out-dir", default="results/qualitative")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
