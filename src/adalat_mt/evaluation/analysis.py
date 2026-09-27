"""Error-analysis tooling for EN->HI legal MT outputs.

Computes number/citation fidelity, glossary-term hit rates, degenerate-decoding
(repetition) flags, omission/addition flags, and pairwise sentence-chrF++
win/tie/loss statistics between systems. Pure CPU, no model downloads.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import sacrebleu
import yaml

from adalat_mt.evaluation.metrics import sentence_chrf as _batch_sentence_chrf


def _sentence_chrf(hyp: str, ref: str) -> float:
    """Sentence-level chrF++ (chrF2, word_order=2), via the shared metrics module."""
    return _batch_sentence_chrf([hyp], [ref])[0]


def sentence_chrf(hyp: str, ref: str) -> float:
    """Public wrapper around the sentence-level chrF++ scorer."""
    return _sentence_chrf(hyp, ref)


# --------------------------------------------------------------------------
# Text normalisation and low-level fidelity primitives
# --------------------------------------------------------------------------

_NUKTA = "़"
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"
_ZW_CHARS = "‌‍"  # ZWNJ, ZWJ

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_WS_RE = re.compile(r"\s+")


def normalize_hi(text: str) -> str:
    """Normalise Hindi text for robust string matching.

    NFC-normalises, drops the nukta (U+093C), maps chandrabindu (U+0901) to
    anusvara (U+0902), strips ZWJ/ZWNJ, and collapses whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_NUKTA, "")
    text = text.replace(_CHANDRABINDU, _ANUSVARA)
    for ch in _ZW_CHARS:
        text = text.replace(ch, "")
    text = _WS_RE.sub(" ", text).strip()
    return text


_NUMBER_RE = re.compile(r"\d+(?:[./,\-]\d+)*")


def extract_numbers(text: str) -> list[str]:
    """Extract digit runs, including internal ``.``/``/``/``-``/``,`` separators.

    Devanagari digits are mapped to ASCII first. Standalone single digits count
    too (e.g. an isolated ``5``).
    """
    ascii_text = text.translate(_DEVANAGARI_DIGITS)
    return _NUMBER_RE.findall(ascii_text)


def _digit_groups(number: str) -> tuple[str, ...]:
    """Split a matched number into its constituent digit groups."""
    return tuple(g for g in re.split(r"[^\d]", number) if g)


def number_fidelity(src: str, hyp: str) -> tuple[int, int]:
    """Count numbers in ``src`` and how many are preserved verbatim in ``hyp``.

    Preservation is judged on digit groups: ``27.05.2003`` is considered the
    same number as ``27/05/2003`` or ``27-05-2003``.
    """
    src_nums = extract_numbers(src)
    hyp_groups = Counter(_digit_groups(n) for n in extract_numbers(hyp))
    preserved = 0
    for n in src_nums:
        groups = _digit_groups(n)
        if hyp_groups[groups] > 0:
            hyp_groups[groups] -= 1
            preserved += 1
    return len(src_nums), preserved


_CITATION_PATTERNS = [
    re.compile(r"\(\d{4}\)\s*\d+\s*[A-Z][A-Za-z.]{0,8}\s*\d+"),  # (2019) 5 SCC 123
    re.compile(r"AIR\s*\d{4}\s*[A-Z]{1,5}\s*\d+"),  # AIR 1990 SC 123
]


def find_citations(text: str) -> list[str]:
    """Find case-citation substrings such as ``(2019) 5 SCC 123`` or ``AIR 1990 SC 123``."""
    ascii_text = text.translate(_DEVANAGARI_DIGITS)
    matches: list[str] = []
    for pat in _CITATION_PATTERNS:
        matches.extend(m.group(0) for m in pat.finditer(ascii_text))
    return matches


def citation_fidelity(src: str, hyp: str) -> tuple[int, int]:
    """Count case citations in ``src`` and how many are preserved in ``hyp``.

    A citation is preserved when all of its digit groups appear, in order,
    somewhere in ``hyp`` (not necessarily contiguous).
    """
    citations = find_citations(src)
    hyp_ascii = hyp.translate(_DEVANAGARI_DIGITS)
    preserved = 0
    for cit in citations:
        groups = re.findall(r"\d+", cit)
        pos = 0
        ok = True
        for g in groups:
            idx = hyp_ascii.find(g, pos)
            if idx == -1:
                ok = False
                break
            pos = idx + len(g)
        if ok:
            preserved += 1
    return len(citations), preserved


def _compile_term(en_pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{en_pattern})\b", re.IGNORECASE)


def term_hits(src: str, ref: str, hyp: str, glossary: dict[str, Any]) -> list[dict[str, Any]]:
    """Glossary-term hit report for one segment.

    For every glossary term whose EN regex matches ``src`` (case-insensitive,
    word-bounded) *and* whose HI variant occurs in ``ref``, report whether the
    hypothesis contains any accepted variant (``hit_any``) or the specific
    variant used in the reference (``hit_ref``).
    """
    norm_ref = normalize_hi(ref)
    norm_hyp = normalize_hi(hyp)
    hits: list[dict[str, Any]] = []
    for term in glossary.get("terms", []):
        en_re = _compile_term(term["en"])
        if not en_re.search(src):
            continue
        ref_variant = None
        for variant in term["hi"]:
            if normalize_hi(variant) in norm_ref:
                ref_variant = variant
                break
        if ref_variant is None:
            continue
        hit_any = any(normalize_hi(v) in norm_hyp for v in term["hi"])
        hit_ref = normalize_hi(ref_variant) in norm_hyp
        hits.append(
            {
                "term": term["en"],
                "category": term["category"],
                "ref_variant": ref_variant,
                "hit_any": hit_any,
                "hit_ref": hit_ref,
            }
        )
    return hits


_CHAR_RUN_RE = re.compile(r"(\S)\1{9,}")


def repetition_flag(hyp: str) -> bool:
    """True if ``hyp`` shows signs of degenerate-decoding repetition.

    Triggers when any word 4-gram repeats >= 3 times, a single token
    repeats >= 4 times in a row, or any character (e.g. ``.``) repeats >= 10
    times in a row (degenerate output such as ``19.08.1992.......``).
    """
    if _CHAR_RUN_RE.search(hyp):
        return True
    tokens = hyp.split()
    if len(tokens) >= 4:
        for i in range(len(tokens) - 3):
            if tokens[i] == tokens[i + 1] == tokens[i + 2] == tokens[i + 3]:
                return True
    if len(tokens) >= 4:
        grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
        counts = Counter(grams)
        if any(c >= 3 for c in counts.values()):
            return True
    return False


def length_ratio(hyp: str, ref: str) -> float:
    """Ratio of non-whitespace character counts, ``hyp`` over ``ref``."""
    hyp_len = len(_WS_RE.sub("", hyp))
    ref_len = len(_WS_RE.sub("", ref))
    if ref_len == 0:
        return 1.0 if hyp_len == 0 else float("inf")
    return hyp_len / ref_len


def omission_flag(hyp: str, ref: str) -> bool:
    """True when ``hyp`` is much shorter than ``ref`` (ratio < 0.6)."""
    return length_ratio(hyp, ref) < 0.6


def addition_flag(hyp: str, ref: str) -> bool:
    """True when ``hyp`` is much longer than ``ref`` (ratio > 1.6)."""
    return length_ratio(hyp, ref) > 1.6


# --------------------------------------------------------------------------
# Aggregate analysis
# --------------------------------------------------------------------------

_LEN_BUCKETS = ("<=20", "21-40", ">40")


def _length_bucket(n_words: int) -> str:
    if n_words <= 20:
        return "<=20"
    if n_words <= 40:
        return "21-40"
    return ">40"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _rate(preserved: int, total: int) -> float | None:
    return (preserved / total) if total else None


def _system_hyp_path(pred_dir: Path, system: str, split: str) -> Path:
    return pred_dir / f"{system}.{split}.hi"


def _load_system(pred_dir: Path, system: str, n_rows: int, split: str) -> list[str] | None:
    path = _system_hyp_path(pred_dir, system, split)
    if not path.exists():
        return None
    lines = read_lines(path)
    if len(lines) != n_rows:
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


def _per_system_stats(
    srcs: list[str], refs: list[str], hyps: list[str], glossary: dict[str, Any]
) -> dict[str, Any]:
    n_total = n_preserved = 0
    c_total = c_preserved = 0
    term_n = term_hit_any = term_hit_ref = 0
    by_cat: dict[str, dict[str, int]] = {}
    repetition_segments = omission_segments = addition_segments = 0
    length_ratio_sum = 0.0

    for src, ref, hyp in zip(srcs, refs, hyps):
        nt, npv = number_fidelity(src, hyp)
        n_total += nt
        n_preserved += npv
        ct, cpv = citation_fidelity(src, hyp)
        c_total += ct
        c_preserved += cpv
        for hit in term_hits(src, ref, hyp, glossary):
            term_n += 1
            term_hit_any += int(hit["hit_any"])
            term_hit_ref += int(hit["hit_ref"])
            cat = by_cat.setdefault(hit["category"], {"n": 0, "hit_any": 0, "hit_ref": 0})
            cat["n"] += 1
            cat["hit_any"] += int(hit["hit_any"])
            cat["hit_ref"] += int(hit["hit_ref"])
        if repetition_flag(hyp):
            repetition_segments += 1
        if omission_flag(hyp, ref):
            omission_segments += 1
        if addition_flag(hyp, ref):
            addition_segments += 1
        length_ratio_sum += length_ratio(hyp, ref)

    by_category = {
        cat: {
            "n": v["n"],
            "hit_any_rate": _rate(v["hit_any"], v["n"]),
            "hit_ref_rate": _rate(v["hit_ref"], v["n"]),
        }
        for cat, v in by_cat.items()
    }

    n = len(srcs)
    return {
        "numbers": {"total": n_total, "preserved": n_preserved, "rate": _rate(n_preserved, n_total)},
        "citations": {"total": c_total, "preserved": c_preserved, "rate": _rate(c_preserved, c_total)},
        "terms": {
            "n": term_n,
            "hit_any_rate": _rate(term_hit_any, term_n),
            "hit_ref_rate": _rate(term_hit_ref, term_n),
            "by_category": by_category,
        },
        "repetition_segments": repetition_segments,
        "omission_segments": omission_segments,
        "addition_segments": addition_segments,
        "mean_length_ratio": (length_ratio_sum / n) if n else None,
    }


_TIE_BAND = 2.0


def _compare_systems(
    a: str,
    b: str,
    rows: list[dict[str, Any]],
    srcs: list[str],
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    glossary: dict[str, Any],
) -> dict[str, Any]:
    overall = {"win": 0, "tie": 0, "loss": 0}
    by_bucket = {bucket: {"win": 0, "tie": 0, "loss": 0} for bucket in _LEN_BUCKETS}
    by_category: dict[str, dict[str, int]] = {}
    deltas: list[tuple[str, float]] = []

    for row, src, ref, hyp_a, hyp_b in zip(rows, srcs, refs, hyps_a, hyps_b):
        chrf_a = _sentence_chrf(hyp_a, ref)
        chrf_b = _sentence_chrf(hyp_b, ref)
        delta = chrf_b - chrf_a
        if delta > _TIE_BAND:
            outcome = "win"
        elif delta < -_TIE_BAND:
            outcome = "loss"
        else:
            outcome = "tie"
        overall[outcome] += 1
        bucket = _length_bucket(len(src.split()))
        by_bucket[bucket][outcome] += 1

        for hit in term_hits(src, ref, hyp_b, glossary):
            cat_counts = by_category.setdefault(hit["category"], {"win": 0, "tie": 0, "loss": 0})
            cat_counts[outcome] += 1

        deltas.append((row.get("pair_id", ""), delta))

    gains = sorted(deltas, key=lambda t: t[1], reverse=True)[:10]
    losses = sorted(deltas, key=lambda t: t[1])[:10]

    return {
        "overall": overall,
        "by_source_length": by_bucket,
        "by_term_category": by_category,
        "top_gains": [{"pair_id": pid, "delta": d} for pid, d in gains],
        "top_losses": [{"pair_id": pid, "delta": d} for pid, d in losses],
    }


def analyse(
    cfg_eval: dict[str, Any],
    systems_cfg: dict[str, Any],
    glossary: dict[str, Any],
    data_dir: str | Path,
    pred_dir: str | Path,
) -> dict[str, Any]:
    """Compute the full error-analysis report.

    ``cfg_eval`` provides ``split`` (defaults to ``"test"``), ``baseline``,
    ``systems`` (list of system ids to analyse) and ``comparisons`` (list of
    ``[a, b]`` system-id pairs) — the same shape as ``configs/eval.yaml``.
    ``systems_cfg`` provides display labels (``configs/systems.yaml`` shape:
    ``{"systems": {id: {"label": ...}}}``), ``glossary`` is the loaded
    glossary dict. ``data_dir`` holds ``<split>.jsonl`` (row order gives
    ``pair_id``/``doc_id``); ``pred_dir`` holds ``<split>.src.en``,
    ``<split>.ref.hi`` and one ``<system>.<split>.hi`` per system. Systems
    whose hypothesis file is missing or misaligned are skipped.
    """
    data_dir = Path(data_dir)
    pred_dir = Path(pred_dir)
    split = cfg_eval.get("split", "test")

    srcs = read_lines(pred_dir / f"{split}.src.en")
    refs = read_lines(pred_dir / f"{split}.ref.hi")
    n = len(refs)
    rows = _load_rows(data_dir, split, n)

    systems = cfg_eval.get("systems", [])
    loaded: dict[str, list[str]] = {}
    skipped: list[str] = []
    for system in systems:
        hyps = _load_system(pred_dir, system, n, split)
        if hyps is None:
            skipped.append(system)
            continue
        loaded[system] = hyps

    per_system = {
        system: _per_system_stats(srcs, refs, hyps, glossary) for system, hyps in loaded.items()
    }

    comparisons: dict[str, Any] = {}
    for pair in cfg_eval.get("comparisons", []):
        a, b = pair[0], pair[1]
        if a not in loaded or b not in loaded:
            continue
        comparisons[f"{a}_vs_{b}"] = _compare_systems(
            a, b, rows, srcs, refs, loaded[a], loaded[b], glossary
        )

    return {
        "systems": list(loaded.keys()),
        "skipped_systems": skipped,
        "per_system": per_system,
        "comparisons": comparisons,
        "robustness": length_robustness(srcs, refs, loaded),
    }


def length_robustness(
    srcs: list[str], refs: list[str], hyps_by_system: dict[str, list[str]], min_words: tuple[int, ...] = (0, 5, 10)
) -> list[dict[str, Any]]:
    """Corpus chrF++ per system on subsets with at least ``min_words`` source words.

    Guards against a corpus-level gain that comes only from a few short formulaic segments ("Leave granted.").
    """
    chrf = sacrebleu.CHRF(word_order=2)
    out = []
    for n in min_words:
        idx = [i for i, src in enumerate(srcs) if len(src.split()) >= n]
        if not idx:
            continue
        ref_sub = [refs[i] for i in idx]
        scores = {
            system: round(chrf.corpus_score([hyps[i] for i in idx], [ref_sub]).score, 2)
            for system, hyps in hyps_by_system.items()
        }
        out.append({"min_src_words": n, "n_segments": len(idx), "chrf": scores})
    return out


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _label(systems_cfg: dict[str, Any], system: str) -> str:
    """Look up a system's display label from a ``configs/systems.yaml``-shaped dict."""
    systems = (systems_cfg or {}).get("systems", {})
    return systems.get(system, {}).get("label", system)


def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def render_markdown(report: dict[str, Any], systems_cfg: dict[str, Any]) -> str:
    """Render the error-analysis report as readable Markdown tables."""
    lines: list[str] = ["# Error analysis", ""]

    if report.get("skipped_systems"):
        lines.append(f"_Skipped (missing/misaligned hypotheses): {', '.join(report['skipped_systems'])}_")
        lines.append("")

    lines.append("## Per-system fidelity")
    lines.append("")
    lines.append(
        "| System | Numbers | Citations | Terms (any/ref) | Repetition | Omission | Addition | Mean length ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for system, stats in report.get("per_system", {}).items():
        lines.append(
            "| {label} | {num} | {cit} | {term_any}/{term_ref} | {rep} | {om} | {add} | {ratio:.2f} |".format(
                label=_label(systems_cfg, system),
                num=_fmt_rate(stats["numbers"]["rate"]),
                cit=_fmt_rate(stats["citations"]["rate"]),
                term_any=_fmt_rate(stats["terms"]["hit_any_rate"]),
                term_ref=_fmt_rate(stats["terms"]["hit_ref_rate"]),
                rep=stats["repetition_segments"],
                om=stats["omission_segments"],
                add=stats["addition_segments"],
                ratio=stats["mean_length_ratio"] or 0.0,
            )
        )
    lines.append("")

    robustness = report.get("robustness", [])
    if robustness:
        systems = list(robustness[0]["chrf"])
        lines += ["## Robustness: corpus chrF++ on segments with >= N source words", ""]
        lines.append("| N | segments | " + " | ".join(_label(systems_cfg, x) for x in systems) + " |")
        lines.append("|---|---|" + "---|" * len(systems))
        for r in robustness:
            lines.append(f"| {r['min_src_words']} | {r['n_segments']} | " + " | ".join(
                f"{r['chrf'][x]:.2f}" for x in systems) + " |")
        lines.append("")

    for cmp_name, cmp_stats in report.get("comparisons", {}).items():
        a, b = cmp_name.rsplit("_vs_", 1)
        lines.append(f"## {_label(systems_cfg, b)} vs {_label(systems_cfg, a)}")
        lines.append("")
        o = cmp_stats["overall"]
        lines.append(f"Overall: win={o['win']} tie={o['tie']} loss={o['loss']}")
        lines.append("")
        lines.append("| Source length | Win | Tie | Loss |")
        lines.append("|---|---|---|---|")
        for bucket, counts in cmp_stats["by_source_length"].items():
            lines.append(f"| {bucket} | {counts['win']} | {counts['tie']} | {counts['loss']} |")
        lines.append("")
        if cmp_stats["by_term_category"]:
            lines.append("| Term category | Win | Tie | Loss |")
            lines.append("|---|---|---|---|")
            for cat, counts in cmp_stats["by_term_category"].items():
                lines.append(f"| {cat} | {counts['win']} | {counts['tie']} | {counts['loss']} |")
            lines.append("")
        lines.append("Top gains: " + ", ".join(f"{g['pair_id']} ({g['delta']:+.1f})" for g in cmp_stats["top_gains"]))
        lines.append("")
        lines.append("Top losses: " + ", ".join(f"{g['pair_id']} ({g['delta']:+.1f})" for g in cmp_stats["top_losses"]))
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m adalat_mt.evaluation.analysis --eval-config configs/eval.yaml``."""
    parser = argparse.ArgumentParser(description="Error analysis for EN->HI MT outputs")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--systems-config", default="configs/systems.yaml")
    parser.add_argument("--glossary", default="configs/glossary.yaml")
    parser.add_argument("--data-dir", default=None, help="defaults to configs/eval.yaml's data_dir or data/processed")
    parser.add_argument("--pred-dir", default=None, help="defaults to configs/eval.yaml's pred_dir")
    parser.add_argument("--out-dir", default="results/eval")
    args = parser.parse_args(argv)

    cfg_eval = yaml.safe_load(Path(args.eval_config).read_text(encoding="utf-8")) or {}
    systems_cfg_path = Path(args.systems_config)
    systems_cfg = yaml.safe_load(systems_cfg_path.read_text(encoding="utf-8")) if systems_cfg_path.exists() else {}
    glossary = yaml.safe_load(Path(args.glossary).read_text(encoding="utf-8")) or {}

    data_dir = args.data_dir or cfg_eval.get("data_dir", "data/processed")
    pred_dir = args.pred_dir or cfg_eval.get("pred_dir", "results/predictions")

    report = analyse(cfg_eval, systems_cfg or {}, glossary, data_dir, pred_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "error_analysis.md").write_text(render_markdown(report, systems_cfg or {}), encoding="utf-8")


if __name__ == "__main__":
    main()
