"""Tests for adalat_mt.evaluation.qualitative (fixtures only, no real predictions)."""

import json
from pathlib import Path

import pytest

from adalat_mt.evaluation import qualitative


GLOSSARY = {
    "terms": [
        {"en": "appellants?", "hi": ["अपीलार्थी", "अपीलकर्ता"], "category": "legal-term"},
        {"en": "bail", "hi": ["जमानत"], "category": "legal-term"},
    ]
}


def test_tag_segment_legal_term():
    src = "The appellant sought bail."
    ref = "अपीलकर्ता ने जमानत मांगी।"
    hyps = {"sys": "अपीलकर्ता ने जमानत मांगी।"}
    assert "legal-term" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_archaic_formulaic():
    src = "The learned counsel submitted that notwithstanding the above, the said order stands."
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "archaic/formulaic" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_long_sentence():
    src = " ".join(["word"] * 46)
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "long-sentence" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_not_long_sentence_at_boundary():
    src = " ".join(["word"] * 45)
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "long-sentence" not in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_citation_section_date():
    src_date = "The order dated 27.05.2003 was set aside."
    src_section = "This appeal arises under Section 302 of the IPC."
    src_citation = "The court in (2019) 5 SCC 123 held as follows."
    ref = "ref"
    hyps = {"sys": "hyp"}
    for src in (src_date, src_section, src_citation):
        assert "citation/section/date" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_named_entity_consecutive_caps():
    src = "This appeal was heard by Justice Ram Kumar Sharma on merits."
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "named-entity" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_named_entity_case_name():
    src = "The principle laid down in Union of India v. Raghubir Singh applies here."
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "named-entity" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_no_named_entity_for_stoplisted_words():
    src = "The High Court dismissed the appeal."
    ref = "ref"
    hyps = {"sys": "hyp"}
    assert "named-entity" not in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_omission_hallucination():
    src = "A short source sentence."
    ref = "यह एक संदर्भ वाक्य है जो पर्याप्त रूप से लंबा है ताकि लंबाई अनुपात की जांच हो सके।"
    short_hyp = "छोटा।"
    hyps = {"sys": short_hyp}
    assert "omission/hallucination" in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


def test_tag_segment_no_omission_when_hyp_normal_length():
    src = "A short source sentence."
    ref = "यह एक सामान्य लंबाई वाला संदर्भ वाक्य है।"
    hyp = "यह एक सामान्य लंबाई वाला अनुवाद वाक्य है।"
    hyps = {"sys": hyp}
    assert "omission/hallucination" not in qualitative.tag_segment(src, ref, hyps, GLOSSARY)


# --------------------------------------------------------------------------
# render(): fixture-based, checks validation + headings
# --------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def prediction_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    pred_dir = tmp_path / "pred"
    data_dir.mkdir()
    pred_dir.mkdir()

    n = 12
    rows = [{"pair_id": f"p{i}", "doc_id": "d1"} for i in range(n)]
    with (data_dir / "test.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    srcs = [f"Source sentence number {i} about appellant bail." for i in range(n)]
    refs = [f"संदर्भ वाक्य संख्या {i} अपीलकर्ता जमानत के बारे में।" for i in range(n)]
    baseline_hyps = [f"आधार अनुवाद संख्या {i}।" for i in range(n)]
    adapted_hyps = [f"अनुकूलित अनुवाद संख्या {i} अपीलकर्ता जमानत के बारे में।" for i in range(n)]

    _write_lines(pred_dir / "test.src.en", srcs)
    _write_lines(pred_dir / "test.ref.hi", refs)
    _write_lines(pred_dir / "baseline.test.hi", baseline_hyps)
    _write_lines(pred_dir / "adapted.test.hi", adapted_hyps)

    return data_dir, pred_dir


def _curated_with_categories(pair_ids: list[str]) -> dict:
    categories_cycle = list(qualitative.CATEGORIES)
    examples = []
    for i, pid in enumerate(pair_ids):
        cat = categories_cycle[i % len(categories_cycle)]
        examples.append({"pair_id": pid, "categories": [cat], "analysis": f"Analysis for {pid}."})
    return {"intro": "Intro text.", "critic_summary": "Critic said it looks fine.", "examples": examples}


def test_render_examples_produces_expected_headings(prediction_fixture: tuple[Path, Path]):
    data_dir, pred_dir = prediction_fixture
    pair_ids = [f"p{i}" for i in range(12)]
    curated = _curated_with_categories(pair_ids)
    records = qualitative._load_records(data_dir, pred_dir, "baseline", "adapted")

    markdown, jsonl_rows = qualitative.render_examples(curated, records, "Baseline", "Adapted")

    assert "### Example 1 —" in markdown
    assert "### Example 12 —" in markdown
    assert "**Source (EN)**" in markdown
    assert "**Reference (HI)**" in markdown
    assert "**Baseline (Baseline)**" in markdown
    assert "**Adapted (Adapted)**" in markdown
    assert "**Analysis.**" in markdown
    assert "Second opinion (LLM judgement — not a metric)" in markdown
    assert "Intro text." in markdown
    assert len(jsonl_rows) == 12
    assert set(jsonl_rows[0].keys()) == {
        "id",
        "source_en",
        "reference_hi",
        "baseline_hi",
        "adapted_hi",
    }


def test_render_examples_raises_on_missing_category(prediction_fixture: tuple[Path, Path]):
    data_dir, pred_dir = prediction_fixture
    pair_ids = [f"p{i}" for i in range(12)]
    # Only ever use "legal-term" -> other five categories missing.
    examples = [
        {"pair_id": pid, "categories": ["legal-term"], "analysis": "x"} for pid in pair_ids
    ]
    curated = {"examples": examples}
    records = qualitative._load_records(data_dir, pred_dir, "baseline", "adapted")

    with pytest.raises(ValueError, match="categories"):
        qualitative.render_examples(curated, records, "Baseline", "Adapted")


def test_render_examples_raises_on_too_few_examples(prediction_fixture: tuple[Path, Path]):
    data_dir, pred_dir = prediction_fixture
    pair_ids = [f"p{i}" for i in range(5)]
    curated = _curated_with_categories(pair_ids)
    records = qualitative._load_records(data_dir, pred_dir, "baseline", "adapted")

    with pytest.raises(ValueError, match="12"):
        qualitative.render_examples(curated, records, "Baseline", "Adapted")


def test_build_candidates_sorted_by_abs_delta(prediction_fixture: tuple[Path, Path]):
    data_dir, pred_dir = prediction_fixture
    candidates = qualitative.build_candidates(data_dir, pred_dir, "baseline", "adapted", GLOSSARY)
    assert len(candidates) == 12
    deltas = [abs(c["delta"]) for c in candidates]
    assert deltas == sorted(deltas, reverse=True)
    for c in candidates:
        assert set(
            {
                "pair_id",
                "doc_id",
                "tags",
                "source_en",
                "reference_hi",
                "baseline_hi",
                "adapted_hi",
                "chrf_baseline",
                "chrf_adapted",
                "delta",
                "others",
            }
        ).issubset(c.keys())
