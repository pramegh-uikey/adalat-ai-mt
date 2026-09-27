"""Tests for adalat_mt.evaluation.analysis (fixtures only, no real predictions)."""

import json
from pathlib import Path

import pytest

from adalat_mt.evaluation import analysis


def test_normalize_hi_drops_nukta_and_maps_chandrabindu():
    # ड़ (DA + nukta) -> ड ; चाँद (chandrabindu) -> चांद (anusvara)
    with_nukta = "ड़"
    assert analysis.normalize_hi(with_nukta) == "ड"

    chandrabindu_word = "चाँद"
    anusvara_word = "चांद"
    assert analysis.normalize_hi(chandrabindu_word) == analysis.normalize_hi(anusvara_word)


def test_normalize_hi_strips_zwj_zwnj_and_collapses_whitespace():
    text = "क‍ख‌ग   घ"
    assert analysis.normalize_hi(text) == "कखग घ"


def test_extract_numbers_ascii_and_devanagari_digits():
    assert analysis.extract_numbers("Order dated 27.05.2003 in case 302/34") == [
        "27.05.2003",
        "302/34",
    ]
    # Devanagari digits १,५,०,०,० -> mapped to ascii and grouped with the comma.
    assert analysis.extract_numbers("राशि १,५०,००० रुपये") == ["1,50,000"]


def test_extract_numbers_standalone_single_digit():
    assert analysis.extract_numbers("clause 5 applies") == ["5"]


def test_number_fidelity_separator_equivalence():
    src = "dated 27.05.2003 and section 302/34"
    hyp_same = "दिनांक 27.05.2003 और धारा 302/34"
    hyp_variant_seps = "दिनांक 27/05/2003 और धारा 302-34"
    hyp_missing = "दिनांक 01.01.2001 और धारा 100"

    assert analysis.number_fidelity(src, hyp_same) == (2, 2)
    assert analysis.number_fidelity(src, hyp_variant_seps) == (2, 2)
    assert analysis.number_fidelity(src, hyp_missing) == (2, 0)


def test_citation_fidelity_preserved_and_missing():
    src = "as held in (2019) 5 SCC 123 and AIR 1990 SC 45"
    hyp_preserved = "जैसा कि (2019) 5 SCC 123 और AIR 1990 SC 45 में कहा गया"
    hyp_partial = "जैसा कि 2019 5 123 में कहा गया"  # only first citation's groups, in order
    hyp_missing = "कुछ और पाठ"

    total, preserved = analysis.citation_fidelity(src, hyp_preserved)
    assert total == 2
    assert preserved == 2

    total, preserved = analysis.citation_fidelity(src, hyp_partial)
    assert total == 2
    assert preserved == 1

    total, preserved = analysis.citation_fidelity(src, hyp_missing)
    assert total == 2
    assert preserved == 0


GLOSSARY = {
    "terms": [
        {"en": "appellants?", "hi": ["अपीलार्थी", "अपीलकर्ता"], "category": "legal-term"},
        {"en": "bail", "hi": ["जमानत", "ज़मानत"], "category": "legal-term"},
    ]
}


def test_term_hits_hit_any_vs_hit_ref():
    src = "The appellant sought bail."
    ref = "अपीलकर्ता ने जमानत मांगी।"

    # hyp uses the *other* accepted variant for "appellant" (hit_any but not hit_ref)
    # and the exact ref variant for "bail" (hit_any and hit_ref).
    hyp = "अपीलार्थी ने जमानत मांगी।"
    hits = {h["term"]: h for h in analysis.term_hits(src, ref, hyp, GLOSSARY)}

    assert hits["appellants?"]["ref_variant"] == "अपीलकर्ता"
    assert hits["appellants?"]["hit_any"] is True
    assert hits["appellants?"]["hit_ref"] is False

    assert hits["bail"]["hit_any"] is True
    assert hits["bail"]["hit_ref"] is True


def test_term_hits_skips_terms_not_in_src_or_ref():
    src = "The respondent objected."
    ref = "प्रत्यर्थी ने आपत्ति की।"
    hyp = "प्रत्यर्थी ने आपत्ति की।"
    # Neither "appellant" nor "bail" appear in src -> no hits.
    assert analysis.term_hits(src, ref, hyp, GLOSSARY) == []


def test_repetition_flag_detects_degenerate_decoding():
    assert analysis.repetition_flag("यह यह यह यह ठीक है") is True
    ngram_repeat = " ".join(["क ख ग घ"] * 3)
    assert analysis.repetition_flag(ngram_repeat) is True
    assert analysis.repetition_flag("यह एक सामान्य वाक्य है") is False


def test_length_ratio_and_omission_addition_flags():
    ref = "क" * 100
    short_hyp = "क" * 50
    long_hyp = "क" * 200
    normal_hyp = "क" * 100

    assert analysis.omission_flag(short_hyp, ref) is True
    assert analysis.addition_flag(long_hyp, ref) is True
    assert analysis.omission_flag(normal_hyp, ref) is False
    assert analysis.addition_flag(normal_hyp, ref) is False


# --------------------------------------------------------------------------
# analyse(): fixture-based end-to-end
# --------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def fixture_dirs(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    pred_dir = tmp_path / "pred"
    data_dir.mkdir()
    pred_dir.mkdir()

    rows = [
        {"pair_id": "p1", "doc_id": "d1"},
        {"pair_id": "p2", "doc_id": "d1"},
        {"pair_id": "p3", "doc_id": "d2"},
    ]
    with (data_dir / "test.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    srcs = [
        "The appellant sought bail under section 302.",
        "This is a short one.",
        "The respondent filed a writ petition.",
    ]
    refs = [
        "अपीलकर्ता ने धारा 302 के तहत जमानत मांगी।",
        "यह एक छोटा वाक्य है।",
        "प्रत्यर्थी ने रिट याचिका दायर की।",
    ]
    baseline_hyps = [
        "अपीलार्थी ने धारा 100 के तहत जमानत मांगी।",  # wrong section number, other term variant
        "यह एक छोटा वाक्य है।",
        "प्रतिवादी ने रिट याचिका दायर की।",
    ]
    adapted_hyps = [
        "अपीलकर्ता ने धारा 302 के तहत जमानत मांगी।",  # correct
        "यह एक छोटा वाक्य है।",
        "प्रत्यर्थी ने रिट याचिका दायर की।",
    ]

    _write_lines(pred_dir / "test.src.en", srcs)
    _write_lines(pred_dir / "test.ref.hi", refs)
    _write_lines(pred_dir / "baseline.test.hi", baseline_hyps)
    _write_lines(pred_dir / "adapted.test.hi", adapted_hyps)

    return data_dir, pred_dir


GLOSSARY_FULL = {
    "terms": [
        {"en": "appellants?", "hi": ["अपीलार्थी", "अपीलकर्ता"], "category": "legal-term"},
        {"en": "bail", "hi": ["जमानत", "ज़मानत"], "category": "legal-term"},
        {"en": "respondents?", "hi": ["प्रत्यर्थी", "प्रतिवादी"], "category": "legal-term"},
        {"en": "writ petitions?", "hi": ["रिट याचिका"], "category": "legal-term"},
    ]
}


def test_analyse_end_to_end(fixture_dirs: tuple[Path, Path]):
    data_dir, pred_dir = fixture_dirs
    cfg_eval = {
        "baseline": "baseline",
        "systems": ["baseline", "adapted"],
        "comparisons": [["baseline", "adapted"]],
    }
    systems_cfg = {"labels": {"baseline": "Baseline", "adapted": "Adapted"}}

    report = analysis.analyse(cfg_eval, systems_cfg, GLOSSARY_FULL, data_dir, pred_dir)

    assert set(report["systems"]) == {"baseline", "adapted"}
    assert report["skipped_systems"] == []

    baseline_stats = report["per_system"]["baseline"]
    adapted_stats = report["per_system"]["adapted"]

    # only the first segment has a number (302); baseline mistranslates it as 100.
    assert baseline_stats["numbers"]["total"] == 1
    assert baseline_stats["numbers"]["preserved"] == 0
    # adapted preserves it.
    assert adapted_stats["numbers"]["preserved"] == 1

    cmp_stats = report["comparisons"]["baseline_vs_adapted"]
    assert cmp_stats["overall"]["win"] + cmp_stats["overall"]["tie"] + cmp_stats["overall"]["loss"] == 3
    # adapted segment 1 fixes the term+number -> should not be a loss for adapted.
    assert cmp_stats["overall"]["loss"] == 0


def test_analyse_skips_missing_system(fixture_dirs: tuple[Path, Path]):
    data_dir, pred_dir = fixture_dirs
    cfg_eval = {
        "baseline": "baseline",
        "systems": ["baseline", "adapted", "ghost"],
        "comparisons": [["baseline", "ghost"]],
    }
    systems_cfg = {"labels": {}}

    report = analysis.analyse(cfg_eval, systems_cfg, GLOSSARY_FULL, data_dir, pred_dir)
    assert "ghost" in report["skipped_systems"]
    assert "ghost" not in report["per_system"]
    assert report["comparisons"] == {}  # comparison involving missing system dropped


def test_render_markdown_smoke(fixture_dirs: tuple[Path, Path]):
    data_dir, pred_dir = fixture_dirs
    cfg_eval = {
        "baseline": "baseline",
        "systems": ["baseline", "adapted"],
        "comparisons": [["baseline", "adapted"]],
    }
    systems_cfg = {"systems": {"baseline": {"label": "Baseline"}, "adapted": {"label": "Adapted"}}}
    report = analysis.analyse(cfg_eval, systems_cfg, GLOSSARY_FULL, data_dir, pred_dir)
    md = analysis.render_markdown(report, systems_cfg)
    assert "# Error analysis" in md
    assert "Adapted vs Baseline" in md


def test_repetition_flag_catches_character_runs() -> None:
    from adalat_mt.evaluation.analysis import repetition_flag

    assert repetition_flag("19.08.1992" + "." * 40)
    assert not repetition_flag("दिनांक 19.08.1992 का आदेश...")


def test_length_robustness_subsets() -> None:
    from adalat_mt.evaluation.analysis import length_robustness

    srcs = ["Leave granted.", "the appeal is dismissed with costs today by the court here"]
    refs = ["अनुमति प्रदान की गई।", "अपील आज खारिज की जाती है"]
    out = length_robustness(srcs, refs, {"a": refs, "b": ["x", "अपील आज खारिज की जाती है"]}, min_words=(0, 10))
    assert [r["n_segments"] for r in out] == [2, 1]
    assert out[1]["chrf"]["a"] == out[1]["chrf"]["b"] == 100.0
