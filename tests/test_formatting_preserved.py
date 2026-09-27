"""Assert that segmentation never alters legal-formatting substrings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adalat_mt.data.segment import segment_document, unwrap
from adalat_mt.data.text_clean import clean

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _strip_ws(text: str) -> str:
    return "".join(text.split())


_EN_TEXT = """1. The court in (2019) 5 SCC 123 considered Section 331 of the U.P.
Zamidari Abolition and Land Reforms Act, 1950 in some detail. The order
dated 27.05.2003 was set aside by the High Court. Clause (ii) and clause
(a) were examined together by the Bench. Para 2. dealt with the question
of limitation. The court quoted the statutory sentence: "the suit shall
not be maintainable in a civil court" while dismissing the appeal.
"""

_EN_SUBSTRINGS = [
    "(2019) 5 SCC 123",
    "Section 331 of the U.P. Zamidari Abolition and Land Reforms Act, 1950",
    "27.05.2003",
    "(ii)",
    "(a)",
    "2.",
    '"the suit shall not be maintainable in a civil court"',
]

_HI_TEXT = """1. न्यायालय ने माना कि धारा 331 के अंतर्गत यह मामला पोषणीय नहीं
था। (2019) 1 एससीसी 207 में दिए गए निर्णय का उल्लेख किया गया। आदेश
दिनांक 22.05.2014 को पारित हुआ था।
"""

_HI_SUBSTRINGS = [
    "धारा 331",
    "(2019) 1 एससीसी 207",
    "22.05.2014",
]


def test_en_formatting_preserved() -> None:
    sentences = segment_document(_EN_TEXT, "en")
    joined = " ".join(sentences)
    for substr in _EN_SUBSTRINGS:
        assert substr in joined, f"missing/altered: {substr!r} in {joined!r}"


def test_hi_formatting_preserved() -> None:
    sentences = segment_document(_HI_TEXT, "hi")
    joined = " ".join(sentences)
    for substr in _HI_SUBSTRINGS:
        assert substr in joined, f"missing/altered: {substr!r} in {joined!r}"


def test_processed_rows_are_substrings_of_cleaned_source() -> None:
    processed_dir = _REPO_ROOT / "data" / "processed"
    if not processed_dir.exists():
        pytest.skip("data/processed/ not built yet")

    from adalat_mt.data.text_clean import read_text

    from adalat_mt.data.sources import load_data_config, source_path

    data_cfg = load_data_config(_REPO_ROOT / "configs" / "data.yaml")
    cleaned_unwrapped_cache: dict[tuple[str, str], str] = {}

    def cleaned_unwrapped(doc_id: str, lang: str) -> str:
        key = (doc_id, lang)
        if key not in cleaned_unwrapped_cache:
            raw = read_text(source_path(doc_id, lang, data_cfg, root=_REPO_ROOT))
            segs = unwrap(clean(raw, lang))
            cleaned_unwrapped_cache[key] = _strip_ws(" ".join(segs))
        return cleaned_unwrapped_cache[key]

    checked = 0
    for split_name in ("train", "dev", "test"):
        path = processed_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc_id = row["doc_id"]
                # Whitespace is ignored: joining a multi-sentence block may add a space after a danda that had none.
                assert _strip_ws(row["en"]) in cleaned_unwrapped(doc_id, "en")
                assert _strip_ws(row["hi"]) in cleaned_unwrapped(doc_id, "hi")
                checked += 1
    if checked == 0:
        pytest.skip("no processed rows found")
