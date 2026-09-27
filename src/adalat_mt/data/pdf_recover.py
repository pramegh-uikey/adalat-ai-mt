"""Recover Hindi text for documents whose provided ``clean`` text lost all Devanagari.

Five Hindi ``clean`` files (6, 14, 22, 25, 26) contain only ``?`` characters. Their PDFs have a text layer, but it was
produced from legacy (KrutiDev-style) glyphs and is systematically garbled (``भार`` for ``भारत``, ``सिसविवल`` for
``सिविल``). The recovery therefore combines two sources:

* **Words** come from Tesseract OCR (``hin`` model, 300 dpi page renders), which yields well-formed Devanagari.
* **Numbers** come from the PDF text layer, where digits are plain ASCII and exact. Tesseract's Hindi model
  systematically drops or misreads the digit ``1`` (``1989`` → ``989``/``4989``), so every OCR number is aligned to the
  text-layer number sequence of the same page and replaced by its match.

Page-footer disclaimers ("उद्घोषणा"/"अस्वीकरण" …) are removed. Requires the system packages ``tesseract-ocr`` and
``tesseract-ocr-hin``; the recovered texts are committed under ``data/recovered/`` so the rest of the pipeline does not
need OCR.

Usage::

    python -m adalat_mt.data.pdf_recover --ids 6 14 22 25 26 [--dataset-dir dataset] [--out-dir data/recovered/hindi]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

NUMBER_RE = re.compile(r"\d+(?:[./-]+\d+)*")  # `+`: OCR can drop a whole digit group ("23..2009")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
#: A page's footer disclaimer starts at the first line containing one of these fragments (prefixes, because OCR
#: sometimes garbles the heading itself, e.g. "उद्घोष्णा").
_DISCLAIMER_START = ("उद्घो", "अस्वीकर", "भाषा में अनुवादित निर्णय")
_CONFUSABLE = str.maketrans("", "", "14")


@dataclass
class PageRecovery:
    """Recovered text of one page plus number-repair bookkeeping."""

    text: str
    ocr_numbers: int = 0
    repaired: list[tuple[str, str]] = field(default_factory=list)


def number_key(num: str) -> str:
    """Alignment key that is invariant to Tesseract's 1-dropping / 1→4 confusion."""
    return num.translate(_CONFUSABLE)


def repair_numbers(ocr_text: str, layer_numbers: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Replace OCR numbers by the text-layer numbers they align to.

    The two number sequences are aligned with ``difflib`` on :func:`number_key`; numbers in ``equal`` blocks are
    replaced one-to-one, and in equal-length ``replace`` blocks when the strings are at least 50% similar. Unaligned OCR
    numbers are kept. Returns the repaired text and the list of (ocr, layer) substitutions that changed something.
    """
    ocr_text = ocr_text.translate(_DEVANAGARI_DIGITS)
    matches = list(NUMBER_RE.finditer(ocr_text))
    ocr_nums = [m.group() for m in matches]
    sm = difflib.SequenceMatcher(None, [number_key(n) for n in ocr_nums], [number_key(n) for n in layer_numbers],
                                 autojunk=False)
    replacement: dict[int, str] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                similar = difflib.SequenceMatcher(None, ocr_nums[i], layer_numbers[j]).ratio() >= 0.5
                if tag == "equal" or similar:
                    replacement[i] = layer_numbers[j]
    out, last, changes = [], 0, []
    for i, m in enumerate(matches):
        new = replacement.get(i, m.group())
        if new != m.group():
            changes.append((m.group(), new))
        out.append(ocr_text[last : m.start()] + new)
        last = m.end()
    out.append(ocr_text[last:])
    return "".join(out), changes


def strip_disclaimer(page_text: str) -> str:
    """Drop the page footer disclaimer (from its first line to the end of the page)."""
    lines = page_text.splitlines()
    for k, line in enumerate(lines):
        if any(word in line for word in _DISCLAIMER_START):
            return "\n".join(lines[:k])
    return page_text


def ocr_image(png: Path, lang: str = "hin", psm: int = 6) -> str:
    """Run the Tesseract CLI on one image and return its text."""
    res = subprocess.run(["tesseract", str(png), "-", "-l", lang, "--psm", str(psm)],
                         capture_output=True, text=True, check=True)
    return res.stdout


def recover_page(page, dpi: int = 300) -> PageRecovery:
    """OCR one PyMuPDF page, repair its numbers from the text layer and strip the disclaimer."""
    layer_numbers = NUMBER_RE.findall(page.get_text().translate(_DEVANAGARI_DIGITS))
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        page.get_pixmap(dpi=dpi).save(png)
        ocr = ocr_image(png)
    ocr = strip_disclaimer(ocr)
    text, changes = repair_numbers(ocr, layer_numbers)
    return PageRecovery(text=text, ocr_numbers=len(NUMBER_RE.findall(text)), repaired=changes)


def recover_document(pdf: Path, dpi: int = 300) -> tuple[str, dict]:
    """Recover a whole Hindi judgment PDF; returns its text and a small report dict."""
    import pymupdf

    doc = pymupdf.open(pdf)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(lambda p: recover_page(p, dpi), doc))
    text = "\n".join(p.text.rstrip() for p in pages) + "\n"
    report = {"pdf": str(pdf), "pages": len(pages), "numbers": sum(p.ocr_numbers for p in pages),
              "numbers_repaired": sum(len(p.repaired) for p in pages),
              "repair_examples": [c for p in pages for c in p.repaired][:10]}
    return text, report


def main(argv: list[str] | None = None) -> None:
    """CLI: recover the given document ids into ``<out-dir>/<id>.txt`` plus ``recovery_report.json``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/recovered/hindi"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for doc_id in args.ids:
        text, report = recover_document(args.dataset_dir / "hindi" / "original" / f"{doc_id}.pdf", args.dpi)
        (args.out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        reports[doc_id] = report
        print(json.dumps({"doc_id": doc_id, **{k: v for k, v in report.items() if k != "repair_examples"}}), flush=True)
    (args.out_dir / "recovery_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
