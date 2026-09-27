"""CLI: fill the generated table blocks of `REPORT.md` from `results/`.

``REPORT.md`` is hand-written prose with generated table blocks between markers::

    <!-- BEGIN:tokenizers -->
    ...generated markdown...
    <!-- END:tokenizers -->

``python -m adalat_mt.reporting.build_report [--report REPORT.md]`` replaces the content
between every ``BEGIN:<name>``/``END:<name>`` pair with ``TABLES[name]()``'s output, leaving
everything else untouched. Unknown names raise ``ValueError``; running twice on the same
REPORT.md is a no-op (idempotent) because each table is re-rendered from the same source files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

from adalat_mt.reporting.tables import TABLES

_BEGIN_RE = re.compile(r"<!--\s*BEGIN:([\w.-]+)\s*-->")


def render_markers(text: str, tables: dict[str, Callable[[], str]]) -> str:
    """Replace every `<!-- BEGIN:name --> ... <!-- END:name -->` block's body with `tables[name]()`.

    Raises `ValueError` for any marker name not in `tables` or used more than once. Content outside marker pairs is
    left byte-for-byte untouched.
    """
    names = _BEGIN_RE.findall(text)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate report table markers: {duplicates}")
    out = text
    for name in names:
        if name not in tables:
            raise ValueError(f"unknown report table marker: {name!r}")
        block_re = re.compile(
            r"(<!--\s*BEGIN:" + re.escape(name) + r"\s*-->)(.*?)(<!--\s*END:" + re.escape(name) + r"\s*-->)",
            re.DOTALL,
        )
        rendered = tables[name]()
        out = block_re.sub(lambda m, r=rendered: f"{m.group(1)}\n{r}\n{m.group(3)}", out, count=1)
    return out


def build_report(report_path: Path, tables: dict[str, Callable[[], str]] | None = None) -> str:
    """Render `report_path` in place (returns the new text)."""
    tables = TABLES if tables is None else tables
    text = report_path.read_text(encoding="utf-8")
    rendered = render_markers(text, tables)
    report_path.write_text(rendered, encoding="utf-8")
    return rendered


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fill generated table blocks in REPORT.md.")
    parser.add_argument("--report", default="REPORT.md", type=Path)
    args = parser.parse_args(argv)
    build_report(args.report)


if __name__ == "__main__":
    main()
