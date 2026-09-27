"""CLI: build `dist/submission.zip` with only the assignment deliverables.

    python scripts/package_submission.py [--root .] [--out dist/submission.zip]

Included (relative to `--root`): `src/`, `tests/`, `configs/`, `scripts/`, `results/`,
`data/processed/`, `data/recovered/`, `data/audit/`, `assets/`, `REPORT.md`, `README.md`,
`Makefile`, `pyproject.toml`, `requirements.txt`, `requirements-gpu.txt`, `requirements-comet.txt`,
`DECISIONS.md`, `EXPERIMENTS.md`.

Excluded even if nested inside an included directory: `__pycache__/`, `*.pyc`, `.env*`,
`.harness/`, `.claude/`, `harness/`, `dataset/`, `artifacts/`, `*.safetensors`, `*.bin`, `*.pt`,
`.cache/`, `data/interim/`. After writing the zip, every entry is re-checked against the
exclusion rules and an `AssertionError` is raised if one slipped through.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

#: Paths (relative to the project root) that make up the submission, in the order they are
#: listed in the spec. Each is either a file or a directory (walked recursively).
DELIVERABLE_PATHS: tuple[str, ...] = (
    "src",
    "tests",
    "configs",
    "scripts",
    "results",
    "data/processed",
    "data/recovered",
    "data/audit",
    "assets",
    "REPORT.md",
    "README.md",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-gpu.txt",
    "requirements-comet.txt",
    "DECISIONS.md",
    "EXPERIMENTS.md",
)

#: Directory name components that exclude any file found under them.
_EXCLUDE_DIR_NAMES = frozenset({"__pycache__", ".harness", ".claude", "harness", "dataset", "artifacts", ".cache"})

#: File suffixes that are always excluded.
_EXCLUDE_SUFFIXES = frozenset({".pyc", ".safetensors", ".bin", ".pt"})


def _is_excluded(rel_path: Path) -> bool:
    """True if `rel_path` (relative to the project root) matches an exclusion rule."""
    parts = rel_path.parts
    if any(part in _EXCLUDE_DIR_NAMES or part.endswith(".egg-info") for part in parts):
        return True
    if any(part.startswith(".env") for part in parts):
        return True
    if rel_path.suffix in _EXCLUDE_SUFFIXES:
        return True
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "interim":
        return True
    return False


def iter_deliverable_files(root: Path) -> Iterator[Path]:
    """Yield every file under `root` that belongs in the submission, excluded ones dropped."""
    for entry in DELIVERABLE_PATHS:
        path = root / entry
        if path.is_file():
            if not _is_excluded(Path(entry)):
                yield path
        elif path.is_dir():
            for sub in sorted(path.rglob("*")):
                if not sub.is_file():
                    continue
                rel = sub.relative_to(root)
                if not _is_excluded(rel):
                    yield sub


def _verify_no_excluded(zip_path: Path) -> None:
    """Raise `AssertionError` if any entry written to `zip_path` matches an exclusion rule."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if _is_excluded(Path(name)):
                raise AssertionError(f"excluded pattern present in package: {name!r}")


def build_submission(root: Path, out_path: Path) -> tuple[int, int]:
    """Build `out_path` from the deliverables under `root`. Returns `(file_count, total_bytes)`."""
    root = Path(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(set(iter_deliverable_files(root)))
    total_size = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            rel = f.relative_to(root)
            zf.write(f, arcname=str(rel))
            total_size += f.stat().st_size

    _verify_no_excluded(out_path)
    return len(files), total_size


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Package the submission deliverables into a zip.")
    parser.add_argument("--root", default=".", type=Path, help="project root (default: cwd)")
    parser.add_argument("--out", default="dist/submission.zip", type=Path, help="output zip path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_files, total_size = build_submission(args.root, args.out)
    print(f"[package] wrote {args.out}: {n_files} files, {_human_size(total_size)}")


if __name__ == "__main__":
    main()
