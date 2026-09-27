"""Tests for `scripts.package_submission` using a small fixture project tree (no real repo files)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.package_submission import _is_excluded, build_submission


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"

    # --- included ---
    _write(root / "src" / "adalat_mt" / "__init__.py")
    _write(root / "src" / "adalat_mt" / "foo.py")
    _write(root / "src" / "adalat_mt" / "__pycache__" / "foo.cpython-312.pyc")  # excluded (nested __pycache__)
    _write(root / "tests" / "test_foo.py")
    _write(root / "configs" / "data.yaml")
    _write(root / "scripts" / "package_submission.py")
    _write(root / "results" / "metrics.json")
    _write(root / "data" / "processed" / "train.jsonl")
    _write(root / "data" / "recovered" / "hindi" / "6.txt")
    _write(root / "data" / "audit" / "alignment_sample.jsonl")
    _write(root / "assets" / "fonts" / "font.ttf")
    _write(root / "REPORT.md")
    _write(root / "README.md")
    _write(root / "Makefile")
    _write(root / "pyproject.toml")
    _write(root / "requirements.txt")
    _write(root / "requirements-gpu.txt")
    _write(root / "requirements-comet.txt")
    _write(root / "DECISIONS.md")
    _write(root / "EXPERIMENTS.md")

    # --- excluded (must never appear in the zip) ---
    _write(root / ".env")
    _write(root / ".env.local")
    _write(root / ".harness" / "state.json")
    _write(root / ".claude" / "settings.json")
    _write(root / "harness" / "bin" / "doctor")
    _write(root / "dataset" / "english" / "clean" / "1.txt")
    _write(root / "artifacts" / "tokenizers" / "hi_spm.model")
    _write(root / "data" / "interim" / "alignments.jsonl")
    _write(root / ".cache" / "labse" / "model.pkl")
    _write(root / "outputs" / "lora_it2_1b" / "best" / "adapter_model.bin")  # not a deliverable path at all
    _write(root / "src" / "adalat_mt" / "weights.safetensors")
    _write(root / "src" / "adalat_mt" / "checkpoint.pt")

    return root


def test_build_submission_includes_deliverables(fixture_root: Path, tmp_path: Path):
    out_path = tmp_path / "dist" / "submission.zip"
    n_files, total_size = build_submission(fixture_root, out_path)

    assert out_path.exists()
    assert n_files > 0
    assert total_size > 0

    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())

    for expected in [
        "src/adalat_mt/__init__.py",
        "src/adalat_mt/foo.py",
        "tests/test_foo.py",
        "configs/data.yaml",
        "scripts/package_submission.py",
        "results/metrics.json",
        "data/processed/train.jsonl",
        "data/recovered/hindi/6.txt",
        "data/audit/alignment_sample.jsonl",
        "assets/fonts/font.ttf",
        "REPORT.md",
        "README.md",
        "Makefile",
        "pyproject.toml",
        "requirements.txt",
        "requirements-gpu.txt",
        "requirements-comet.txt",
        "DECISIONS.md",
        "EXPERIMENTS.md",
    ]:
        assert expected in names, f"missing deliverable: {expected}"


def test_build_submission_excludes_forbidden_patterns(fixture_root: Path, tmp_path: Path):
    out_path = tmp_path / "dist" / "submission.zip"
    build_submission(fixture_root, out_path)

    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())

    for forbidden in [
        "src/adalat_mt/__pycache__/foo.cpython-312.pyc",
        ".env",
        ".env.local",
        ".harness/state.json",
        ".claude/settings.json",
        "harness/bin/doctor",
        "dataset/english/clean/1.txt",
        "artifacts/tokenizers/hi_spm.model",
        "data/interim/alignments.jsonl",
        ".cache/labse/model.pkl",
        "outputs/lora_it2_1b/best/adapter_model.bin",
        "src/adalat_mt/weights.safetensors",
        "src/adalat_mt/checkpoint.pt",
    ]:
        assert forbidden not in names, f"forbidden path leaked into package: {forbidden}"

    for name in names:
        assert not _is_excluded(Path(name)), f"excluded pattern present in package: {name}"


def test_is_excluded_rules():
    assert _is_excluded(Path("a/__pycache__/b.py"))
    assert _is_excluded(Path("foo.pyc"))
    assert _is_excluded(Path(".env"))
    assert _is_excluded(Path(".env.example"))  # spec excludes ".env*" unconditionally
    assert _is_excluded(Path("harness/bin/doctor"))
    assert _is_excluded(Path(".harness/x"))
    assert _is_excluded(Path(".claude/x"))
    assert _is_excluded(Path("dataset/x"))
    assert _is_excluded(Path("artifacts/x"))
    assert _is_excluded(Path("model.safetensors"))
    assert _is_excluded(Path("model.bin"))
    assert _is_excluded(Path("checkpoint.pt"))
    assert _is_excluded(Path(".cache/x"))
    assert _is_excluded(Path("data/interim/x.jsonl"))
    assert not _is_excluded(Path("data/processed/train.jsonl"))
    assert not _is_excluded(Path("src/adalat_mt/foo.py"))
