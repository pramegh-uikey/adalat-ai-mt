"""Where each document's raw text lives (provided clean text, or recovered Hindi text for the OCR'd documents)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

DATA_CONFIG = Path("configs/data.yaml")


def load_data_config(path: Path = DATA_CONFIG) -> dict[str, Any]:
    """Read the data-pipeline config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def source_path(doc_id: str, lang: Literal["en", "hi"], cfg: dict[str, Any], root: Path = Path(".")) -> Path:
    """Path of the raw text for ``doc_id`` in ``lang``, honouring ``recovered_hindi`` in the data config."""
    recovered = cfg.get("recovered_hindi", {})
    if lang == "hi" and str(doc_id) in {str(d) for d in recovered.get("ids", [])}:
        return root / recovered["dir"] / f"{doc_id}.txt"
    sub = "english" if lang == "en" else "hindi"
    return root / cfg["dataset_dir"] / sub / "clean" / f"{doc_id}.txt"
