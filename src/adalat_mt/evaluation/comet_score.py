"""GPU CLI: score system outputs with COMET (Colab T4 only; `comet` is a `requirements-gpu.txt` extra).

``comet`` is imported lazily inside `main` so this module can be imported (e.g. for tests) without the
package installed; no unit test in the default suite exercises COMET scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def score_system(
    system: str,
    split: str,
    pred_dir: Path,
    out_dir: Path,
    model,
    model_name: str,
    batch_size: int,
) -> dict:
    """Score one system's hypotheses with an already-loaded COMET `model` and write `<out_dir>/<system>.<split>.json`."""
    import torch

    hyps = _read_lines(pred_dir / f"{system}.{split}.hi")
    srcs = _read_lines(pred_dir / f"{split}.src.en")
    refs = _read_lines(pred_dir / f"{split}.ref.hi")
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]

    gpus = 1 if torch.cuda.is_available() else 0
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    result = {
        "system": system,
        "split": split,
        "model": model_name,
        "system_score": output.system_score,
        "segment_scores": list(output.scores),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{system}.{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: score one or more systems with COMET."""
    parser = argparse.ArgumentParser(description="Score MT system outputs with COMET (GPU).")
    parser.add_argument("--systems", required=True, help="comma-separated system names")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--out-dir", default="results/comet")
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    import comet  # lazy: GPU-only dependency (requirements-gpu.txt)

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    ckpt_path = comet.download_model(args.model)
    model = comet.load_from_checkpoint(ckpt_path)

    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        hyp_path = pred_dir / f"{system}.{args.split}.hi"
        if not hyp_path.exists():
            print(f"[comet] skipping {system!r}: no hyp file at {hyp_path}")
            continue
        score_system(system, args.split, pred_dir, out_dir, model, args.model, args.batch_size)


if __name__ == "__main__":
    main()
