"""Batched en->hi translation with IndicTrans2 / NLLB.

CLI: ``python -m adalat_mt.inference.translate --system it2-1b --split test``.
Writes ``<out-dir>/<system>.<split>.hi``, ``<out-dir>/<split>.src.en``, ``<out-dir>/<split>.ref.hi`` and
``<out-dir>/<system>.<split>.meta.json``. See ``configs/systems.yaml`` for the system registry.
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import transformers
import yaml
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ProgressCB = Callable[[int, int], None]


@dataclass
class TranslationResult:
    """Batched translation output.

    ``tokens_in`` counts non-pad input ids across the whole call, including any language tags and the
    trailing EOS added by the tokenizer (i.e. everything the encoder actually consumes). ``tokens_out``
    counts generated ids across the whole call, excluding pad/BOS/EOS ids and any forced target-language
    token (e.g. NLLB's ``forced_bos_token_id``).
    """

    hyps: list[str]
    tokens_in: int
    tokens_out: int
    seconds: float


class Translator(ABC):
    """Common interface for batched en->hi translation systems."""

    @abstractmethod
    def translate(
        self, sentences: list[str], progress_cb: ProgressCB | None = None
    ) -> TranslationResult:
        """Translate `sentences` and return hypotheses in the same order, plus token/timing counters."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def pick_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if available else ``"cpu"``; otherwise return `device` unchanged."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def pick_dtype(device: str | torch.device) -> torch.dtype:
    """fp16 on CUDA, fp32 otherwise (a Colab T4 has no bf16 support)."""
    device_type = device.type if isinstance(device, torch.device) else str(device)
    return torch.float16 if device_type == "cuda" else torch.float32


def sort_by_length_indices(sentences: list[str]) -> list[int]:
    """Return indices that sort `sentences` by ascending whitespace-token length (stable)."""
    return sorted(range(len(sentences)), key=lambda i: len(sentences[i].split()))


def restore_order(items: list, order: list[int]) -> list:
    """Inverse-permute `items` (produced following `order`) back to the original order.

    `order[k]` is the original index of the item that ended up at position `k` in `items`.
    """
    if len(items) != len(order):
        raise ValueError(f"length mismatch: {len(items)} items vs {len(order)} order indices")
    restored: list = [None] * len(items)
    for pos, orig_idx in enumerate(order):
        restored[orig_idx] = items[pos]
    return restored


def count_input_tokens(input_ids: torch.Tensor, pad_token_id: int | None) -> int:
    """Count non-pad tokens in a batch of input ids (encoder side, incl. tags/EOS)."""
    if pad_token_id is None:
        return int(input_ids.numel())
    return int((input_ids != pad_token_id).sum().item())


def count_output_tokens(
    output_ids: torch.Tensor, pad_token_id: int | None, special_ids: set[int]
) -> int:
    """Count generated tokens excluding pad and any id in `special_ids` (BOS/EOS/forced language token)."""
    mask = torch.ones_like(output_ids, dtype=torch.bool)
    if pad_token_id is not None:
        mask &= output_ids != pad_token_id
    for sid in special_ids:
        if sid is None:
            continue
        mask &= output_ids != sid
    return int(mask.sum().item())


def apply_lora(base_model, adapter_path: str):
    """Load a LoRA adapter onto `base_model` and merge the weights back into the base model (peft)."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    return peft_model.merge_and_unload()


def system_output_name(system: str, limit: int | None) -> str:
    """Base name for prediction files; appends ``.limitN`` for smoke-test (`--limit`) runs."""
    return f"{system}.limit{limit}" if limit is not None else system


def load_split(data_dir: Path, split: str) -> list[dict]:
    """Read ``<data_dir>/<split>.jsonl`` rows (Phase 1 output): one JSON dict per line."""
    path = Path(data_dir) / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------------------
# IndicTrans2
# --------------------------------------------------------------------------------------


def translate_indictrans2(
    model,
    tokenizer,
    sentences: list[str],
    *,
    device: torch.device,
    batch_size: int,
    num_beams: int,
    max_length: int,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "hin_Deva",
    progress_cb: ProgressCB | None = None,
) -> TranslationResult:
    """Translate `sentences` en->hi with an already-loaded IndicTrans2 model/tokenizer.

    Runs ``IndicProcessor(inference=True)`` preprocess -> length-sorted batched ``generate`` -> decode ->
    postprocess, pre/post-processing each batch in the same order so the toolkit's placeholder queue (which
    masks dates/URLs as ``<ID1>`` etc.) stays correct. Uses ``torch.inference_mode()`` throughout, and
    ``torch.autocast(fp16)`` on CUDA when the model's parameters are still fp32 (mixed precision on the fly).
    `model` may be a `peft.PeftModel` and may still be in train mode — the caller is responsible for
    `model.eval()`.
    """
    processor = IndicProcessor(inference=True)
    order = sort_by_length_indices(sentences)
    sorted_sents = [sentences[i] for i in order]
    n = len(sorted_sents)

    pad_id = tokenizer.pad_token_id
    special_ids = {pad_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
    special_ids.discard(None)

    model_dtype = next(model.parameters()).dtype
    use_autocast = device.type == "cuda" and model_dtype == torch.float32

    hyps_sorted: list[str] = []
    tokens_in = 0
    tokens_out = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = sorted_sents[start : start + batch_size]
            batch_proc = processor.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
            enc = tokenizer(
                batch_proc, padding="longest", truncation=True, max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            tokens_in += count_input_tokens(enc["input_ids"], pad_id)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(**enc, num_beams=num_beams, max_length=max_length, use_cache=True)
            else:
                out = model.generate(**enc, num_beams=num_beams, max_length=max_length, use_cache=True)
            tokens_out += count_output_tokens(out, pad_id, special_ids)
            dec = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            hyps_batch = processor.postprocess_batch(dec, lang=tgt_lang)
            hyps_sorted.extend(hyps_batch)
            if progress_cb is not None:
                progress_cb(min(start + batch_size, n), n)
    seconds = time.perf_counter() - t0
    hyps = restore_order(hyps_sorted, order)
    hyps = [h.replace("\n", " ") for h in hyps]
    return TranslationResult(hyps=hyps, tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds)


class IndicTrans2Translator(Translator):
    """IndicTrans2 (en-indic) translator, optionally with a merged LoRA adapter."""

    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        num_beams: int = 5,
        max_length: int = 256,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
    ) -> None:
        self.device = torch.device(pick_device(device))
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype
        )
        if adapter:
            model = apply_lora(model, adapter)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def translate(
        self, sentences: list[str], progress_cb: ProgressCB | None = None
    ) -> TranslationResult:
        return translate_indictrans2(
            self.model,
            self.tokenizer,
            sentences,
            device=self.device,
            batch_size=self.batch_size,
            num_beams=self.num_beams,
            max_length=self.max_length,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            progress_cb=progress_cb,
        )


# --------------------------------------------------------------------------------------
# NLLB
# --------------------------------------------------------------------------------------


class NLLBTranslator(Translator):
    """NLLB-200 translator (zero-shot baseline)."""

    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        num_beams: int = 5,
        max_length: int = 256,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
    ) -> None:
        self.device = torch.device(pick_device(device))
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=self.dtype)
        if adapter:
            model = apply_lora(model, adapter)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

    def translate(
        self, sentences: list[str], progress_cb: ProgressCB | None = None
    ) -> TranslationResult:
        order = sort_by_length_indices(sentences)
        sorted_sents = [sentences[i] for i in order]
        n = len(sorted_sents)

        pad_id = self.tokenizer.pad_token_id
        special_ids = {pad_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.forced_bos_token_id}
        special_ids.discard(None)

        model_dtype = next(self.model.parameters()).dtype
        use_autocast = self.device.type == "cuda" and model_dtype == torch.float32

        hyps_sorted: list[str] = []
        tokens_in = 0
        tokens_out = 0
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                batch = sorted_sents[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch, padding="longest", truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                tokens_in += count_input_tokens(enc["input_ids"], pad_id)
                gen_kwargs = dict(
                    num_beams=self.num_beams,
                    max_length=self.max_length,
                    use_cache=True,
                    forced_bos_token_id=self.forced_bos_token_id,
                )
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model.generate(**enc, **gen_kwargs)
                else:
                    out = self.model.generate(**enc, **gen_kwargs)
                tokens_out += count_output_tokens(out, pad_id, special_ids)
                dec = self.tokenizer.batch_decode(
                    out, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                hyps_sorted.extend(dec)
                if progress_cb is not None:
                    progress_cb(min(start + self.batch_size, n), n)
        seconds = time.perf_counter() - t0
        hyps = restore_order(hyps_sorted, order)
        hyps = [h.replace("\n", " ") for h in hyps]
        return TranslationResult(hyps=hyps, tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds)


def build_translator(system_cfg: dict, gen_cfg: dict, device: str) -> Translator:
    """Instantiate the `Translator` described by a `configs/systems.yaml` entry."""
    kind = system_cfg["kind"]
    common = dict(
        model_name=system_cfg["model"],
        adapter=system_cfg.get("adapter"),
        device=device,
        batch_size=gen_cfg.get("batch_size", 16),
        num_beams=gen_cfg.get("num_beams", 5),
        max_length=gen_cfg.get("max_length", 256),
    )
    if kind == "indictrans2":
        return IndicTrans2Translator(**common)
    if kind == "nllb":
        return NLLBTranslator(**common)
    raise ValueError(f"unknown system kind: {kind!r}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: translate one split with one system and write predictions + metadata."""
    parser = argparse.ArgumentParser(description="Translate a data split with one MT system.")
    parser.add_argument("--system", required=True, help="key into configs/systems.yaml")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="use only the first N rows (smoke test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default="configs/systems.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results/predictions")
    parser.add_argument("--adapter", default=None, help="override the adapter path from configs/systems.yaml")
    parser.add_argument(
        "--num-beams", type=int, default=None, help="override configs/systems.yaml generation.num_beams "
        "(e.g. 1 for greedy decoding in smoke tests)"
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen_cfg = dict(cfg["generation"])
    if args.num_beams is not None:
        gen_cfg["num_beams"] = args.num_beams
    system_cfg = dict(cfg["systems"][args.system])
    if args.adapter:
        system_cfg["adapter"] = args.adapter
    label = system_cfg.get("label", args.system)
    model_name = system_cfg["model"]

    rows = load_split(Path(args.data_dir), args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    en = [r["en"] for r in rows]
    hi = [r["hi"] for r in rows]
    n = len(en)

    device = pick_device(args.device)
    translator = build_translator(system_cfg, gen_cfg, device=device)

    def progress(done: int, total: int) -> None:
        payload = json.dumps({"system": args.system, "done": done, "n": total})
        print(f"[metric] {payload}")

    result = translator.translate(en, progress_cb=progress)

    name = system_output_name(args.system, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_lines(out_dir / f"{name}.{args.split}.hi", result.hyps)
    _write_lines(out_dir / f"{args.split}.src.en", en)
    _write_lines(out_dir / f"{args.split}.ref.hi", hi)

    dtype_str = str(translator.dtype).replace("torch.", "")
    meta = {
        "system": name,
        "label": label,
        "model": model_name,
        "adapter": system_cfg.get("adapter"),
        "split": args.split,
        "n": n,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "seconds": result.seconds,
        "sents_per_sec": (n / result.seconds) if result.seconds > 0 else None,
        "tokens_out_per_sec": (result.tokens_out / result.seconds) if result.seconds > 0 else None,
        "device": device,
        "dtype": dtype_str,
        "num_beams": gen_cfg.get("num_beams", 5),
        "batch_size": gen_cfg.get("batch_size", 16),
        "max_length": gen_cfg.get("max_length", 256),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (out_dir / f"{name}.{args.split}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
