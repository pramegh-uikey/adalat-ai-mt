"""LoRA fine-tuning of IndicTrans2 on the aligned legal train split.

A deliberately small, explicit PyTorch loop (instead of ``Seq2SeqTrainer``) because dev evaluation must run the full
IndicTransToolkit pre/post-processing around ``generate`` — the toolkit keeps per-sentence placeholder state that does
not fit the Trainer's ``predict_with_generate`` path.

* fp16 autocast with fp32 master weights and a GradScaler (Colab T4: no bf16).
* Model selection uses **dev** chrF++ only; the test split is never read here.
* Checkpoints (adapter + optimizer/scheduler/scaler state) are written every epoch to ``<out>/last`` so a lost Colab VM
  can resume with ``--resume``; the best adapter by dev chrF++ is kept in ``<out>/best``.
* Progress is printed as ``[metric] {json}`` lines, easy to grep from job logs.

Usage::

    python -m adalat_mt.training.train_lora --config configs/lora_it2_1b.yaml [--resume] [--limit-train N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from adalat_mt.evaluation.metrics import corpus_metrics
from adalat_mt.inference.translate import load_split, translate_indictrans2


def log_metric(**fields: Any) -> None:
    """Print one machine-readable progress line."""
    print("[metric] " + json.dumps(fields, ensure_ascii=False), flush=True)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Batch:
    """A padded training batch on the target device."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def encode_pairs(tokenizer: Any, processor: Any, en: list[str], hi: list[str], cfg: dict) -> list[dict[str, list[int]]]:
    """Pre-process and tokenize parallel sentences the way IndicTrans2 was trained.

    Sources get language tags + English normalisation; targets get Indic normalisation (``is_target=True``). The
    tokenizer appends EOS to both; labels are the target ids (the model shifts them right internally, starting from
    ``decoder_start_token_id``).
    """
    src = processor.preprocess_batch(en, src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"])
    tgt = processor.preprocess_batch(hi, src_lang=cfg["tgt_lang"], is_target=True)
    enc = tokenizer(src, truncation=True, max_length=cfg["max_length"])
    lab = tokenizer(text_target=tgt, truncation=True, max_length=cfg["max_length"])
    return [{"input_ids": i, "labels": l} for i, l in zip(enc["input_ids"], lab["input_ids"])]


def collate(examples: list[dict[str, list[int]]], pad_id: int, device: torch.device) -> Batch:
    """Right-pad a list of encoded examples; padded label positions become -100."""
    src_len = max(len(e["input_ids"]) for e in examples)
    tgt_len = max(len(e["labels"]) for e in examples)
    ids = torch.full((len(examples), src_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), src_len), dtype=torch.long)
    labels = torch.full((len(examples), tgt_len), -100, dtype=torch.long)
    for row, e in enumerate(examples):
        ids[row, : len(e["input_ids"])] = torch.tensor(e["input_ids"])
        mask[row, : len(e["input_ids"])] = 1
        labels[row, : len(e["labels"])] = torch.tensor(e["labels"])
    return Batch(ids.to(device), mask.to(device), labels.to(device))


def batches(examples: list[dict], batch_size: int, rng: random.Random | None) -> list[list[dict]]:
    """Split examples into batches; shuffled when ``rng`` is given, length-sorted (for fast eval) otherwise."""
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    else:
        order.sort(key=lambda i: len(examples[i]["input_ids"]))
    return [[examples[i] for i in order[k : k + batch_size]] for k in range(0, len(order), batch_size)]


def seq2seq_loss(model: Any, batch: Batch, label_smoothing: float) -> torch.Tensor:
    """Token-level (label-smoothed) cross-entropy, computed in fp32."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
    logits = out.logits.float()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), batch.labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing
    )


@torch.inference_mode()
def dev_loss(model: Any, examples: list[dict], pad_id: int, device: torch.device, cfg: dict) -> float:
    """Mean teacher-forced dev loss (no label smoothing)."""
    model.eval()
    total, n = 0.0, 0
    for chunk in batches(examples, cfg["eval"]["batch_size"], rng=None):
        batch = collate(chunk, pad_id, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["train"]["fp16"] and device.type == "cuda"):
            loss = seq2seq_loss(model, batch, label_smoothing=0.0)
        total += loss.item() * len(chunk)
        n += len(chunk)
    return total / max(n, 1)


def dev_scores(model: Any, tokenizer: Any, en: list[str], hi: list[str], device: torch.device, cfg: dict) -> dict:
    """Translate the dev sources with beam search and score them with chrF++/BLEU."""
    model.eval()
    result = translate_indictrans2(
        model, tokenizer, en, device=device, batch_size=cfg["eval"]["batch_size"],
        num_beams=cfg["eval"]["num_beams"], max_length=cfg["max_length"],
        src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"],
    )
    m = corpus_metrics(result.hyps, hi)
    return {"dev_chrf": m["chrf"], "dev_bleu": m["bleu"]}


def build_model(cfg: dict, device: torch.device) -> tuple[Any, Any]:
    """Load the base model/tokenizer and wrap the model with a fresh LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"], trust_remote_code=True, torch_dtype=torch.float32)
    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none",
    )
    model = get_peft_model(model, peft_cfg).to(device)
    return model, tokenizer


def count_params(model: torch.nn.Module) -> dict[str, int]:
    """Trainable vs total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


def save_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any, scaler: Any, state: dict) -> None:
    """Save the adapter plus everything needed to resume training."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
         "state": state, "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}},
        path / "trainer_state.pt",
    )


def train(
    cfg: dict, resume: bool = False, limit_train: int | None = None, limit_dev: int | None = None
) -> dict:
    """Run LoRA training with per-epoch dev evaluation and early stopping; return the training summary.

    `limit_dev`, if given, restricts dev evaluation to the first N dev rows (smoke tests: makes
    the per-epoch `dev_loss`/`dev_scores` pass fast; model selection is never done on the test set
    regardless of this limit).
    """
    from IndicTransToolkit.processor import IndicProcessor
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights
    from transformers import get_linear_schedule_with_warmup

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["train"]

    train_rows = load_split(Path(cfg["data_dir"]), "train")
    dev_rows = load_split(Path(cfg["data_dir"]), "dev")
    if limit_train:
        train_rows = train_rows[:limit_train]
    if limit_dev:
        dev_rows = dev_rows[:limit_dev]

    model, tokenizer = build_model(cfg, device)
    processor = IndicProcessor(inference=False)
    train_ex = encode_pairs(tokenizer, processor, [r["en"] for r in train_rows], [r["hi"] for r in train_rows], cfg)
    dev_ex = encode_pairs(tokenizer, processor, [r["en"] for r in dev_rows], [r["hi"] for r in dev_rows], cfg)
    dev_en, dev_hi = [r["en"] for r in dev_rows], [r["hi"] for r in dev_rows]
    pad_id = tokenizer.pad_token_id
    params = count_params(model)
    log_metric(event="setup", n_train=len(train_ex), n_dev=len(dev_ex), device=device.type, **params)

    steps_per_epoch = math.ceil(len(train_ex) / tcfg["batch_size"] / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = tcfg["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    state: dict[str, Any] = {"epoch": 0, "step": 0, "best_chrf": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": []}
    last_dir, best_dir = out_dir / "last", out_dir / "best"
    if resume and (last_dir / "trainer_state.pt").exists():
        ckpt = torch.load(last_dir / "trainer_state.pt", weights_only=False)
        set_peft_model_state_dict(model, load_peft_weights(str(last_dir), device=device.type))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        log_metric(event="resumed", epoch=state["epoch"], best_chrf=state["best_chrf"])
    else:
        zero = {"epoch": 0, "dev_loss": round(dev_loss(model, dev_ex, pad_id, device, cfg), 4),
                **dev_scores(model, tokenizer, dev_en, dev_hi, device, cfg)}
        state["history"].append(zero)
        log_metric(**zero)

    rng = random.Random(cfg["seed"])
    for _ in range(state["epoch"]):  # keep the shuffling sequence identical after a resume
        batches(train_ex, tcfg["batch_size"], rng)
    t0 = time.time()
    for epoch in range(state["epoch"] + 1, tcfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        epoch_batches = batches(train_ex, tcfg["batch_size"], rng)
        for i, chunk in enumerate(epoch_batches):
            batch = collate(chunk, pad_id, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = seq2seq_loss(model, batch, tcfg["label_smoothing"])
            scaler.scale(loss / tcfg["grad_accum"]).backward()
            running += loss.item()
            seen += 1
            if (i + 1) % tcfg["grad_accum"] == 0 or i + 1 == len(epoch_batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state["step"] += 1
                if state["step"] % tcfg["log_every"] == 0:
                    log_metric(epoch=epoch, step=state["step"], loss=round(running / seen, 4),
                               lr=scheduler.get_last_lr()[0])
                    running, seen = 0.0, 0
        record = {"epoch": epoch, "step": state["step"], "dev_loss": round(dev_loss(model, dev_ex, pad_id, device, cfg), 4),
                  **dev_scores(model, tokenizer, dev_en, dev_hi, device, cfg), "elapsed_s": round(time.time() - t0, 1)}
        state["history"].append(record)
        log_metric(**record)
        state["epoch"] = epoch
        if record["dev_chrf"] > state["best_chrf"]:
            state.update(best_chrf=record["dev_chrf"], best_epoch=epoch, bad_epochs=0)
            model.save_pretrained(best_dir)
        else:
            state["bad_epochs"] += 1
        save_checkpoint(last_dir, model, optimizer, scheduler, scaler, state)
        if state["bad_epochs"] >= cfg["eval"]["patience"]:
            log_metric(event="early_stop", epoch=epoch, best_epoch=state["best_epoch"])
            break

    summary = {"config": cfg, **params, "best_epoch": state["best_epoch"], "best_dev_chrf": state["best_chrf"],
               "history": state["history"], "train_seconds": round(time.time() - t0, 1),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_metric(event="done", best_epoch=state["best_epoch"], best_dev_chrf=state["best_chrf"])
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/last")
    ap.add_argument("--limit-train", type=int, default=None, help="use only the first N train pairs (smoke tests)")
    ap.add_argument("--limit-dev", type=int, default=None, help="use only the first N dev pairs (smoke tests)")
    ap.add_argument("--out-dir", default=None, help="override out_dir from the config")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, resume=args.resume, limit_train=args.limit_train, limit_dev=args.limit_dev)


if __name__ == "__main__":
    main()
