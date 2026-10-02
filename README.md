# adalat-mt: English → Hindi translation of Indian court judgments

Tokenizer-efficiency study + LoRA adaptation of IndicTrans2 on a 30-judgment parallel corpus, built for a free Colab T4.
**Read [`REPORT.md`](REPORT.md) for the write-up**; every number in it is generated from `results/`.


## Layout

```
src/adalat_mt/
  data/          text_clean · segment (unwrap + EN/HI sentence split) · align (LaBSE monotonic DP) · split ·
                 pdf_recover (OCR + text-layer number repair) · build (CLI) · report
  tokenization/  registry · metrics · study (CLI) · figures · vocab_extend (SPM merge + embedding resize) · embedding_probe
  inference/     translate (IndicTrans2 / NLLB, LoRA merge, token + latency accounting; CLI)
  training/      train_lora (LoRA loop, dev-chrF++ early stopping, resumable checkpoints; CLI)
  evaluation/    metrics (sacreBLEU) · significance (paired bootstrap) · comet_score · evaluate (CLI) ·
                 analysis (error analysis with counts) · qualitative (candidates + curated examples)
  reporting/     tables · build_report (fills the generated tables in REPORT.md)
configs/         data, split (doc ids), tokenizers, systems, eval, glossary, lora_it2_1b, lora_it2_200m, ...
data/            processed/{train,dev,test}.jsonl · recovered/hindi (OCR'd docs) · audit (alignment judgements)
results/         data/ · tokenization/ · figures/ · predictions/ · training/ · eval/ · qualitative/ · metrics.json
tests/           pytest suite (no downloads needed except tests marked `slow`)
scripts/         smoke.sh · package_submission.py
```

## Setup

```bash
uv venv -p 3.12 .venv && make setup          # or: pip install -r requirements.txt && pip install -e .
# optional system packages: libraqm0 (Devanagari shaping in figures), tesseract-ocr tesseract-ocr-hin (OCR recovery)
export HF_TOKEN=...                           # IndicTrans2-1B is gated: accept its licence on the Hub first
```

`transformers` is pinned to 4.47.1 because IndicTrans2's remote modelling code breaks on the newer cache API.

## Reproduce from scratch

```bash
make smoke        # < 2 min CPU end-to-end on 3 docs / a few sentences (writes only under .cache/smoke/)
make test         # pytest

make data         # clean + segment + LaBSE-align + split  -> data/processed, results/data/alignment_report.md
make tokenize     # 12-tokenizer study + vocabulary extension -> results/tokenization, results/figures
make baseline     # zero-shot IT2-1B, IT2-200M, NLLB-600M on test (GPU; DEVICE=cpu works, slowly)
make train-gpu    # LoRA IT2-1B and IT2-200M + adapted test inference (CUDA GPU, e.g. Colab T4; ~25 min)
make setup-comet comet   # COMET-22 in an isolated venv (CPU is fine)
make eval         # metrics.json (BLEU 13a, spBLEU, chrF++, COMET, bootstrap), error analysis, qualitative examples
make report       # regenerate every table in REPORT.md
make package      # dist/submission.zip (deliverables only)
```

`make recover-ocr` regenerates `data/recovered/hindi/` (the five Hindi documents whose provided text is all `?`); the
output is committed, so Tesseract is not needed for the normal path.

### Running the GPU steps on Colab

Open a Colab T4 runtime, clone/upload this folder, then:

```bash
pip install -r requirements-gpu.txt && pip install -e .
make baseline train-gpu
```

Copy `results/predictions/`, `results/training/` and `outputs/lora_*/best` back. (I drove Colab from the command
line; see `EXPERIMENTS.md` for the exact jobs, wall-clock times and the two failures fixed along the way.)

## Key results (test split, 120 segments; see REPORT.md §4 for CIs and p-values)

| system | chrF++ | spBLEU |
|---|---|---|
| IndicTrans2-1B zero-shot | 64.42 | 47.64 |
| IndicTrans2-1B + LoRA | 67.43 | 51.80 |
| IndicTrans2-200M + LoRA | 66.94 | 51.36 |
| NLLB-200-600M zero-shot | 55.39 | 35.25 |

(Copied from `results/metrics.json`; `make report` keeps REPORT.md in sync.)
