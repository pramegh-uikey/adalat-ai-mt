# Decisions (ADR-lite): context → options → decision → why → consequences

## D1 — Main MT model
- Context: free Colab T4 (16 GB, fp16 only, ≤3 GPU-h), ~1.5–3k aligned legal pairs, need strong zero-shot EN→HI.
- Options: IndicTrans2-en-indic-1B, IndicTrans2-en-indic-dist-200M, NLLB-200-distilled-600M, mBART-50, a 1–2B instruct LLM.
- Decision: **IndicTrans2-en-indic-1B + LoRA** is the main system; its zero-shot output is the "before" baseline.
- Why: IndicTrans2 is one of the strongest open EN→Indic models at this size according to its paper (Gala et al., TMLR 2023), MIT licence, Indic-specific
  SentencePiece vocab (lower Hindi fertility than LLM tokenizers). At 1.1B params in fp16 plus LoRA it fits on a T4 comfortably.
  The 1B model is gated on the Hugging Face Hub; I accepted its licence there.
- Consequences: needs `trust_remote_code` and IndicTransToolkit pre/post-processing; transformers version must be pinned to one
  the remote code supports.

## D2 — Second baselines
- Decision: zero-shot **IndicTrans2-dist-200M** (same family, a size ablation that runs on CPU) and **NLLB-200-distilled-600M**
  (a second family with a different tokenizer, CC-BY-NC, fine for research).
- Why: gives a tokenizer-vs-quality comparison across families and a cheap CPU fallback system.

## D3 — Tokenizers to compare (all on the same train-split text)
- tiktoken `cl100k_base`, `o200k_base` (GPT-4 / GPT-4o API cost story)
- Llama-3.2 (via the ungated `unsloth/Llama-3.2-1B` mirror, which has identical tokenizer files), Gemma-3 (`unsloth/gemma-3-1b-it`)
- Qwen2.5 (the LLM used for the P2 vocab-extension experiment)
- Sarvam-1 (Indic-optimised LLM tokenizer), MuRIL (Indic BERT WordPiece)
- IndicTrans2 SPM (the MT model's tokenizer), NLLB-200 SPM, mBART-50 SPM
- SUTRA: `TWO/sutra-mlt256-v2` exposes no tokenizer files on the Hub, so it is excluded (checked 2026-09-27).

## D4 — GPU budget (target ≤ 3 GPU-h, one or two sessions)
- Session A (~1.5 h): zero-shot inference IT2-1B/NLLB on dev+test → LoRA training IT2-1B (dev chrF early stopping) →
  adapted test inference → COMET for all systems → `gpu down`.
- Optional session B (≤1 h): P2 LLM vocab-extension quality run, only if P0/P1 are done.
- The VM is released between sessions and while doing CPU work.

## D5 — Alignment method
- Context: paragraph numbering is not parallel; English has missing italic spans; some Hindi paragraphs contain broken
  legacy-font Devanagari; both sides are hard-wrapped.
- Decision: unwrap into segments (a line break becomes a boundary only at list/para markers or after terminal punctuation +
  blank line), rule-based sentence split (EN abbreviation list; HI on `।?!` and sentence-final `.` guarded by abbreviations),
  then **document-level monotonic DP over LaBSE embeddings** (vecalign-style) with moves 1-1, 1-2, 2-1, 1-3, 3-1, 1-0, 0-1,
  a small bonus when both blocks start with the same para number (soft anchor), and a post-filter on similarity and length ratio.
- Why: bertalign/vecalign as packages bring heavy deps (faiss, googletrans); a ~150-line DP is testable and adequate at 30 docs.

## D6 — Split
- By document, 24/3/3, seed 13, ids recorded in `configs/split.yaml`. Dev drives early stopping; test is touched only for
  final reporting.

## D7 — LoRA configuration and training loop
- Context: 1,149 training pairs, IT2-1B (18+18 layers, d=1024), T4 fp16, dev set of 3 docs.
- Decision: LoRA r=16, α=32, dropout 0.1 on q/k/v/out_proj + fc1/fc2 in encoder and decoder (~1.6% trainable), lr 3e-4,
  effective batch 16, label smoothing 0.1, ≤6 epochs, early stopping on **dev chrF++** (beam 4, full IndicProcessor
  pipeline) with patience 2. fp32 master weights + fp16 autocast. A small custom loop instead of Seq2SeqTrainer.
- Why: FFN targets matter for domain/terminology shift (not only attention); r=16 is the middle of the 8–32 range and the
  dataset is too small to tune rank properly. The custom loop keeps IndicTransToolkit's placeholder state correct in dev
  generation, and resume-from-checkpoint is explicit (Colab VM loss).
- Consequences: one config and one seed. The rank is not tuned on test; any rank ablation would use dev only.
- Amendment (2026-09-27): micro-batch 8×2 ran out of CUDA memory on the T4 (1B model + fp32 logits over the 122k Hindi
  vocabulary for 256-token targets), so it is now 4×4 (effective batch unchanged at 16), with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The same recipe is applied to IT2-dist-200M
  (`configs/lora_it2_200m.yaml`) as a size ablation, added after its zero-shot test score turned out on par with the 1B.

## D8 — Devanagari figures
- matplotlib has no complex-script shaping, so Devanagari conjuncts render broken. Plots with Latin labels use matplotlib;
  the Hindi tokenization illustration uses Pillow + libraqm (system package `libraqm0`) with Noto Sans Devanagari (OFL,
  vendored in `assets/fonts/`).

## D9 — Recover the 5 Hindi documents with no Devanagari (6, 14, 22, 25, 26)
- Context: their provided Hindi `clean` files are entirely `?`. The PDF text layer (PyMuPDF) exists but is legacy-font
  garbled (`भार` for `भारत`, `सिसविवल` for `सिविल`); its digits, however, are exact ASCII. I chose to try recovering these documents instead of excluding them.
- Options: exclude (19/3/3 on 25 docs); use the garbled text layer; OCR.
- Decision: **Tesseract 5 `hin` OCR at 300 dpi for words + number repair from the PDF text layer** (difflib alignment of
  the two number sequences on a key that ignores Tesseract's systematic 1-drop / 1→4 confusion), footer disclaimers
  stripped (`src/adalat_mt/data/pdf_recover.py`). Recovered docs go to **train only**; dev/test keep references from the
  provided clean text. Split becomes 24/3/3 (the original 25-doc random split + the 5 recovered docs in train).
- Evidence: +226 train pairs (1,205 → 1,431 total); median LaBSE score 0.856 vs 0.875 for clean-text docs; a
  supplementary read-through of 10 recovered pairs: 9 correct, 1 partial (`results/data/alignment_report.md`).
  75% of recovered HI numbers (≥3 chars) occur in the EN doc vs 91% for clean docs; the gap is mostly header case numbers
  and citations that the EN clean text omits, with a few residual OCR number errors (e.g. `5.0.7`).
- Consequences: system deps `tesseract-ocr tesseract-ocr-hin` are needed only to regenerate `data/recovered/`, which is
  committed. Residual OCR noise sits only in training data.

## D10 — COMET on CPU in an isolated venv
- Context: `unbabel-comet` pins numpy<2 and protobuf<5; installing it on the Colab image broke `transformers`, so the
  GPU job skipped COMET.
- Decision: run `Unbabel/wmt22-comet-da` locally on CPU in `.venv-comet` (`make setup-comet comet`), instead of
  spending more GPU quota on dependency surgery.
- Consequences: COMET scores for all 5 systems (≈ 40 min on CPU including the download); the main environment is untouched.
