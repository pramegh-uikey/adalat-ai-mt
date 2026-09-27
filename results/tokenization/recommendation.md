# Tokenizer recommendation (feeds Phase 3)

Source numbers: `results/tokenization/summary.json`, `cost_table.md`, `vocab_extension.json` (train split: 1,149 pairs,
24 judgments; test split for length limits).

## Findings

1. **Indic-aware vocabularies are at parity.** IndicTrans2's Hindi SPM (hi fertility 1.28, HI/EN token ratio 1.02),
   MuRIL (1.28 / 1.01) and IndicBERT-v2 (1.23 / 1.05) encode Hindi about as compactly as English. mBART-50 and NLLB-200
   (1.42 and 1.45) and Gemma-3 (1.46, ratio 1.13) are close behind.
2. **General-purpose LLM tokenizers carry a large Hindi premium.** Hindi costs 4.09× the English tokens under
   `cl100k_base` (GPT-4/3.5), 3.69× under Qwen2.5, 3.74× under Llama-2/TinyLlama and 2.20× under Llama-3.2. The newer
   `o200k_base` (GPT-4o) cuts this to 1.35×.
3. **The premium lands on the expensive side.** In EN→HI translation the Hindi side is the *output*, i.e. both the
   pricier tokens and the sequential decoding steps. An average train judgment needs 8,242 Hindi tokens under
   `cl100k_base` vs 2,731 under `o200k_base` and 2,080 under IndicTrans2, so about 3–4× more decoding steps (latency).
   At one illustrative price for every tokenizer (gpt-4o list price, an assumption), that is $0.0873 vs $0.0322 per
   judgment for `cl100k_base` vs `o200k_base`.
4. **Context and length limits.** 15.8% of test Hindi sentences exceed 256 tokens under `cl100k_base` (20.0% for
   TinyLlama) but 0% for every Indic-aware tokenizer. So IndicTrans2's 256-token window never truncates a test sentence.
5. **Vocabulary extension works where the base vocabulary is poor.** Merging Devanagari-only pieces from an 8k
   SentencePiece model trained on the *train* Hindi into the Llama-2 (TinyLlama) tokenizer adds 6,475 tokens and cuts
   Hindi test fertility from 5.45 to 1.40 (−74%), with English fertility unchanged (1.495). The gains flatten beyond 8k
   (16k: 1.35). New embedding rows are initialised as the mean of the old sub-token embeddings (input and untied
   `lm_head`) (`resize_and_init` in `src/adalat_mt/tokenization/vocab_extend.py`). The new rows still need training
   before they carry meaning; see REPORT.md for what was and was not run.

## Recommendation

- **MT system: IndicTrans2 (EN→Indic), used with its own tokenizer, unmodified.** It already has the lowest Hindi
  fertility of the MT candidates and near-parity length, so any vocabulary surgery would bring cost with no efficiency
  gain. Adaptation goes into LoRA weights, not the vocabulary (Phase 4).
- **Second family: NLLB-200-distilled-600M** (hi fertility 1.45). It lets us check whether tokenizer efficiency tracks
  translation quality across model families (Phase 5).
- **If an LLM must be used** (e.g. for summarisation or Q&A over Hindi judgments): prefer a tokenizer at or near parity
  (Gemma-3, `o200k_base`, Sarvam-1). For an older Llama-2-class model, extend the vocabulary with a domain SentencePiece
  model (Devanagari-only pieces, mean-initialised embeddings) and budget continued pre-training for the new rows.
