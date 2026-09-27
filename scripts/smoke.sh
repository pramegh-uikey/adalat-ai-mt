#!/usr/bin/env bash
# CPU end-to-end smoke test on tiny data (< 10 minutes). Writes everything under .cache/smoke/ and
# touches nothing under results/ or data/ (the real configs/split.yaml is also left untouched).
#
#   make smoke
#   bash scripts/smoke.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-.venv/bin/python}"
SMOKE_ROOT=".cache/smoke"
DOC_IDS=(1 2 3) # assigned round-robin -> train=[1] dev=[2] test=[3] (see adalat_mt.data.build --doc-ids)

echo "[smoke] repo root: $ROOT_DIR"
echo "[smoke] resetting $SMOKE_ROOT"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

START=$(date +%s)

echo "[smoke] 1/5 data build (docs ${DOC_IDS[*]}, real LaBSE, cached after first run)"
"$PY" -m adalat_mt.data.build \
  --config configs/data.yaml \
  --doc-ids "${DOC_IDS[@]}" \
  --out-root "$SMOKE_ROOT"

echo "[smoke] 2/5 tokenizer study (cl100k_base, o200k_base only)"
"$PY" -m adalat_mt.tokenization.study \
  --config configs/tokenizers.yaml \
  --split-config "$SMOKE_ROOT/split.yaml" \
  --data-dir "$SMOKE_ROOT/data/processed" \
  --dataset-dir dataset \
  --out-dir "$SMOKE_ROOT" \
  --names cl100k_base o200k_base

echo "[smoke] 3/5 translate it2-200m (CPU, greedy, limit 4)"
"$PY" -m adalat_mt.inference.translate \
  --system it2-200m \
  --split test \
  --limit 4 \
  --device cpu \
  --num-beams 1 \
  --data-dir "$SMOKE_ROOT/data/processed" \
  --out-dir "$SMOKE_ROOT/predictions"

echo "[smoke] 4/5 train_lora (dist-200M, 1 epoch, tiny limits, CPU)"
cat > "$SMOKE_ROOT/lora_smoke.yaml" <<EOF
base_model: ai4bharat/indictrans2-en-indic-dist-200M
src_lang: eng_Latn
tgt_lang: hin_Deva
data_dir: $SMOKE_ROOT/data/processed
out_dir: $SMOKE_ROOT/outputs/lora_smoke
seed: 13
max_length: 256
lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, out_proj, fc1, fc2]
train:
  epochs: 1
  lr: 3.0e-4
  batch_size: 4
  grad_accum: 1
  warmup_ratio: 0.06
  weight_decay: 0.01
  label_smoothing: 0.1
  max_grad_norm: 1.0
  fp16: false
  log_every: 1
eval:
  num_beams: 1
  batch_size: 4
  patience: 2
EOF
"$PY" -m adalat_mt.training.train_lora \
  --config "$SMOKE_ROOT/lora_smoke.yaml" \
  --limit-train 16 \
  --limit-dev 4 \
  --epochs 1

echo "[smoke] 5/5 evaluate"
cat > "$SMOKE_ROOT/eval_smoke.yaml" <<EOF
split: test
pred_dir: $SMOKE_ROOT/predictions
comet_dir: $SMOKE_ROOT/comet
data_dir: $SMOKE_ROOT/data/processed
baseline: it2-200m.limit4
systems: [it2-200m.limit4]
comparisons: []
bootstrap: {n_samples: 50, seed: 13, metrics: [bleu, chrf]}
EOF
"$PY" -m adalat_mt.evaluation.evaluate \
  --config "$SMOKE_ROOT/eval_smoke.yaml" \
  --systems-config configs/systems.yaml \
  --metrics-out "$SMOKE_ROOT/metrics.json" \
  --per-segment-out "$SMOKE_ROOT/eval/per_segment.jsonl"

END=$(date +%s)
ELAPSED=$((END - START))
echo "[smoke] elapsed: ${ELAPSED}s"
echo "SMOKE OK"
