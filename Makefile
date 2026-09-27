PY ?= .venv/bin/python
DEVICE ?= cuda
ADAPTER ?= outputs/lora_it2_1b/best
ADAPTER_200M ?= outputs/lora_it2_200m/best
COMET_PY ?= .venv-comet/bin/python

.PHONY: setup test recover-ocr data tokenizer-study vocab-extend tokenize baseline baseline-cpu \
	train-gpu setup-comet comet analysis qualitative eval report package smoke all

setup:
	uv pip install -p .venv -r requirements.txt
	uv pip install -p .venv -e .

test:
	$(PY) -m pytest -q -m "not slow"

# Needs system tesseract-ocr + tesseract-ocr-hin.
recover-ocr:
	$(PY) -m adalat_mt.data.pdf_recover --ids 6 14 22 25 26

data:
	$(PY) -m adalat_mt.data.build --config configs/data.yaml

tokenizer-study:
	$(PY) -m adalat_mt.tokenization.study --config configs/tokenizers.yaml

vocab-extend:
	$(PY) -m adalat_mt.tokenization.vocab_extend --config configs/vocab_extend.yaml

tokenize: tokenizer-study vocab-extend

# Zero-shot baselines on the test split. DEVICE ?= cuda (override e.g. `make baseline DEVICE=cpu`).
baseline:
	$(PY) -m adalat_mt.inference.translate --system it2-1b --split test --device $(DEVICE)
	$(PY) -m adalat_mt.inference.translate --system nllb-600m --split test --device $(DEVICE)
	$(PY) -m adalat_mt.inference.translate --system it2-200m --split test --device $(DEVICE)

baseline-cpu:
	$(PY) -m adalat_mt.inference.translate --system it2-200m --split test --device cpu \
		--out-dir results/predictions_cpu

# Needs a CUDA GPU (we used a Colab T4). ADAPTER / ADAPTER_200M default to the training outputs; point them at
# artifacts/lora_it2_1b/best and artifacts/lora_it2_200m/best to translate with the fetched adapters without retraining.
train-gpu:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) -m adalat_mt.training.train_lora --config configs/lora_it2_1b.yaml
	$(PY) -m adalat_mt.inference.translate --system it2-1b-lora --split test --device $(DEVICE) \
		--adapter $(ADAPTER)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) -m adalat_mt.training.train_lora --config configs/lora_it2_200m.yaml
	$(PY) -m adalat_mt.inference.translate --system it2-200m-lora --split test --device $(DEVICE) \
		--adapter $(ADAPTER_200M)

# COMET's pinned dependencies (numpy<2, protobuf<5) conflict with the main environment, so it gets its own venv.
setup-comet:
	uv venv -p 3.12 .venv-comet
	uv pip install -p .venv-comet --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cpu -r requirements-comet.txt
	uv pip install -p .venv-comet --no-deps -e .

# Unbabel/wmt22-comet-da; runs on CPU (≈ minutes for 5 × 120 segments) or GPU.
comet:
	$(COMET_PY) -m adalat_mt.evaluation.comet_score --systems it2-1b,it2-200m,nllb-600m,it2-1b-lora,it2-200m-lora \
		--split test

analysis:
	$(PY) -m adalat_mt.evaluation.analysis --eval-config configs/eval.yaml

qualitative:
	$(PY) -m adalat_mt.evaluation.qualitative render --curated results/qualitative/curated.yaml \
		--eval-config configs/eval.yaml

eval:
	$(PY) -m adalat_mt.evaluation.evaluate --config configs/eval.yaml
	$(MAKE) analysis
	$(MAKE) qualitative

report:
	$(PY) -m adalat_mt.reporting.build_report --report REPORT.md

package:
	$(PY) scripts/package_submission.py

smoke:
	bash scripts/smoke.sh

# CPU-only parts of the pipeline, end to end.
all: data tokenize eval report package
