# LoRA Fine-Tuning Runbook

This is an experimental path for improving closed-book knowledge while keeping
the final inference architecture compliant: one open model, no RAG, no
embedding/reranker, no second LLM.

## Inputs

Config:

```text
configs/finetune_config.yaml
```

Data:

```text
data/finetune/qlora_2_5k_v1_train.jsonl
data/finetune/qlora_2_5k_v1_val.jsonl
data/finetune/qlora_2_5k_v1_test.jsonl
```

The default config trains **full bf16 LoRA** for A100/H100 quality runs.
On T4/smaller GPUs, set `model.load_in_4bit: true` to train 4-bit QLoRA.

## Colab Setup

Use an A100 runtime if available. T4/smaller runtimes should set
`load_in_4bit: true` in `configs/finetune_config.yaml`.

```bash
pip uninstall -y transformers tokenizers torchaudio torchvision torchtext torchao
pip install --no-cache-dir git+https://github.com/huggingface/transformers.git
pip install -U peft accelerate datasets safetensors sentencepiece protobuf PyYAML bitsandbytes
```

Restart the Colab runtime after installing Transformers from GitHub. This is
needed because `Qwen/Qwen3.5-4B` uses `model_type="qwen3_5"`, which older
Transformers builds do not recognize. The uninstall line also removes optional
audio/vision/quantization packages that can break imports when their CUDA
versions do not match Colab's current PyTorch install.

Do not install `vllm` before training unless you need inference in the same
runtime. It has stricter CUDA/Torch dependencies and can disturb an otherwise
working LoRA environment. Install it only after merge, or use a fresh runtime
for the smoke/full inference run.

If Colab or the environment sets fast transfer incorrectly:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## 1. Validate Data

```bash
python scripts/validate_finetune_data.py \
  --config configs/finetune_config.yaml \
  --stats-output data/finetune/qlora_2_5k_v1_validated_stats.json
```

Expected:

- no schema errors
- no duplicate IDs
- no duplicate exact questions across train/val/test
- all answers are valid labels

## 2. Train LoRA Adapter

```bash
python scripts/train_lora.py \
  --config configs/finetune_config.yaml
```

Default output:

```text
outputs/finetune/qwen35_4b_lora_2_5k_v1/
```

This directory contains adapter weights, not a merged full model.

The script prints the selected precision at startup. Expected examples:

- A100/H100 with default config: `bf16 (full LoRA)`
- T4/smaller with `load_in_4bit: true`: `fp16 (4-bit QLoRA)`

## 3. Preferred Inference: Base Model + Live LoRA Adapter

Use `v03_delta` to avoid merged-model metadata issues. This loads the clean
base model and applies the adapter directly in vLLM.

Smoke test:

```bash
python src/v03_delta.py \
  --input data/public-test_1780368312.json \
  --limit 5 \
  --safe-mode \
  --model-id Qwen/Qwen3.5-4B \
  --lora-adapter outputs/finetune/qwen35_4b_lora_2_5k_v1 \
  --output data/submissions/submission_v03_delta_smoke.csv \
  --trace-output data/traces/trace_v03_delta_smoke.jsonl
```

Full public run:

```bash
python src/v03_delta.py \
  --input data/public-test_1780368312.json \
  --safe-mode \
  --model-id Qwen/Qwen3.5-4B \
  --lora-adapter outputs/finetune/qwen35_4b_lora_2_5k_v1 \
  --output data/submissions/submission_v03_delta.csv \
  --trace-output data/traces/trace_v03_delta.jsonl
```

## 4. Optional Fallback: Merge Adapter

```bash
python scripts/merge_lora.py \
  --config configs/finetune_config.yaml
```

Default merged model output:

```text
outputs/finetune/qwen35_4b_lora_2_5k_v1_merged/
```

Use this merged directory for inference comparison.

If vLLM complains about vision/processor metadata after merge, sanitize the
merged directory:

```bash
python scripts/sanitize_merged_model.py \
  outputs/finetune/qwen35_4b_lora_2_5k_v1_merged \
  --tokenizer-source outputs/finetune/qwen35_4b_lora_2_5k_v1
```

If the merged checkpoint contains `model.language_model.*` keys instead of
`model.*`, convert it once:

```bash
python scripts/convert_qwen35_merged_to_causal_lm.py \
  --input-dir outputs/finetune/qwen35_4b_lora_2_5k_v1_merged \
  --output-dir outputs/finetune/qwen35_4b_lora_2_5k_v1_merged_text
```

## 5. Smoke Test Merged Inference

Run a tiny sample first:

```bash
pip install "vllm>=0.17.0"
```

```bash
python src/v02_gamma.py \
  --input data/public-test_1780368312.json \
  --limit 5 \
  --safe-mode \
  --model-id outputs/finetune/qwen35_4b_lora_2_5k_v1_merged \
  --output data/submissions/submission_lora_smoke.csv \
  --trace-output data/traces/trace_lora_smoke.jsonl
```

Then run the full public set:

```bash
python src/v02_gamma.py \
  --input data/public-test_1780368312.json \
  --safe-mode \
  --model-id outputs/finetune/qwen35_4b_lora_2_5k_v1_merged \
  --output data/submissions/submission_v03_lora_2_5k_v1.csv \
  --trace-output data/traces/trace_v03_lora_2_5k_v1.jsonl
```

## 5. Evaluate

Open:

```text
notebooks/evaluation.ipynb
```

Compare against the current best baseline. Keep the LoRA model only if:

- overall score improves
- `knowledge` improves
- `reading` does not regress materially
- `stem` does not regress materially
- `safety` remains correct
- regression count is acceptable

## Notes

- Do not train on public-test answers directly.
- Do not commit `outputs/`; they contain model weights.
- If the model overfits, try lower learning rate (`5e-5`) or one epoch.
- If the model underfits, try LoRA rank `32` or three epochs.
