"""Merge a trained LoRA adapter into the base model.

The merged directory can be passed to the existing inference runners via
``--model-id outputs/finetune/..._merged`` so final inference still loads a
single model directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

QWEN35_TRANSFORMERS_HELP = """
This environment cannot load Qwen/Qwen3.5 checkpoints because its
Transformers install does not recognize model_type='qwen3_5'.

In Colab, run these commands, then restart the runtime:

  pip uninstall -y transformers tokenizers
  pip install --no-cache-dir git+https://github.com/huggingface/transformers.git
  pip install -U peft accelerate datasets safetensors sentencepiece protobuf

After the restart, rerun scripts/merge_lora.py.
""".strip()


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _torch_dtype(name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _is_qwen35_support_error(exc: Exception) -> bool:
    message = str(exc)
    return "qwen3_5" in message or "does not recognize this architecture" in message


def _assert_model_supported(model_id: str, *, trust_remote_code: bool) -> None:
    try:
        AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        if _is_qwen35_support_error(exc):
            raise RuntimeError(QWEN35_TRANSFORMERS_HELP) from exc
        raise


def _load_base_model(model_cfg: dict[str, Any], *, trust_remote_code: bool):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            torch_dtype=_torch_dtype(model_cfg["dtype"]),
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        if _is_qwen35_support_error(exc):
            raise RuntimeError(QWEN35_TRANSFORMERS_HELP) from exc
        raise


def _copy_base_model_file(base_model: str, output_dir: str, filename: str, *, required: bool) -> None:
    """Copy raw base-model metadata after merge.

    Some Qwen3.5 Transformers builds save the inner text config after
    ``merge_and_unload``. vLLM expects the original top-level config, so keep
    the raw base-model config in the merged directory.
    """
    local_file = Path(base_model) / filename
    if local_file.exists():
        source = local_file
    else:
        try:
            from huggingface_hub import hf_hub_download

            source = Path(hf_hub_download(base_model, filename))
        except Exception:
            if required:
                raise
            return

    shutil.copyfile(source, Path(output_dir) / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    adapter_dir = args.adapter_dir or train_cfg["output_dir"]
    output_dir = args.output_dir or train_cfg["merged_output_dir"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    _assert_model_supported(model_cfg["base_model"], trust_remote_code=trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=trust_remote_code)
    base = _load_base_model(model_cfg, trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.config.use_cache = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    _copy_base_model_file(model_cfg["base_model"], output_dir, "config.json", required=True)
    _copy_base_model_file(model_cfg["base_model"], output_dir, "generation_config.json", required=False)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model written to: {output_dir}")


if __name__ == "__main__":
    main()
