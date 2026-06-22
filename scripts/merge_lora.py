"""Merge a trained LoRA adapter into the base model.

The merged directory can be passed to the existing inference runners via
``--model-id outputs/finetune/..._merged`` so final inference still loads a
single model directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=_torch_dtype(model_cfg["dtype"]),
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.config.use_cache = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model written to: {output_dir}")


if __name__ == "__main__":
    main()
