"""Merge a trained LoRA adapter into the base model.

The merged directory can be passed to the existing inference runners via
``--model-id outputs/finetune/..._merged`` so final inference still loads a
single model directory.
"""

from __future__ import annotations

import argparse
import json
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
  pip install -U peft accelerate datasets safetensors sentencepiece protobuf PyYAML

After the restart, rerun scripts/merge_lora.py.
""".strip()


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_dtype_name(name: str) -> str:
    normalized = name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if normalized in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Config requested bf16, but this GPU does not support bf16. "
                "Use model.dtype: auto or fp16."
            )
        return "bf16"
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    if normalized in {"fp32", "float32"}:
        return "fp32"
    raise ValueError(f"Unsupported dtype: {name}")


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
    resolved_dtype = model_cfg.get("resolved_dtype", model_cfg["dtype"])
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            torch_dtype=_torch_dtype(resolved_dtype),
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        if _is_qwen35_support_error(exc):
            raise RuntimeError(QWEN35_TRANSFORMERS_HELP) from exc
        raise


def _copy_base_model_file(base_model: str, output_dir: str, filename: str, *, required: bool) -> None:
    """Copy optional base-model metadata after merge."""
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


MULTIMODAL_METADATA_TERMS = (
    "image",
    "video",
    "vision",
    "visual",
    "mm_",
    "multimodal",
    "processor",
)


def _strip_multimodal_metadata(value: Any, *, strip_auto_map: bool = False) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if (strip_auto_map and key == "auto_map") or any(
                term in key_lower for term in MULTIMODAL_METADATA_TERMS
            ):
                continue
            cleaned[key] = _strip_multimodal_metadata(item, strip_auto_map=strip_auto_map)
        return cleaned
    if isinstance(value, list):
        return [_strip_multimodal_metadata(item, strip_auto_map=strip_auto_map) for item in value]
    return value


def _flatten_text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text_config = cfg.get("text_config")
    if not isinstance(text_config, dict):
        return cfg

    flattened = dict(cfg)
    for key, value in text_config.items():
        flattened.setdefault(key, value)
    flattened.pop("text_config", None)
    return flattened


def _fill_required_text_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    if "pad_token_id" not in cfg:
        cfg["pad_token_id"] = cfg.get("eos_token_id", 151645)
    if "bos_token_id" not in cfg and "eos_token_id" in cfg:
        cfg["bos_token_id"] = cfg["eos_token_id"]
    return cfg


def _normalize_qwen35_text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if str(cfg.get("model_type", "")).startswith("qwen3_5"):
        cfg["model_type"] = "qwen3_5"
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        cfg["use_cache"] = True
        cfg = _fill_required_text_defaults(cfg)
    return cfg


def _sanitize_text_only_metadata(output_dir: str) -> None:
    root = Path(output_dir)
    config_path = root / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    cfg = _flatten_text_config(cfg)
    cfg = _strip_multimodal_metadata(cfg)
    cfg = _normalize_qwen35_text_config(cfg)

    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer_config_path = root / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config = _strip_multimodal_metadata(tokenizer_config)
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for filename in [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "video_processor_config.json",
    ]:
        path = root / filename
        if path.exists():
            path.unlink()


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
    model_cfg["resolved_dtype"] = _resolve_dtype_name(str(model_cfg.get("dtype", "auto")))
    print(f"Merge precision: {model_cfg['resolved_dtype']}", flush=True)

    _assert_model_supported(model_cfg["base_model"], trust_remote_code=trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=trust_remote_code)
    base = _load_base_model(model_cfg, trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.config.use_cache = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    _copy_base_model_file(model_cfg["base_model"], output_dir, "generation_config.json", required=False)
    tokenizer.save_pretrained(output_dir)
    _sanitize_text_only_metadata(output_dir)
    print(f"Merged model written to: {output_dir}")


if __name__ == "__main__":
    main()
