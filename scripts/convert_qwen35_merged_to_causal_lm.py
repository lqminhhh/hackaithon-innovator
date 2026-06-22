"""Convert a merged Qwen3.5 wrapper checkpoint into text-only CausalLM form.

Some Transformers builds save merged Qwen3.5 weights with keys like
``model.language_model.layers...`` while a text-only ``Qwen3_5ForCausalLM``
config expects ``model.layers...``. This script creates a new model directory
with remapped weights and a text-only config so HuggingFace/vLLM do not try to
load multimodal processor or visual weights.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

MULTIMODAL_METADATA_TERMS = (
    "image",
    "video",
    "vision",
    "visual",
    "mm_",
    "multimodal",
    "processor",
)

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
PROCESSOR_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        return [
            _strip_multimodal_metadata(item, strip_auto_map=strip_auto_map)
            for item in value
        ]
    if isinstance(value, str):
        lowered = value.lower()
        if any(term in lowered for term in MULTIMODAL_METADATA_TERMS):
            return None
    return value


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text_config = cfg.get("text_config")
    if isinstance(text_config, dict):
        merged = dict(text_config)
        for key in ["model_type", "torch_dtype", "transformers_version"]:
            if key in cfg:
                merged.setdefault(key, cfg[key])
    else:
        merged = dict(cfg)

    merged = _strip_multimodal_metadata(merged, strip_auto_map=True)
    merged["model_type"] = "qwen3_5"
    merged["architectures"] = ["Qwen3_5ForCausalLM"]
    merged["use_cache"] = True
    if "pad_token_id" not in merged:
        merged["pad_token_id"] = merged.get("eos_token_id", 151645)
    if "bos_token_id" not in merged and "eos_token_id" in merged:
        merged["bos_token_id"] = merged["eos_token_id"]
    return merged


def _copy_metadata(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in input_dir.iterdir():
        if source.name in PROCESSOR_FILES:
            continue
        if source.name == "model.safetensors.index.json":
            continue
        if source.suffix in WEIGHT_SUFFIXES:
            continue
        if source.is_file():
            shutil.copyfile(source, output_dir / source.name)


def _sanitize_copied_metadata(output_dir: Path) -> None:
    for filename in ("tokenizer_config.json", "generation_config.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        value = _load_json(path)
        value = _strip_multimodal_metadata(value, strip_auto_map=True)
        _write_json(path, value)

    for filename in PROCESSOR_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _weight_files(input_dir: Path) -> list[Path]:
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        files = sorted(set(index.get("weight_map", {}).values()))
        return [input_dir / filename for filename in files]

    files = sorted(input_dir.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors weights found in {input_dir}")


def _remap_key(key: str) -> str | None:
    if ".visual." in key or key.startswith("visual.") or key.startswith("model.visual."):
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return "model." + key[len("language_model.") :]
    return key


def _infer_config_from_shapes(config: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    inferred = dict(config)

    embed_shape = shapes.get("model.embed_tokens.weight")
    if embed_shape and len(embed_shape) == 2:
        inferred["vocab_size"] = embed_shape[0]
        inferred["hidden_size"] = embed_shape[1]

    gate_shape = shapes.get("model.layers.0.mlp.gate_proj.weight")
    if gate_shape and len(gate_shape) == 2:
        inferred["intermediate_size"] = gate_shape[0]

    layer_ids = []
    for key in shapes:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
    if layer_ids:
        inferred["num_hidden_layers"] = max(layer_ids) + 1

    return inferred


def _convert_weights(input_dir: Path, output_dir: Path) -> dict[str, tuple[int, ...]]:
    tensors: dict[str, torch.Tensor] = {}
    for weight_file in _weight_files(input_dir):
        shard = load_file(weight_file)
        for key, tensor in shard.items():
            new_key = _remap_key(key)
            if new_key is not None:
                tensors[new_key] = tensor

    if "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()

    shapes = {key: tuple(tensor.shape) for key, tensor in tensors.items()}
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    return shapes


def convert(input_dir: Path, output_dir: Path) -> None:
    if not (input_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {input_dir}")

    _copy_metadata(input_dir, output_dir)
    _sanitize_copied_metadata(output_dir)
    config = _text_config(_load_json(input_dir / "config.json"))
    shapes = _convert_weights(input_dir, output_dir)
    config = _infer_config_from_shapes(config, shapes)
    _write_json(output_dir / "config.json", config)

    print(f"Converted model written to: {output_dir}")
    print(f"architectures: {config.get('architectures')}")
    print(f"has vocab_size: {'vocab_size' in config}")
    print(f"hidden_size: {config.get('hidden_size')}")
    print(f"intermediate_size: {config.get('intermediate_size')}")
    print(f"num_hidden_layers: {config.get('num_hidden_layers')}")
    print(f"pad_token_id: {config.get('pad_token_id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 wrapper weights to CausalLM keys")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    convert(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
