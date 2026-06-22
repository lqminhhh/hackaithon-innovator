"""Sanitize a merged Qwen3.5 text-only model directory for vLLM.

Run this after LoRA merge if vLLM tries to load image/vision processors or
complains about missing ``visual.*`` weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MULTIMODAL_METADATA_TERMS = (
    "image",
    "video",
    "vision",
    "visual",
    "mm_",
    "multimodal",
    "processor",
)

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
)


def _strip_multimodal_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if key == "auto_map" or any(term in key_lower for term in MULTIMODAL_METADATA_TERMS):
                continue
            cleaned[key] = _strip_multimodal_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_multimodal_metadata(item) for item in value]
    return value


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_model_dir(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    cfg = _load_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config JSON: {config_path}")

    cfg = _strip_multimodal_metadata(cfg)
    if cfg.get("model_type") == "qwen3_5":
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        cfg["use_cache"] = True
    _write_json(config_path, cfg)

    for json_path in model_dir.glob("*.json"):
        if json_path.name in {"config.json", "generation_config.json", "model.safetensors.index.json"}:
            continue
        metadata = _load_json(json_path)
        if metadata is None:
            continue
        _write_json(json_path, _strip_multimodal_metadata(metadata))

    for filename in PROCESSOR_FILES:
        path = model_dir / filename
        if path.exists():
            path.unlink()

    remaining = []
    for json_path in model_dir.glob("*.json"):
        text = json_path.read_text(encoding="utf-8").lower()
        if any(term in text for term in MULTIMODAL_METADATA_TERMS):
            remaining.append(json_path.name)

    print(f"Sanitized: {model_dir}")
    print(f"architectures: {cfg.get('architectures')}")
    print(f"remaining suspicious json files: {remaining}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize merged Qwen3.5 text-only model metadata")
    parser.add_argument("model_dir", help="Path to merged model directory")
    args = parser.parse_args()

    sanitize_model_dir(Path(args.model_dir))


if __name__ == "__main__":
    main()
