"""Model loading and quantisation configuration.

Centralises all model instantiation so the rest of the codebase never
calls `from_pretrained` directly.

On CUDA GPUs: loads in 4-bit via bitsandbytes to fit on a single 24 GB card.
On Apple Silicon: loads in float16 on CPU (MPS VRAM is too small for 7B+
  models; CPU can use full system RAM and MPS ops will spill automatically).
On CPU: loads in float32.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path

from src.config import GPU_MEM_UTIL, LLM_MODEL

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"

# Let MPS allocations spill into system RAM instead of hard-crashing
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _get_torch():
    import torch

    return torch


def _get_device_info() -> tuple[str, object, bool]:
    """Determine the best available device, dtype, and whether 4-bit is usable.

    Returns (device_map, dtype, use_4bit).
    """
    torch = _get_torch()
    if torch.cuda.is_available():
        return "auto", torch.float16, True
    # Apple Silicon: load to CPU so the full system RAM is available.
    # MPS has a hard VRAM cap (~9 GB on 16 GB machines) that can't fit 7B fp16.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu", torch.float16, False
    return "cpu", torch.float32, False


def _load_model(model_id: str, device_map_override: str | None = None):
    """Load a causal LM with the best available quantisation strategy."""
    torch = _get_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_config()
    auto_device, dtype, can_4bit = _get_device_info()
    device_map = device_map_override or auto_device

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs: dict = dict(trust_remote_code=True)

    if can_4bit and cfg["quantisation"]["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map.get(cfg["quantisation"]["compute_dtype"], torch.float16)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["device_map"] = device_map
        print(f"Loading {model_id} in 4-bit on CUDA")
    else:
        load_kwargs["torch_dtype"] = dtype
        load_kwargs["device_map"] = device_map
        is_apple = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        label = "Apple Silicon (CPU + float16)" if is_apple else f"CPU ({dtype})"
        print(f"Loading {model_id} on {label}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_primary_model(
    device_map: str | None = None,
    model_id: str | None = None,
):
    """Load the configured primary model, or an explicit override."""
    cfg = _load_config()
    return _load_model(model_id or cfg["models"]["primary"], device_map)


def load_secondary_model(device_map: str | None = None):
    """Load Gemma-2-9B-IT."""
    cfg = _load_config()
    return _load_model(cfg["models"]["secondary"], device_map)


def load_embedder(device: str | None = None):
    """Load the Vietnamese-tuned sentence embedding model.

    Pass device="cpu" to keep GPU free for vLLM.
    """
    from sentence_transformers import SentenceTransformer

    cfg = _load_config()
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    return SentenceTransformer(cfg["models"]["embedder"], **kwargs)


def load_vllm_primary(
    model_id: str | None = None,
    *,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
):
    """Load primary model via vLLM for fast batched inference (CUDA only)."""
    from src.llm import LLM

    cfg = _load_config()
    vllm_cfg = cfg.get("vllm", {})
    chosen_model = model_id or cfg.get("models", {}).get("primary") or LLM_MODEL
    quantization = "awq" if "awq" in chosen_model.lower() else None
    return LLM(
        model=chosen_model,
        quantization=quantization,
        gpu_memory_utilization=(
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else vllm_cfg.get("gpu_memory_utilization", GPU_MEM_UTIL)
        ),
        max_model_len=(
            max_model_len if max_model_len is not None else vllm_cfg.get("max_model_len", 8192)
        ),
        max_num_seqs=(
            max_num_seqs if max_num_seqs is not None else vllm_cfg.get("max_num_seqs")
        ),
        enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
    )
