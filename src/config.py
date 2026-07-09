"""Central configuration loader for the final-compliant pipeline.

`configs/pipeline_config.yaml` is the runtime source of truth. This module
loads that file once and exposes the settings as Python constants for the rest
of the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _PROJECT_ROOT / "configs" / "pipeline_config.yaml"


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    with _CFG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = load_project_config()

LLM_MODEL = str(_CFG["models"]["primary"])

GPU_MEM_UTIL = float(_CFG["vllm"]["gpu_memory_utilization"])
MAX_MODEL_LEN = int(_CFG["vllm"]["max_model_len"])
MAX_NUM_SEQS = _CFG["vllm"].get("max_num_seqs")
ENABLE_PREFIX_CACHING = bool(_CFG["vllm"].get("enable_prefix_caching", True))
ENABLE_CHUNKED_PREFILL = bool(_CFG["vllm"].get("enable_chunked_prefill", True))
SAFE_GPU_MEM_UTIL = float(_CFG["safe_vllm"]["gpu_memory_utilization"])
SAFE_MAX_MODEL_LEN = int(_CFG["safe_vllm"]["max_model_len"])
SAFE_MAX_NUM_SEQS = int(_CFG["safe_vllm"]["max_num_seqs"])
SAFE_DYNAMIC_HEADROOM_GB = float(_CFG["safe_vllm"].get("dynamic_headroom_gb", 1.0))
SAFE_HEADROOM_LADDER_GB = tuple(
    float(value) for value in _CFG["safe_vllm"].get("headroom_ladder_gb", [1.0, 2.0, 3.0])
)
SAFE_UTILIZATION_CLAMP_MIN = float(_CFG["safe_vllm"].get("utilization_clamp_min", 0.50))
SAFE_UTILIZATION_CLAMP_MAX = float(_CFG["safe_vllm"].get("utilization_clamp_max", 0.92))
SAFE_WAVE_RETRY_CHUNK_SIZES = tuple(
    int(value) for value in _CFG["safe_vllm"].get("wave_retry_chunk_sizes", [128, 64, 32, 16])
)

FALLBACK = str(_CFG["submission"]["fallback_answer"])
MAX_CHOICES = int(_CFG["question_parsing"]["max_choices"])

MARGIN_LOW = float(_CFG["legacy_solver"]["margin_low"])
SC_N = int(_CFG["legacy_solver"]["sc_n"])
SC_TEMP = float(_CFG["legacy_solver"]["temperature"])
TOK = {
    route: int(tokens)
    for route, tokens in _CFG["legacy_solver"]["tokens_by_route"].items()
}

MARGIN_LOW_BY_ROUTE = {
    route: float(value)
    for route, value in _CFG["route_policy"]["margin_low_by_route"].items()
}
SC_N_STEM = {
    key: int(value)
    for key, value in _CFG["route_policy"]["stem_sc"].items()
}
SC_N_DEFAULT = int(_CFG["route_policy"]["default_sc_n"])
SC_N_HIGH_CHOICE_KNOWLEDGE = int(_CFG["route_policy"]["high_choice_knowledge_sc_n"])
HIGH_CHOICE_KNOWLEDGE_MIN_CHOICES = int(
    _CFG["route_policy"]["high_choice_knowledge_min_choices"]
)
SC_TOP_P = float(_CFG["route_policy"]["sc_top_p"])
SC_SEED = int(_CFG["route_policy"]["sc_seed"])
SHUFFLE_OPTIONS = bool(_CFG["route_policy"]["shuffle_options"])
TOKENS_BY_ROUTE = {
    route: int(tokens)
    for route, tokens in _CFG["route_policy"]["tokens_by_route"].items()
}
WAVE2_THINK_TOKENS_BY_ROUTE = {
    route: int(tokens)
    for route, tokens in _CFG["route_policy"].get("wave2_think_tokens_by_route", {}).items()
}

GAMMA_GPU_MEM_UTIL = GPU_MEM_UTIL
GAMMA_MAX_MODEL_LEN = MAX_MODEL_LEN
