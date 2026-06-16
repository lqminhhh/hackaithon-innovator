"""Central configuration constants for the v2 pipeline.

This module mirrors the S0/S1 build contract in ``docs/planning_v2.md``.
YAML files can still hold experiment-specific settings, but core invariants
live here so every runner can share the same defaults.
"""

LLM_MODEL = "Qwen/Qwen3-8B-AWQ"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

GPU_MEM_UTIL = 0.85
MARGIN_LOW = 0.15
SC_N = 5
SC_TEMP = 0.6
RERANK_MIN = 0.5

FORCE_RETRIEVE_DOMAINS = {
    "vn_law",
    "vn_decree",
    "vn_admin",
    "local_facts",
}

TOK = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

FALLBACK = "A"
MAX_CHOICES = 26
