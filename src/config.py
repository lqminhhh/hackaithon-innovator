"""Central configuration constants for the v3 single-model pipeline.

This module mirrors the S0/S1 build contract in ``docs/planning_v3.md``.
YAML files can still hold experiment-specific settings, but core invariants
live here so every runner can share the same defaults.

v3 is a single ≤5B model on vLLM with no RAG, embedder, reranker, or second
model — those were removed (illegal under the v3 rules and measured to hurt).
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_MODEL = "Qwen/Qwen3.5-4B"

GPU_MEM_UTIL = 0.90  # single model owns the card now (no RAG contention)
MARGIN_LOW = 0.15
SC_N = 5
SC_TEMP = 0.6

TOK = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

FALLBACK = "A"
MAX_CHOICES = 26
