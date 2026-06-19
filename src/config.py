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

# Per-route low-margin thresholds. Margin = prob(top1) - prob(top2); a question
# whose first-pass margin is below its route threshold is escalated. Distributions
# differ by route, so a single global threshold under/over-escalates (Issue 6).
MARGIN_LOW = {
    "READING": 0.10,
    "STEM": 0.15,
    "KNOWLEDGE": 0.20,
    "SAFETY": 0.05,
}

# Self-consistency depth.
#   SC_N      — default sample count for non-STEM escalation
#   SC_N_STEM — adaptive STEM depth: shallow vote when the first pass is confident,
#               deep vote when it is not (STEM always votes; no early-exit).
SC_N = 5
SC_N_STEM = {"high": 3, "low": 7}
SC_TEMP = 0.6
SC_TOP_P = 0.95

TOK = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

FALLBACK = "A"
MAX_CHOICES = 26
