"""Central configuration constants for the final-compliant pipeline.

The current competition constraints require one open LLM <=5B parameters,
offline inference, and no embedding/reranker/RAG models. YAML files can still
hold runtime settings, but core invariants live here so every runner shares the
same defaults.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_MODEL = "Qwen/Qwen3.5-4B"

GPU_MEM_UTIL = 0.80

# --- Legacy: only used by solve.py (v01_baseline / v02_alpha / v02_beta). ---
# v02_gamma uses route-specific replacements in src.sc_policy instead.
MARGIN_LOW = 0.15  # replaced by sc_policy.MARGIN_LOW_BY_ROUTE
SC_N = 5           # replaced by sc_policy.SC_N_DEFAULT / SC_N_STEM
SC_TEMP = 0.6      # canonical copy; sc_policy re-exports from here

# Legacy per-route token budgets (replaced by sc_policy.TOKENS_BY_ROUTE).
TOK = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

FALLBACK = "A"
MAX_CHOICES = 26
