# Version Results

> Scores are from the **HackAIthon Bảng C leaderboard** (public test set).
> Model: `Qwen/Qwen3.5-4B` on vLLM throughout.

| Version | Goal | Architecture | Key changes | Leaderboard | Inference time | s/question |
| --- | --- | --- | --- | --- | --- | --- |
| `v01_baseline` | Establish the simplest same-model baseline. | Single LLM free-form answer parsing; no routing, no guided-choice, no SC. | Updated baseline to use the same configured primary model as later versions. | 28.73% | 1761.9 s inference loop (2144.3 s total) | 3.81 s/q |
| `v02_alpha` | Test whether routing + constrained answer extraction fixes free-form parsing failures. | Rule router + two-pass guided-choice decoding. | Added READING / STEM / KNOWLEDGE / SAFETY routing and constrained final letter extraction via valid answer labels. No self-consistency. | 60.48% | 43.7 s inference loop (139.7 s total) | 0.09 s/q |
| `v02_beta` | Improve correctness with S4 escalation. | Rule router + two-pass guided-choice + per-question self-consistency. | Added STEM SC, low-margin KNOWLEDGE SC, and reason/purpose READING SC. Per-question loop; no wave batching. | 80.13% | 18411.7 s inference loop (18508.6 s total) | 39.77 s/q |
| `v02_gamma` | Keep beta-style accuracy gains while improving throughput and robustness. | Wave-batched router/guided-choice/SC pipeline. | Batches all first passes and all escalations; adaptive STEM SC depth; option shuffle de-bias; per-wave checkpointing. | **85.31%** | not separately recorded; 6424.4 s total | 12.77 s/q |
| `v03_alpha` | Harden router for 2000-question private set generalization. | Same wave pipeline as v02_gamma; router changes only. | Removed `n_choices >= 8 → STEM` rule; tightened harmful detection to actionable intent phrases; added STEM keywords. Routes: stem=201, knowledge=155, reading=100, safety=7. | 84.23% | 1790.3 s inference loop (2801.7 s total) | 3.87 s/q |

## Key Notes

- Final-compliant inference uses one model only: `Qwen/Qwen3.5-4B`.
- `configs/pipeline_config.yaml` now marks final inference as offline, no external APIs, no embedding model, no reranker model, and no RAG.
- `src/config.py` holds shared Python constants such as `LLM_MODEL`, `GPU_MEM_UTIL`, `FALLBACK`, token budgets, and legacy S4 defaults.
- `v02_gamma` uses the refactored wave pipeline: `src/wave_solver.py`, `src/batch_extract.py`, and `src/sc_policy.py`.
- S5 semantic routing and RAG are not part of final-compliant runners because they require extra embedding/reranker models.
- Runtime numbers are from local/Colab runs and can vary by GPU, vLLM version, warmup, and `safe-mode` settings.
- `v02_gamma_router_v2` runtime was measured on a 24 GB VRAM card; judge hardware is 16 GB.

## Router v2 Regression Analysis

`v03_alpha` scored **-1.08 pts** below `v02_gamma` (84.23% vs 85.31%).

**Root cause:** The `n_choices >= 8 → STEM` rule removed 13 knowledge questions
from STEM routing. In v02_gamma those 13 items received think-mode reasoning +
always-on SC. In v02_gamma_router_v2 they were reclassified as KNOWLEDGE, but
because **margin computation is broken (all margins = 1.0)**, KNOWLEDGE SC never
fires. They fell back to a cheap no-think direct pass — worse than STEM treatment.

**Conclusion:** The router fix is correct in principle but premature. The right
sequence is: fix margin computation first → KNOWLEDGE SC activates → *then*
reclassify those 13 items. Until margins are fixed, the `n_choices >= 8` rule
was accidentally beneficial by forcing better compute on ambiguous items.

**Current best submission remains `v02_gamma` (85.31%).**
