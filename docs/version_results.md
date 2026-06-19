# Version Results

> Scores are from the **HackAIthon Bảng C leaderboard** (public test set).
> Model: `Qwen/Qwen3.5-4B` on vLLM throughout.

| Version | Goal | Architecture | Key changes | Leaderboard | Inference time | s/question |
| --- | --- | --- | --- | --- | --- | --- |
| `v01_baseline` | Establish the simplest same-model baseline. | Single LLM free-form answer parsing; no routing, no guided-choice, no SC. | Updated baseline to use the same configured primary model as later versions. | 28.73% | 1761.9 s inference loop (2144.3 s total) | 3.81 s/q |
| `v02_alpha` | Test whether routing + constrained answer extraction fixes free-form parsing failures. | Rule router + two-pass guided-choice decoding. | Added READING / STEM / KNOWLEDGE / SAFETY routing and constrained final letter extraction via valid answer labels. No self-consistency. | 60.48% | 43.7 s inference loop (139.7 s total) | 0.09 s/q |
| `v02_beta` | Improve correctness with S4 escalation. | Rule router + two-pass guided-choice + per-question self-consistency. | Added STEM SC, low-margin KNOWLEDGE SC, and reason/purpose READING SC. Per-question loop; no wave batching. | 80.13% | 18411.7 s inference loop (18508.6 s total) | 39.77 s/q |
| `v02_gamma` | Keep beta-style accuracy gains while improving throughput and robustness. | Wave-batched router/guided-choice/SC pipeline. | Batches all first passes and all escalations; adaptive STEM SC depth; option shuffle de-bias; per-wave checkpointing. | **85.31%** | not separately recorded; 6424.4 s total | 12.77 s/q |

## Key Notes

- Final-compliant inference uses one model only: `Qwen/Qwen3.5-4B`.
- `configs/pipeline_config.yaml` now marks final inference as offline, no external APIs, no embedding model, no reranker model, and no RAG.
- `src/config.py` holds shared Python constants such as `LLM_MODEL`, `GPU_MEM_UTIL`, `FALLBACK`, token budgets, and legacy S4 defaults.
- `v02_gamma` uses the refactored wave pipeline: `src/wave_solver.py`, `src/batch_extract.py`, and `src/sc_policy.py`.
- S5 semantic routing and RAG are not part of final-compliant runners because they require extra embedding/reranker models.
- Runtime numbers are from local/Colab runs and can vary by GPU, vLLM version, warmup, and `safe-mode` settings.
