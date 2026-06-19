# Project Status

This repo is being refactored toward a final HackAIthon Bảng C submission that follows the organizer's latest constraints.

## Current Constraints

Final private-test inference should use:

- one LLM only
- model size `<= 5B`
- no closed-source API
- no internet/search calls
- no separate embedding/reranker/secondary models
- safe operation on an expected 16GB VRAM machine

Current target model:

- `Qwen/Qwen3.5-4B`

## Final-Compliant Runner Set

The retained runnable versions are:

| Version | File | Meaning |
| --- | --- | --- |
| `v01_baseline` | `src/v01_baseline.py` | LLM-only baseline using the shared Qwen3.5-4B model |
| `v02_alpha` | `src/v02_alpha.py` | route-aware guided-choice baseline; this corresponds to the old `v02_beta` |
| `v02_beta` | `src/v02_beta.py` | S4 self-consistency/escalation; this corresponds to the old `v02_gamma` |

The shared implementation lives in:

- `src/version_runner.py`

Each retained runner writes:

- submission CSV under `data/submissions/`
- trace JSONL under `data/traces/`

Default commands:

```bash
python src/v01_baseline.py --safe-mode
python src/v02_alpha.py --safe-mode
python src/v02_beta.py --safe-mode
```

Smoke-test commands:

```bash
python src/v01_baseline.py --safe-mode --limit 5
python src/v02_alpha.py --safe-mode --limit 5
python src/v02_beta.py --safe-mode --limit 5
```

## Version Mapping

The active version names were simplified:

| New Name | Old Equivalent | Reason |
| --- | --- | --- |
| `v01_baseline` | old baseline, model updated | simplest baseline, now uses Qwen3.5-4B |
| `v02_alpha` | old `v02_beta` | corrected route-aware guided-choice version |
| `v02_beta` | old `v02_gamma` | best final-compliant S4 version |

Old `v02_s5_no_rag` is not retained as a final runner because it uses S5 semantic routing with `BAAI/bge-m3`, which is a second model.

## Non-Final / Offline-Only Components

The following can still be useful for research or error analysis, but should not be part of final inference unless organizer rules change:

- S5 semantic router: `src/semantic_router.py`, `src/semantic_shadow.py`, `scripts/shadow_semantic_routes.py`, `configs/semantic_router_config.yaml`, `data/route_prototypes.yaml`
- RAG: `src/rag.py`, `scripts/build_vmlu_index.py`
- older retrieval/indexing experiments: `src/retrieval_agent.py`, `scripts/build_index.py`
- ensemble/secondary model logic: `src/ensemble_agent.py`

## Evaluation Setup

Reference file:

- `data/reference/reference_answers.csv`

Submission files:

- `data/submissions/*.csv`

Trace files:

- `data/traces/*.jsonl`

Evaluation plan:

- `docs/evaluation_plan.md`

Important wording: `reference_answers.csv` scored 91.58%, so notebook metrics should be called **agreement with reference**, not true hidden-test accuracy.

## What To Do Next

1. Run the three retained versions in Colab with `--safe-mode`.
2. Confirm each run produces both submission and trace files.
3. Build/run the evaluation notebook against `data/reference/reference_answers.csv`.
4. Use the evaluation results to decide the next final-compliant improvement.

Likely next final-compliant improvement:

- same-Qwen verifier/arbiter
- deterministic router-rule improvements derived from S5 analysis
- safer self-consistency gating for speed and VRAM

Avoid:

- final inference with S5 embedder
- final inference with RAG embedder/reranker
- second LLM or ensemble
- internet/search during private-test inference
