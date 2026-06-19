# HackAIthon 2026 - Bang C Innovator

Vietnamese multiple-choice QA system for HackAIthon 2026. The current retained
architecture is final-inference compliant under the latest organizer rules:

- one open LLM only: `Qwen/Qwen3.5-4B`
- model size <= 5B parameters
- offline inference, no external APIs or internet calls
- no embedding model, reranker, RAG, semantic-router model, or second LLM
- target hardware: 16 GB VRAM

This README is intentionally short for now. See `docs/planning_v3.md`,
`docs/version_results.md`, and `docs/refactor_plan.md` for the deeper history.

## Current Architecture

The strongest retained runner is `v02_gamma`:

1. Parse input JSON/CSV into normalized questions.
2. Route each question with deterministic rules:
   `READING`, `STEM`, `SAFETY`, otherwise `KNOWLEDGE`.
3. Run a two-pass answer process:
   first reason freely, then extract one valid answer label.
4. Escalate harder cases with self-consistency:
   STEM, low-margin KNOWLEDGE, and reason/purpose READING.
5. Use wave batching so vLLM processes first passes and escalations in large batches.
6. Always write a complete submission and trace file.

No final runner uses RAG or S5 semantic routing.

## Main Runners

| Version | File | Purpose |
| --- | --- | --- |
| `v01_baseline` | `src/v01_baseline.py` | Same-model free-form baseline |
| `v02_alpha` | `src/v02_alpha.py` | Rule routing + guided-choice extraction |
| `v02_beta` | `src/v02_beta.py` | Per-question S4 self-consistency |
| `v02_gamma` | `src/v02_gamma.py` | Wave-batched adaptive self-consistency, current best |

Latest tracked results are in `docs/version_results.md`.

## Quick Start

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the current best pipeline:

```bash
python src/v02_gamma.py \
  --input data/public-test_1780368312.json \
  --output data/submissions/submission_v02_gamma.csv \
  --trace-output data/traces/trace_v02_gamma.jsonl
```

Safer 16 GB VRAM mode:

```bash
python src/v02_gamma.py \
  --input data/public-test_1780368312.json \
  --output data/submissions/submission_v02_gamma.csv \
  --trace-output data/traces/trace_v02_gamma.jsonl \
  --safe-mode
```

Smoke test with a small limit:

```bash
python src/v02_gamma.py \
  --input data/public-test_1780368312.json \
  --limit 5 \
  --safe-mode
```

## Project Layout

```text
configs/
  pipeline_config.yaml      Final-compliant runtime settings
  prompts.yaml              Prompt templates

data/
  public-test_1780368312.json
  reference/                Reference answers for evaluation
  submissions/              Generated submission CSVs
  traces/                   Per-question trace JSONL files

docs/
  planning_v3.md            Current plan
  version_results.md        Score/runtime log
  evaluation_plan.md        Notebook evaluation design

src/
  v01_baseline.py
  v02_alpha.py
  v02_beta.py
  v02_gamma.py
  version_runner.py         Shared runner utilities
  wave_solver.py            Wave-batched S4 logic
  batch_extract.py          Batched guided-choice extraction
  sc_policy.py              Adaptive self-consistency policy
  router.py                 Rule router
  parser.py                 Input parser
  reasoning_agent.py        LLM wrapper

tests/
  Unit tests for parser, routing, guided choice, vLLM wrapper, and adaptive SC
```

## Configuration Notes

- `configs/pipeline_config.yaml` states the final inference constraints and vLLM defaults.
- `src/config.py` contains stable Python constants such as `LLM_MODEL`, `FALLBACK`, and token budgets.
- `src/sc_policy.py` contains the route-specific self-consistency policy used by `v02_gamma`.

## Evaluation

Reference answers live in:

```text
data/reference/reference_answers.csv
```

Submission files live in:

```text
data/submissions/
```

Trace files live in:

```text
data/traces/
```

Use these with the planned evaluation notebook workflow in `docs/evaluation_plan.md`.

## Important Exclusions

The repository may still contain old/offline-analysis artifacts from earlier
experiments, but final inference should not use:

- RAG / FAISS / retrieval
- embedding models
- reranker models
- S5 semantic router
- secondary LLM ensemble

Those paths were removed from the retained final runners because they violate
the updated one-model rule and were not improving the measured result.
