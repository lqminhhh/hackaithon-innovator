# Project Status

## Constraints

- One LLM ≤ 5B params · no closed-source API · no internet at inference
- No separate embedding / reranker / secondary model
- Safe on a 16 GB VRAM machine
- Target model: `Qwen/Qwen3.5-4B`

## Active Runners

| Version | File | Description |
| --- | --- | --- |
| `v01_baseline` | `src/v01_baseline.py` | LLM-only baseline (no routing, no SC) |
| `v02_gamma` | `src/v02_gamma.py` | Wave-batched escalation + adaptive SC (current best) |

Shared implementation: `src/version_runner.py`

Each runner writes:
- Submission CSV → `data/submissions/`
- Trace JSONL → `data/traces/`

```bash
python src/v01_baseline.py --input <path/to/test.json>
python src/v02_gamma.py   --input <path/to/test.json>
```

Smoke-test (5 questions):

```bash
python src/v01_baseline.py --input <path/to/test.json> --safe-mode --limit 5
python src/v02_gamma.py   --input <path/to/test.json> --safe-mode --limit 5
```

## Version Results

See [`docs/version_results.md`](version_results.md) for the full table with descriptions.

| Version | Leaderboard Score | s/question |
| --- | --- | --- |
| `v01_baseline` | 28.73% | — |
| `v02_alpha` | 60.48% | — |
| `v02_beta` | 80.13% | — |
| `v02_gamma` | **85.31%** | 12.77 s/q |

## What To Do Next

1. Tune `MARGIN_LOW`, `SC_N_STEM`, `TOK["STEM"]` on a held-out dev split (S8 in `planning_v3.md`) — this is where the remaining gap to 91.58% lives.
2. Investigate the STEM bucket specifically (largest accuracy gap).

