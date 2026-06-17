# V2 Progress Handoff

This document is the practical catch-up file for humans and AI helpers. It
summarizes what the v2 plan is trying to build, what the repo currently
implements, what has been measured, and what should happen next.

Companion docs:

- [planning_v2.md](/Users/minhle/Documents/hackaithon-innovator/docs/planning_v2.md): original build spec
- [note_v2.md](/Users/minhle/Documents/hackaithon-innovator/docs/note_v2.md): plain-language rationale
- [version_results.md](/Users/minhle/Documents/hackaithon-innovator/docs/version_results.md): measured run log

## Current Mental Model

The project is an offline Vietnamese multiple-choice QA agent for HackAIthon
Bảng C. The system reads questions, chooses one option letter, and writes a
complete CSV.

The v2 idea is:

```text
question
  -> parser
  -> Layer-1 router
  -> route-specific prompt
  -> guided-choice extraction with logprob margin
  -> route-specific escalation / self-consistency when needed
  -> complete submission CSV
```

The router decides how much effort a question deserves. The escalation layer
decides whether the first answer is trustworthy enough, or whether the model
should reason multiple times and vote.

Important routing rule:

```text
reading / stem / safety -> no RAG
knowledge               -> RAG later, selectively
```

## Important Drift From The Original Plan

The original `planning_v2.md` mentions `Qwen/Qwen3-8B-AWQ`. The code currently
uses:

```text
Qwen/Qwen3.5-4B
```

Reason: this is faster for Colab iteration and better aligned with the PDF's
Qwen3.5 model-family wording. A larger final model can still be tested later.

The public input also contains one 11-choice question, so the loader supports
labels up to `Z`, not only A-J.

## Active Entry Points

Use these when working on the v2 path:

- [main.py](/Users/minhle/Documents/hackaithon-innovator/main.py): S0 fallback runner, no model
- [src/v02_alpha.py](/Users/minhle/Documents/hackaithon-innovator/src/v02_alpha.py): current v2 model runner
- [src/solve.py](/Users/minhle/Documents/hackaithon-innovator/src/solve.py): current S4 solve policy

Be careful with:

- [src/pipeline.py](/Users/minhle/Documents/hackaithon-innovator/src/pipeline.py): older retrieval/ensemble pipeline, not the current v2 path
- [README.md](/Users/minhle/Documents/hackaithon-innovator/README.md): still describes the older architecture in places

## Segment Status

| Segment | Status | Main Files | Notes |
| --- | --- | --- | --- |
| S0 Skeleton + config + I/O | Done | `src/config.py`, `src/data_loader.py`, `src/io_utils.py`, `main.py` | Writes `qid,answer`; fallback answer is `A`; UTF-8; supports A-Z labels |
| S1 vLLM + model wrapper | Done structurally | `src/llm.py`, `src/models.py`, configs | Defaults to `Qwen/Qwen3.5-4B`; supports think/no-think; fake-engine tests pass |
| S2 guided choice + margin | Done | `src/extract.py`, `src/reasoning_agent.py` | Returns `ChoiceResult(letter, margin, per_letter_logprob)`; no regex fallback in v2 extraction |
| S3 Layer-1 router + prompts | Done | `src/parser.py`, `src/router.py`, `configs/prompts.yaml` | `route_l1()` can abstain; `route_question()` defaults to knowledge; refusal-trap line added to all route prompts |
| S4 escalation + self-consistency | Done and scored | `src/solve.py`, `src/v02_alpha.py` | `v02_gamma` reached 79.91%; STEM always self-consistency votes; low-margin knowledge votes; reading reason/purpose questions use targeted `n=3` self-consistency |
| S6 RAG | Done and scored | `src/rag.py`, `scripts/build_vmlu_index.py`, `src/solve.py`, `src/v02_alpha.py` | `v02_gamma_rag` scored 78.83% (-1.08 pts vs gamma); BGE-m3 + Qwen3-Reranker-0.6B; 606-chunk VMLU dev/valid index (34 knowledge-gap subjects); reranker OOM'd on 24 GB GPU (cosine-only fallback); `--use-rag` / `--no-reranker` CLI flags added |
| S7 never-crash runner + checkpoint | Not done | TBD `run.py` | Current `solve_question()` catches per-question errors, but no checkpoint/resume runner yet |
| S5 semantic router | Scaffold exists | `src/semantic_router.py`, `configs/semantic_router_config.yaml`, `data/route_prototypes.yaml` | Not integrated into main v2 runner |
| S8 dev set + ablation | Not done | TBD | Needed for principled threshold tuning and creativity story |

## Current Measured Results

See [version_results.md](/Users/minhle/Documents/hackaithon-innovator/docs/version_results.md).

Known measured v2 results:

```text
v02_alpha:     54.43%
v02_beta:      60.48%
v02_gamma:     79.91%   ← current best accuracy
v02_gamma_rag: 78.83%   ← S6 RAG run (degraded due to VRAM OOM on reranker)
```

`v02_beta` corresponds roughly to S0-S3 plus S2 margin extraction, before S4
self-consistency was run/scored.

`v02_gamma` corresponds to S4 route-specific escalation. It is the current best
accuracy result, but it is much slower:

```text
total runtime:   5412.9s
inference loop:  4987.0s
per question:    10.77s/question
output:          data/submission_v02_gamma.csv
```

`v02_gamma_rag` added S6 RAG on top of gamma (same S4 policy). It was slower
and slightly less accurate due to VRAM pressure on the 24 GB RunPod GPU:

```text
total runtime:   8788.7s
inference loop:  8352.0s
per question:    18.04s/question
output:          data/submission_v02_gamma_rag.csv (or submission_v02_rag.csv)
accuracy:        78.83%   (-1.08 pts vs gamma)
```

Root cause of regression:
- Qwen3-Reranker-0.6B OOM'd on every call (vLLM held ~20 GB; RAG added ~4 GB on same card).
- Fell back to cosine-only gating, which is weaker than cross-encoder scoring.
- STEM self-consistency slowed from ~20 s/q to ~25-45 s/q due to GPU contention.

To fix for the next RAG run:
- Lower `gpu_memory_utilization` to 0.70 in `configs/pipeline_config.yaml`, OR
- Offload BGE-m3 / reranker to CPU via `--no-reranker` + CPU embedder device, OR
- Use `max_model_len: 4096` to shrink the KV cache and free ~3 GB.

The big lesson from `v02_gamma`: self-consistency works, especially for STEM and
reading distractors, but the current per-question runner is too expensive for
comfortable iteration.

## Current Route Counts

For the public 463-question file, after S3:

```text
route_question:
  reading   = 100
  stem      = 216
  knowledge = 143
  safety    = 4

route_l1:
  reading = 100
  stem    = 216
  safety  = 4
  None    = 143
```

`None` from `route_l1()` means "use knowledge default, or let a later semantic
router review it."

`v02_gamma` path counts:

```text
direct                           = 212
stem_self_consistency            = 216
reading_reason_self_consistency = 15
low_margin_self_consistency      = 16
forced_safety                    = 4
```

## What Each Route Means

`reading`

- The question includes its own passage/context.
- Use the passage only.
- Do not retrieve.
- Current path: direct guided choice.
- Exception: questions asking reason/purpose/cause, such as "lý do", "tại sao",
  "vì sao", or "mục đích", use `reading_reason_self_consistency` with `n=3`.
- This targeted exception fixed `test_0005`, where direct scoring confidently
  picked the Pechenegs distractor (`A`) before self-consistency changed it to
  the correct slave-trade answer (`C`).

`stem`

- Calculation, formulas, quantitative reasoning, or 8+ choices.
- Do not retrieve.
- Current S4 path: first direct answer, then think-mode self-consistency vote.
- This is the main runtime bottleneck because all 216 public-test STEM
  questions currently run `stem_self_consistency` with `n=5`.

`knowledge`

- General fact/concept/recall question.
- This is the only route that should eventually use RAG.
- Current S4+S6 path: direct answer; if margin is below `MARGIN_LOW`, attempt
  RAG retrieval first (if `--use-rag`); if RAG context improves margin, return
  the RAG answer; otherwise fall back to self-consistency.

`safety`

- Harmful request with a refusal option.
- If harmful + refusal option, force the refusal label.
- Do not retrieve.

## Key Files By Responsibility

Data and output:

- [src/data_loader.py](/Users/minhle/Documents/hackaithon-innovator/src/data_loader.py)
- [src/io_utils.py](/Users/minhle/Documents/hackaithon-innovator/src/io_utils.py)
- [main.py](/Users/minhle/Documents/hackaithon-innovator/main.py)

Model and generation:

- [src/config.py](/Users/minhle/Documents/hackaithon-innovator/src/config.py)
- [src/llm.py](/Users/minhle/Documents/hackaithon-innovator/src/llm.py)
- [src/models.py](/Users/minhle/Documents/hackaithon-innovator/src/models.py)
- [src/reasoning_agent.py](/Users/minhle/Documents/hackaithon-innovator/src/reasoning_agent.py)

Routing and prompts:

- [src/parser.py](/Users/minhle/Documents/hackaithon-innovator/src/parser.py)
- [src/router.py](/Users/minhle/Documents/hackaithon-innovator/src/router.py)
- [configs/prompts.yaml](/Users/minhle/Documents/hackaithon-innovator/configs/prompts.yaml)

Answer extraction and escalation:

- [src/extract.py](/Users/minhle/Documents/hackaithon-innovator/src/extract.py)
- [src/solve.py](/Users/minhle/Documents/hackaithon-innovator/src/solve.py)

Semantic/RAG work:

- [src/rag.py](/Users/minhle/Documents/hackaithon-innovator/src/rag.py) — S6 RAGEngine (BGE-m3 + Qwen reranker, integrated into solve.py)
- [scripts/build_vmlu_index.py](/Users/minhle/Documents/hackaithon-innovator/scripts/build_vmlu_index.py) — offline index builder (VMLU dev+valid, 606 chunks)
- [src/semantic_router.py](/Users/minhle/Documents/hackaithon-innovator/src/semantic_router.py)
- [data/route_prototypes.yaml](/Users/minhle/Documents/hackaithon-innovator/data/route_prototypes.yaml)
- [src/retrieval_agent.py](/Users/minhle/Documents/hackaithon-innovator/src/retrieval_agent.py) — older retrieval implementation, not the current S6 path

## Verification Commands

Lightweight local test suite:

```bash
pytest -q \
  tests/test_solve_s4.py \
  tests/test_extract.py \
  tests/test_llm_s1.py \
  tests/test_s0_io.py \
  tests/test_parser.py \
  tests/test_normaliser.py \
  tests/test_confidence_gate.py \
  tests/test_guided_choice.py \
  tests/test_vllm_label_map.py \
  tests/test_semantic_router.py \
  tests/test_route_prompts.py \
  tests/test_pipeline_smoke.py
```

Most recent result after S6 RAG implementation:

```text
164 passed   (all tests including tests/test_rag_s6.py)
86 passed    (S6 tests only: pytest tests/test_rag_s6.py -v)
```

Run S6 tests only:

```bash
pytest tests/test_rag_s6.py -v
```

Run all tests excluding legacy retrieval (needs faiss at import time):

```bash
pytest -q --ignore=tests/test_retrieval.py
```

Run the v2 pipeline with S6 RAG:

```bash
# Build the VMLU index first (run once on the GPU machine)
python scripts/build_vmlu_index.py

# Full run with RAG + reranker
python src/v02_alpha.py \
  --input data/public-test_1780368312.json \
  --output data/submission_v02_rag.csv \
  --use-rag

# Full run with RAG, cosine-only (no reranker, saves ~1-2 GB VRAM)
python src/v02_alpha.py \
  --input data/public-test_1780368312.json \
  --output data/submission_v02_rag_norerank.csv \
  --use-rag --no-reranker
```

Full `pytest` may still fail in some local environments if `faiss` is not
installed, because `tests/test_retrieval.py` imports it at module load time.

Inspect routes without loading the model:

```bash
python - <<'PY'
from collections import Counter
from src.data_loader import load_questions
from src.parser import parse_question
from src.router import route_l1, route_question

parsed = [parse_question(q) for q in load_questions("data/public-test_1780368312.json")]
print("final:", Counter(route_question(q) for q in parsed))
print("l1:", Counter(route_l1(q) for q in parsed))
PY
```

Run the current v2 model path on a small sample:

```bash
python src/v02_alpha.py \
  --input data/public-test_1780368312.json \
  --output data/submission_s4_50.csv \
  --limit 50
```

Run the current full-scoring path:

```bash
python src/v02_alpha.py \
  --input data/public-test_1780368312.json \
  --output data/submission_v02_gamma.csv
```

Inspect only the `test_0005` parser/router regression locally:

```text
notebooks/test_parser_router.ipynb
```

Run the focused "Focused regression: `test_0005` Yaroslav reading distractor"
section. The no-model cell verifies route/gate behavior; the optional solver
cell requires a loaded model.

Run S0 fallback only:

```bash
python main.py \
  --input data/smoke-test-15.json \
  --output data/submission_s0.csv
```

## Notes For Future AI Helpers

- Do not treat [src/pipeline.py](/Users/minhle/Documents/hackaithon-innovator/src/pipeline.py) as the current architecture.
- Do not reintroduce regex answer parsing into the v2 answer path.
- Keep router and escalation separate.
- Keep RAG out of reading/stem/safety.
- Be careful with user worktree changes. At the time this doc was created,
  some docs may have been deleted or moved by the user; do not restore them
  unless asked.
- The current code favors correctness experiments over speed. Wave batching is
  still needed if S4 proves accurate but too slow.
- S6 RAG is wired in via `src/rag.py` + `scripts/build_vmlu_index.py`. The
  index (`data/vmlu_faiss.index`) and chunks (`data/vmlu_chunks.jsonl`) are
  gitignored; rebuild with `python scripts/build_vmlu_index.py` on the GPU machine.
- `HF_HUB_ENABLE_HF_TRANSFER=1` on RunPod breaks model downloads unless
  `pip install hf_transfer` is done first — or set the env var to `0`.
- Qwen3.5 requires `transformers>=5.2.0` and `vllm>=0.17.0`; older versions
  raise `KeyError: 'qwen3_5'`.

## Recommended Next Steps

**Priority 1 — Fix RAG VRAM pressure (quick win)**

The reranker OOM'd in `v02_gamma_rag` because vLLM + RAG exceeded 24 GB.
Try one of:
```bash
# Option A: lower vLLM reservation in configs/pipeline_config.yaml
#   gpu_memory_utilization: 0.70   (frees ~3-4 GB)
#   max_model_len: 4096             (shrinks KV cache)

# Option B: run without reranker (cosine-only, ~1-2 GB less)
python src/v02_alpha.py --input ... --output ... --use-rag --no-reranker

# Option C: load RAG models on CPU (add device="cpu" in src/rag.py)
```
A clean reranker run is expected to recover the ~1 pt regression and possibly
add +1 to +2 pts over gamma.

**Priority 2 — Speed: v02_delta STEM gating**

STEM self-consistency is the main runtime bottleneck (216 questions × ~30 s/q
≈ 90 min). Add a high-margin early exit:

```text
v02_delta_speed idea:
  STEM direct margin >= 0.90 -> accept direct (skip SC)
  STEM direct margin <  0.90 -> run self-consistency
```

Use a 50-question slice first and compare accuracy, path counts, and runtime.
If that is stable, run the full public test and log it as `v02_delta`.

**Priority 3 — Expand RAG corpus**

The current 606-chunk VMLU corpus is narrow (dev+valid only, 34 subjects).
To improve recall, add Vietnamese Wikipedia excerpts or the VMLU test set
Q&As (without answers) as retrieval candidates. Use `--append` in
`scripts/build_vmlu_index.py` to incrementally extend the index.
