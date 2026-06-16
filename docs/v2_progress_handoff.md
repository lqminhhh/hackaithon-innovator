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
| S4 escalation + self-consistency | Done and scored | `src/solve.py`, `src/v02_alpha.py` | `v02_gamma` reached 79.91%; STEM always self-consistency votes; low-margin knowledge votes; reading reason/purpose questions use targeted `n=3` self-consistency; no RAG yet |
| S7 never-crash runner + checkpoint | Not done | TBD `run.py` | Current `solve_question()` catches per-question errors, but no checkpoint/resume runner yet |
| S5 semantic router | Scaffold exists | `src/semantic_router.py`, `configs/semantic_router_config.yaml`, `data/route_prototypes.yaml` | Not integrated into main v2 runner |
| S6 RAG | Older implementation exists | `src/retrieval_agent.py`, `scripts/build_index.py` | Not integrated into v2 route policy yet; also `retrieve()` has a known `score_map` bug in sequential path |
| S8 dev set + ablation | Not done | TBD | Needed for principled threshold tuning and creativity story |

## Current Measured Results

See [version_results.md](/Users/minhle/Documents/hackaithon-innovator/docs/version_results.md).

Known measured v2 results:

```text
v02_alpha: 54.43%
v02_beta:  60.48%
v02_gamma: 79.91%
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
- Current S4 path: direct answer; if margin is below `MARGIN_LOW`, run
  self-consistency. RAG is deferred to S6.

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

- [src/semantic_router.py](/Users/minhle/Documents/hackaithon-innovator/src/semantic_router.py)
- [data/route_prototypes.yaml](/Users/minhle/Documents/hackaithon-innovator/data/route_prototypes.yaml)
- [src/retrieval_agent.py](/Users/minhle/Documents/hackaithon-innovator/src/retrieval_agent.py)

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

Most recent result after S4 implementation:

```text
75 passed
```

Recent focused tests after the `reading_reason_self_consistency` change:

```bash
pytest -q tests/test_solve_s4.py
pytest -q tests/test_route_prompts.py tests/test_parser.py tests/test_extract.py
```

```text
34 passed
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

## Recommended Next Step

Optimize speed while protecting the `v02_gamma` accuracy.

Do not remove the reading reason/purpose gate yet; it is cheap and fixed a real
high-confidence reading error. The highest-leverage experiment is to reduce
STEM self-consistency cost:

```text
v02_delta_speed idea:
  STEM direct margin >= 0.90 -> accept direct
  STEM direct margin <  0.90 -> run self-consistency
```

Use a 50-question slice first and compare accuracy, path counts, and runtime.
If that is stable, run the full public test and log it as `v02_delta`.

RAG should remain deferred until after speed tuning. The current bottleneck is
not missing knowledge; it is that `stem_self_consistency` ran on 216 questions
in `v02_gamma`.
