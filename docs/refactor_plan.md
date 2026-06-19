# Pipeline Version Refactor Plan

Goal: keep a small, reproducible set of Python runner files so all important compliant architecture versions can be rerun and compared with exact submission files, trace files, accuracy, and inference time.

This plan intentionally renames the active version ladder to keep only the versions that matter going forward.

## Organizer Constraints

Final private-test inference must satisfy:

- Use **one LLM model only**.
- The model must be open-source/open-weight with a license that permits the competition use case.
- The model must be `<= 5B` parameters.
- The private-test environment may be internet-isolated.
- Organizer machine VRAM is expected to be **16GB**.

Implications for this project:

- `Qwen/Qwen3.5-4B` is the current target model.
- Current S5 semantic routing is **not final-inference compliant** because it uses `BAAI/bge-m3`.
- Current RAG is **not final-inference compliant** because it uses an embedding model and optionally a reranker model.
- Ensemble/secondary-model paths are not final-inference compliant.
- S5/RAG can still be kept for offline analysis, but not as retained final runners.

## Versions To Retain

### 1. `v01_baseline`

Purpose: preserve the simplest baseline architecture, but update the primary model so it uses the same model family as the v02 runs.

Current intent:

- Keep the original `v01_baseline` logic/architecture.
- Do **not** use the old Qwen Instruct model.
- Use the same configured primary model as other current versions, currently `Qwen/Qwen3.5-4B`.

Expected output paths:

- Submission: `data/submissions/submission_v01_baseline.csv`
- Trace: `data/traces/trace_v01_baseline.jsonl`

### 2. `v02_alpha`

Purpose: represent the improved route-aware direct-answer baseline.

Mapping:

- New `v02_alpha` = old `v02_beta`

Reason:

- Original `v02_alpha` performed poorly.
- Original `v02_beta` was mostly the corrected/updated version of alpha.
- Keeping both would create noise, so the old beta becomes the new alpha.

Architecture:

- Parser
- Layer-1 router
- Route-specific prompt
- Guided-choice answer extraction
- Logprob margin
- No S4 self-consistency
- No S5 semantic router
- No RAG

Expected output paths:

- Submission: `data/submissions/submission_v02_alpha.csv`
- Trace: `data/traces/trace_v02_alpha.jsonl`

### 3. `v02_beta`

Purpose: represent the S4 self-consistency/escalation architecture.

Mapping:

- New `v02_beta` = old `v02_gamma`

Architecture:

- Parser
- Layer-1 router
- Route-specific prompt
- Guided-choice answer extraction
- Logprob margin
- S4 route-specific escalation:
  - STEM self-consistency
  - low-margin knowledge self-consistency
  - reading reason/purpose self-consistency
  - forced safety refusal
- No S5 semantic router
- No RAG

Expected output paths:

- Submission: `data/submissions/submission_v02_beta.csv`
- Trace: `data/traces/trace_v02_beta.jsonl`

## Versions Not Retained As Final Runners

### `v02_gamma` / old `v02_s5_no_rag`

Do not retain as a final runner.

Reason:

- It uses S5 semantic routing.
- Current S5 loads `BAAI/bge-m3`, which is a second model at inference time.
- This violates the organizer's "one LLM model only" clarification.

Keep S5 artifacts only for offline analysis:

- error analysis
- route-shadow diagnostics
- discovering deterministic route rules that can be rewritten without an embedding model

### RAG variants

Do not retain as final runners.

Reason:

- Current RAG loads `BAAI/bge-m3`.
- Current RAG may load `Qwen/Qwen3-Reranker-0.6B`.
- Both violate the one-model final-inference constraint.

RAG can remain as an offline research branch, but should not be part of the final private-test architecture unless the organizer explicitly allows embedding/reranker models, which current clarification does not.

## Naming Policy Going Forward

Use the simplified ladder below:

| New Version | Old Equivalent | Main Feature |
| --- | --- | --- |
| `v01_baseline` | old v01 baseline, model updated | simplest LLM baseline |
| `v02_alpha` | old `v02_beta` | route-aware guided choice |
| `v02_beta` | old `v02_gamma` | S4 self-consistency/escalation |

Avoid keeping low-value or non-compliant historical runner names once the refactor is complete.

If a new `v02_gamma` is introduced later, it must still satisfy the one-model rule. Candidate directions:

- same-Qwen verifier/arbiter
- deterministic S5-derived route rules
- safer self-consistency gating
- final-safe memory/runtime mode

## Runner Design Preference

Preferred shape:

- Keep version-specific Python files as thin runners/wrappers.
- Share common implementation code where possible.
- Each runner should consistently write:
  - submission CSV
  - per-question trace JSONL
  - route/path counts
  - total runtime
  - inference-loop runtime

Possible files:

- `src/v01_baseline.py`
- `src/v02_alpha.py`
- `src/v02_beta.py`

Future alternative:

- one generic runner plus YAML configs under `configs/runs/`

For now, Python files are acceptable because the goal is easy reruns and simple inspection.

## Trace Logging Requirement

Every retained runner should produce per-question traces.

Minimum trace fields:

- `qid`
- `answer`
- `route`
- `path`
- `margin`
- `first_answer`
- `votes`
- `layer1_route`
- `semantic_route`
- `route_override`
- `override_blockers`
- `rag_used`
- `rag_top_score`
- `error`
- `elapsed_seconds`

Some fields can be `null` for versions that do not use that feature.

For final-compliant runners, S5/RAG fields should be present but null/false:

- `semantic_route: null`
- `route_override: false`
- `rag_used: false`
- `rag_top_score: null`

This keeps evaluation notebooks schema-stable without implying those features ran.

## 16GB VRAM Safety Mode

Final runners should support conservative vLLM settings for the organizer machine:

- one model only: `Qwen/Qwen3.5-4B`
- `gpu_memory_utilization`: target `0.65` to `0.75`
- `max_model_len`: target `4096` unless longer context is proven necessary
- small self-consistency batches, ideally `1` or `2`
- no S5 embedder
- no RAG embedder
- no reranker
- no secondary model

The goal is not only compliance, but also avoiding out-of-memory crashes on 16GB VRAM.

## Evaluation Workflow Tie-In

After rerunning all retained versions, the evaluation notebook should compare:

- `data/submissions/submission_v01_baseline.csv`
- `data/submissions/submission_v02_alpha.csv`
- `data/submissions/submission_v02_beta.csv`

against:

- `data/reference/reference_answers.csv`

The notebook should also load matching trace files from `data/traces/` when available.
