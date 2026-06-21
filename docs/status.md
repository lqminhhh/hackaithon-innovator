# Project Status

> Last updated: 2026-06-20. This file gives AI agents fast context on where
> the project stands. Read this before touching any code.

## What this project is

Vietnamese multiple-choice QA system for HackAIthon 2026 Bang C.
Input: JSON of questions with choices. Output: `submission.csv` (`qid,answer`).
Scored on a private set of ~2000 questions on a 16 GB VRAM GPU.

## Competition constraints (non-negotiable)

- One open LLM only: `Qwen/Qwen3.5-4B` (<=5B params)
- Offline inference, no internet, no external APIs
- No embedding model, reranker, RAG, or second LLM
- Target hardware: 16 GB VRAM (unknown judge card, may have desktop/browser overhead)

## Current best runner

**`v03_beta`** — **85.75%** on the public 463-question set.
Submission: `data/submissions/submission_v03_beta.csv`

> v03_beta = v03_alpha router (hardened rules, no `n_choices >= 8`) + fixed
> margin computation. KNOWLEDGE SC now fires for low-confidence items.
> The margin fix **is committed** in `src/extract.py` and `src/batch_extract.py`
> (continuation scoring + `safe_margin` guard). The run was on 24 GB VRAM; judge
> card is 16 GB.

Architecture:
1. Parse input JSON/CSV via `src/parser.py` + `src/data_loader.py`
2. Route each question deterministically via `src/router.py` (READING / STEM / SAFETY / KNOWLEDGE)
3. Two-pass guided-choice: reason freely, then constrain to a valid letter via
   `src/batch_extract.py`. Pass 2 scores every legal label's logprob as the next
   token after `Đáp án: ` and computes a softmax margin
   (`prob(top1) - prob(top2)`) that drives Wave 2 escalation.
4. Wave-batched self-consistency escalation via `src/wave_solver.py` + `src/sc_policy.py`:
   - STEM: always SC, adaptive depth n=3 (high margin) or n=7 (low margin)
   - KNOWLEDGE: SC n=5 when margin < 0.20
   - READING: SC n=3 for reason/purpose questions
   - SAFETY: force refusal label when harmful + refusal option present
5. Option shuffle de-bias across SC samples
6. Checkpoint per wave; atexit writes submission on crash

## Score progression

| Version | File | Score | s/question | Notes |
|---|---|---|---|---|
| v01_baseline | `src/v01_baseline.py` | 28.73% | 3.81 | |
| v02_alpha | `src/v02_alpha.py` | 60.48% | 0.09 | |
| v02_beta | `src/v02_beta.py` | 80.13% | 39.77 | |
| v02_gamma | `src/v02_gamma.py` | 85.31% | 12.77 | |
| v03_alpha | `src/v02_gamma.py` + new parser | 84.23% | 3.87 | Router regression; margin bug made KNOWLEDGE SC dead |
| v03_beta | `src/v02_gamma.py` + new parser + margin fix | **85.75%** | 4.78 | **Current best** — margin fix in `src/extract.py` / `src/batch_extract.py` |

Full details: `docs/version_results.md`

## VRAM safety settings (tuned for unknown 16 GB judge cards)

| Setting | Normal | Safe mode (`--safe-mode`) |
|---|---|---|
| `gpu_memory_utilization` | 0.80 (12.8 GB) | 0.70 (11.2 GB) |
| `max_model_len` | 4096 | 4096 |
| `max_num_seqs` | 16 | 4 |

Model weights (FP16): ~8 GB. KV cache budget in normal mode: ~4.3 GB.
These settings are in `src/config.py`, `src/sc_policy.py`, and `configs/pipeline_config.yaml`.

## Key files and what they do

### Config
- `src/config.py` — LLM_MODEL, GPU_MEM_UTIL, FALLBACK, legacy constants
- `src/sc_policy.py` — route-specific SC policy, margin thresholds, token budgets, option shuffle
- `configs/pipeline_config.yaml` — vLLM settings, runner config, quantisation settings

### Runners (entry points)
- `src/v02_gamma.py` — current best, wave-batched (the one to use)
- `src/run.py` — S7 never-crash sequential runner with checkpoint/resume/always-emit
- `src/v01_baseline.py`, `src/v02_alpha.py`, `src/v02_beta.py` — older versions, kept for eval comparison
- `src/main.py` — S0 fallback runner (writes all FALLBACK, no model)

### Core pipeline
- `src/wave_solver.py` — Wave1 (batch first passes) + Wave2 (batch SC escalations) + trace writer
- `src/batch_extract.py` — batched guided-choice extraction (vLLM continuation
  scoring via `batch_score_continuations`; HF sequential fallback)
- `src/solve.py` — per-question solver (used by older runners, not v02_gamma)
- `src/router.py` — deterministic rule router (READING/STEM/SAFETY/KNOWLEDGE)
- `src/parser.py` — question parsing, context splitting, flag extraction
- `src/extract.py` — guided-choice prompt builder, per-label continuation
  scoring (`batch_score_continuations`), `softmax_margin` / `safe_margin`,
  `GuidedChoiceExtractor` for single-question paths

### Model loading
- `src/llm.py` — vLLM wrapper with Qwen thinking-mode support
- `src/models.py` — model loading (vLLM or HuggingFace fallback)
- `src/reasoning_agent.py` — high-level LLM interface used by solvers

### Infrastructure
- `src/version_runner.py` — shared CLI args and agent loading for older runners
- `src/data_loader.py` — load questions (JSON/CSV), write submission CSV
- `src/normaliser.py` — answer normalisation

## Tests

Run with: `python3.11 -m pytest tests/ -v -m "not slow"`

| Test file | What it covers |
|---|---|
| `test_run.py` | S7 never-crash runner: G1-G4 guarantees, checkpoint/resume, SIGTERM |
| `test_s0_io.py` | Config defaults, I/O contract, UTF-8, letter range |
| `test_llm_s1.py` | vLLM wrapper constructor, batching, thinking-mode |
| `test_adaptive_sc.py` | Adaptive SC depth, wave2 escalation (requires torch) |
| `test_parser.py` | Question parsing, context splitting |
| `test_extract.py` | Guided-choice extraction, margin math, continuation scoring |
| `test_batch_extract.py` | Batched vLLM extraction regression guard (margin collapse) |
| `test_guided_choice.py` | Label mapping and constrained decoding |
| `test_normaliser.py` | Answer normalisation |
| `test_route_prompts.py` | Router + prompt integration |
| `test_solve_s4.py` | Per-question solver (requires torch) |
| `test_vllm_label_map.py` | Label map edge cases |
| `test_pipeline_smoke.py` | Output format smoke test |

## Evaluation

- Reference answers: `data/reference/reference_answers.csv` (91.58% leaderboard, not gold)
- Submissions: `data/submissions/submission_v0*.csv`
- Traces: `data/traces/trace_v0*.jsonl` (include route, path, margin, votes, escalation_reason)
- Notebook: `notebooks/evaluation.ipynb` — runs all 9 analysis sections, exports to `reports/eval/`
- Reports: `reports/eval/*.csv` (version_summary, question_type_summary, deltas, persistent_failures, regressions, audit_queue, per_question_matrix)

## What is NOT used in final inference

These exist in the repo for historical/analysis purposes but are banned by competition rules:
- RAG / FAISS / retrieval (`faiss-cpu` is in requirements but not used at inference)
- Embedding models (`sentence-transformers` is in requirements but not used at inference)
- S5 semantic router (removed — required embedding model)
- Secondary LLM ensemble
- Any external API or internet access

## Recent changes (this session)

### Margin computation fix — committed to `src/extract.py` + `src/batch_extract.py`

This was the #1 accuracy blocker for v03. In v02_gamma and v03_alpha, every
Wave 1 trace showed `margin: 1.0` (or, after an intermediate patch, a
degenerate constant). KNOWLEDGE SC (`margin < 0.20`) never fired because the
solver believed every first-pass answer was maximally confident.

**Root cause — truncated top-k logprobs on a single greedy decode**

The old vLLM path in `_vllm_batch_extract` issued one `raw_generate` call per
question with `allowed_token_ids` and `logprobs=min(n_labels, 64)`. vLLM
returned logprobs only for the single sampled next token in that decode's
top-k list. For a 4-choice question, three labels stayed at `-inf`; for an
11-choice question, ten did. `softmax_margin` then hit its single-finite-value
early exit and returned `1.0` — maximum confidence — for every question.
Downstream effects:

- `wave_solver.py` compared every margin against `MARGIN_LOW_BY_ROUTE` thresholds
  and concluded nothing was low-confidence.
- KNOWLEDGE SC (`MARGIN_LOW_BY_ROUTE["KNOWLEDGE"] = 0.20`) was effectively dead
  in v03_alpha (0 `wave_knowledge_sc` fires on the public set).
- STEM adaptive depth (`stem_sc_n`) always picked the high-margin branch (n=3)
  because every margin read as high.
- v03_beta's +0.44 pt gain over v03_alpha came entirely from restoring real
  margins so escalation policy could discriminate.

**Fix part 1 — `safe_margin` guard (`ef9ee6f`)**

Added `safe_margin(logprobs, expected_labels)` in `src/extract.py`. If a
multi-option question (`expected_labels >= 2`) returns fewer than two finite
logprobs, the extraction is malformed. `softmax_margin` alone would return
`1.0` (skip escalation); `safe_margin` inverts that to `0.0` (force
escalation). Healthy outputs still delegate to `softmax_margin`.

```
finite_count < 2  →  margin = 0.0   (escalate)
finite_count >= 2 →  margin = softmax_margin(logprobs)
```

This was a safety net on the old top-k path: it stopped the "all margins 1.0"
symptom but margins were still not *real* — only one label had a score, so
every item escalated (the "aggressive SC" problem).

**Fix part 2 — prompt-logprob continuation scoring (`da667a0`)**

Replaced the single-decode top-k approach with `batch_score_continuations` in
`src/extract.py`, shared by both `GuidedChoiceExtractor` and
`_vllm_batch_extract`.

Algorithm per `(prompt, label)` pair:

1. `build_label_token_map(tokenizer, valid_labels)` maps each legal letter to a
   single token id (bare label preferred; `" {label}"` fallback for tokenizers
   that only expose whitespace-prefixed forms).
2. Encode the full prompt with `add_special_tokens=True`.
3. For each legal label, append that label's token id to the prompt token
   sequence and submit as a separate request with `prompt_token_ids`.
4. Run all `(question, label)` continuations in **one batched** `raw_generate`
   call (`temperature=0.0`, `max_tokens=1`, `prompt_logprobs=1`). The generated
   token is ignored; we read the logprob of the **appended** prompt token from
   `output.prompt_logprobs[-1][token_id]`.
5. Build `{label → logprob}` per question. Missing continuations stay at
   `-inf`.

Properties:

- Every legal label gets a directly comparable finite logprob, including
  11-choice questions where top-k truncation previously dropped most labels.
- Heterogeneous choice sets across a wave batch correctly: 3-label and
  2-label questions in the same call produce 5 continuation requests total.
- Prefix sharing: all labels for one question reuse the same encoded prompt
  prefix; only the final token differs.

**Margin computation after the fix**

`softmax_margin(logprobs)`:

1. Collect finite logprob values.
2. Softmax with log-sum-exp stabilisation (`exp(v - max)`).
3. Return `prob(rank1) - prob(rank2)`.

Typical healthy margins are in `(0, 1)` — e.g. a clear winner at -0.2 vs
runner-up at -2.0 gives a margin well below the KNOWLEDGE threshold of 0.20,
triggering SC. Close calls (e.g. -0.2 vs -0.5) produce small margins.

`safe_margin` is still applied in `GuidedChoiceExtractor.extract` (single-question
path) as a guard when continuation scoring fails partially. The batched vLLM
path in `_vllm_batch_extract` calls `softmax_margin` directly because
continuation scoring recovers finite logprobs for every legal label on healthy
runs. `_fallback_choice` returns `margin=0.0` on extraction exceptions.

**HuggingFace fallback** — unchanged path: `_hf_batch_extract` scores labels
sequentially via `ReasoningAgent.score_valid_labels` and applies
`softmax_margin`. HF scores all labels independently so truncation does not
apply.

**Files touched**

| File | Change |
|---|---|
| `src/extract.py` | `batch_score_continuations`, `_continuation_sampling_params`, `_continuation_logprob`, `safe_margin`, refactored `GuidedChoiceExtractor` |
| `src/batch_extract.py` | `_vllm_batch_extract` delegates to `batch_score_continuations`; removed `_get_logprob` / per-question `allowed_token_ids` decode |
| `src/llm.py` | `raw_generate` passthrough for token-id requests (no chat template) |
| `tests/test_extract.py` | Margin math, 11-choice continuation scoring, logprob-required contract |
| `tests/test_batch_extract.py` | Regression guard: batched path must produce per-question margins in `(0, 1)`, one engine call with one request per label |

**Observed impact (v03_beta run, 463 questions)**

- `wave_knowledge_sc=1` — KNOWLEDGE SC fired for at least one low-confidence item
  (was 0 in v03_alpha).
- Score: 85.75% vs 84.23% (v03_alpha) and 85.31% (v02_gamma).
- Traces in `data/traces/trace_v03_beta.jsonl` now carry heterogeneous margins
  instead of the degenerate `1.0` constant.

### v03_alpha: Router hardened for 2000-question private set (score: 84.23%)

**Safety detection** — replaced 13 broad `_HARMFUL_TERMS` (generic words like
"trộm", "vũ khí", "tấn công" that matched historical/encyclopedic content)
with high-precision two-tier detection:
- `_HARMFUL_INTENT_PHRASES` (26 actionable patterns like "cách hack", "làm thế
  nào để phá hoại", "hiệu quả nhất để")
- `_HARMFUL_KEYWORDS` (7 specific dangerous terms like "chế tạo bom", "phát tán
  tài liệu mật")
- Result: all 6 true safety items caught (was 4), 0 false positives on benign
  historical content (was 6 false `is_harmful` flags on war/crime passages)

**STEM detection** — removed the `n_choices >= 8` rule from `_looks_quantitative`
that was mislabeling 13 knowledge questions as STEM. Added stronger keywords
("tốc độ", "gia tốc", "lực", "khối lượng", "kỳ vọng", "hệ phương trình").
Removed noisy "giá trị". Route counts: stem 216->201, knowledge 141->155.

**Reading detection** — unchanged (already perfect 100/100).

**Routing confusion matrix (new):**
```
              knowledge  reading  safety  stem
knowledge          151        0       0     4    = 155
reading              0      100       0     0    = 100
safety               1        0       6     0    =   7
stem                 2        0       0   199    = 201
```
7 mismatches vs true labels (was 13). All 7 mismatched items are currently
answered correctly by v02_gamma.

### VRAM safety hardened

- `GPU_MEM_UTIL`: 0.85 -> 0.80 (12.8 GB, leaves 3.2 GB for OS/driver/desktop)
- `max_model_len` fallback: 8192 -> 4096 (prevents KV cache blowup if YAML missing)
- `max_num_seqs`: unlimited -> 16 in normal mode, 4 in safe mode

### Trace logging improved

`Wave2Result` dataclass now carries actual SC votes and escalation reason.
`write_traces` outputs real `votes` list and `escalation_reason` field instead
of hardcoded `"votes": []`. Existing traces still have empty votes (pre-fix).

## Known bugs and accuracy blockers

### FIXED: margin computation (committed in `src/extract.py` / `src/batch_extract.py`)

Margins were all `1.0` in v02_gamma/v03_alpha traces because vLLM's top-k
logprob list only surfaced the sampled token. Continuation scoring via
`batch_score_continuations` now recovers a finite logprob per legal label;
`softmax_margin` produces real `prob(top1) - prob(top2)` values that drive
Wave 2 escalation. `safe_margin` guards the single-question extractor against
partially malformed outputs. See **Margin computation fix** above for full
detail.

Remaining margin-related risks:

- If `prompt_logprobs` is unavailable from the vLLM engine, continuation scoring
  returns all `-inf` and extraction falls back to `_fallback_choice` (`margin=0.0`,
  first sorted label) — escalates but answer quality is poor.
- HF path does not use continuation scoring; it relies on per-label completion
  scoring in `ReasoningAgent`, which is correct but slower.

### Error breakdown by route (v02_gamma baseline, 44 errors / 463 questions)

| Route | Errors | Total | Error Rate | SC fires? | Root cause |
|---|---|---|---|---|---|
| knowledge | **26** | 143 | **18.2%** | Now fires when margin < 0.20 (was dead in v02/v03_alpha) | No SC rescue was biggest leak in v02 |
| stem | 13 | 216 | 6.0% | Always | 3 had correct first_answer broken by SC vote |
| reading | 5 | 100 | 5.0% | Only 15 reason/purpose | Most reading errors get no SC |
| safety | 0 | 7 | 0% | Forced answer | Perfect |

### SC net impact (v02_gamma)

- SC rescued 12 wrong first-answers to correct: **+12**
- SC broke 3 correct first-answers to wrong: **-3**
- **Net: +9**

The 3 SC breaks are: `test_0222` (10-choice), `test_0227`, `test_0432`.

### 21 persistent failures (wrong across ALL versions through v02_gamma)

These questions likely need a capability the model lacks, not a prompt tweak.
See `reports/eval/persistent_failures.csv`.

## Improvement priorities for v3

1. **Tune KNOWLEDGE SC threshold and depth** — v03_beta fired `wave_knowledge_sc=1`
   on 463 questions. On 2000 private questions with real margins, this will fire
   more. Review whether `MARGIN_LOW_BY_ROUTE["KNOWLEDGE"] = 0.20` is the right
   threshold, and whether n=5 SC samples is the right depth for knowledge items.

2. **Protect first_answer from SC on high-choice questions** — the 3 SC breaks
   in v02_gamma were all high-choice items (>=8 options). Option shuffle with
   many choices is more likely to confuse the model. Consider skipping SC or
   weighting the first-pass answer more heavily for these.

3. **Expand reading SC** — only 15/100 reading questions get SC (reason/purpose
   keyword match). The 5 reading errors are all non-reason questions. Consider
   broader SC coverage for reading.

4. **Re-run v03_beta on 16 GB judge-equivalent settings** — margin fix source is
   committed; confirm score holds under `GPU_MEM_UTIL=0.80`, `max_num_seqs=16`,
   and safe-mode fallback before private-set submission.

## Remaining work

- Docker image (`Dockerfile` exists but not yet finalized — will be last step)
- `run.sh` entrypoint currently calls `src/run.py` (sequential); should be updated to call `src/v02_gamma.py` or pass safe-mode flags before submission
- Private test set is ~2000 questions (~4.3x public set); wall-clock estimate: 2-4 hours on judge GPU

## Docs index

| File | Purpose |
|---|---|
| `docs/status.md` | This file — current project state for AI context |
| `docs/planning_v3.md` | Build spec: segments S0-S8, architecture, invariants |
| `docs/version_results.md` | Score and runtime log per version |
| `docs/note_v3.md` | Design rationale and decision log |
| `docs/research_v3.md` | Evidence map for architectural choices |
