# Project Status

> Last updated: 2026-06-21. This file gives AI agents fast context on where
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

**`v03_gamma`** — **85.96%** on the public 463-question set.

> `v03_gamma` keeps the hardened `v03_alpha` router, restores useful compute for
> hard KNOWLEDGE and READING cases, and adds length-safe Wave 2 extraction so
> long-context SC does not overflow the 4096-token vLLM limit.

Architecture:
1. Parse input JSON/CSV via `src/parser.py` + `src/data_loader.py`
2. Route each question deterministically via `src/router.py` (READING / STEM / SAFETY / KNOWLEDGE)
3. Two-pass guided-choice: reason freely, then constrain to a valid letter via `src/batch_extract.py`
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
| v02_gamma | `src/v02_gamma.py` | 85.31% | 12.77 | Original wave-batched best |
| v03_alpha | `src/v02_gamma.py` + new parser | 84.23% | 3.87 | Router regression; margin bug makes KNOWLEDGE SC dead |
| v03_gamma | `src/v02_gamma.py` + hardened parser + compute/context fixes | **85.96%** | - | Current best |

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
- `src/batch_extract.py` — batched guided-choice extraction via vLLM
- `src/solve.py` — per-question solver (used by older runners, not v02_gamma)
- `src/router.py` — deterministic rule router (READING/STEM/SAFETY/KNOWLEDGE)
- `src/parser.py` — question parsing, context splitting, flag extraction
- `src/extract.py` — single-question guided-choice extraction + logprob margin

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
| `test_extract.py` | Guided-choice extraction |
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

### v03_gamma: hardened router + targeted compute recovery (score: 85.96%)

`v03_gamma` is the current best public-set run. It keeps the `v03_alpha`
router improvements but changes the compute policy instead of reverting to the
old broad `n_choices >= 8 -> STEM` route hack.

Key changes:
- **High-choice KNOWLEDGE recovery** — 8+ choice knowledge questions stay on
  the `knowledge` route but receive extra SC / think-mode treatment instead of
  being reclassified as STEM.
- **Broader READING rereads** — READING SC now covers detail-lookup questions
  (dates, first-occurrence, exact evidence, `theo ngữ cảnh`, etc.), not just
  reason/purpose questions.
- **Broader KNOWLEDGE rescue** — extra compute now covers ambiguous options and
  combination-style choices in addition to low-margin items.
- **Length-safe Wave 2 extraction** — long SC prompts are compacted before
  guided-choice extraction, preventing 4096-token context overflow on long
  reading passages.

Net result: `v03_gamma` beats both `v02_gamma` (85.31%) and `v03_alpha`
(84.23%) on the public set while preserving the cleaner v3 router.

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

### BUG: All margins are 1.0 (broken margin computation)

Every single question in v02_gamma traces has `margin=1.0` and `votes=[]`.
The logprob margin extraction in the wave pipeline (`src/batch_extract.py` /
`src/extract.py`) is returning 1.0 for everything. This means:

- **KNOWLEDGE SC never fires** — the `MARGIN_LOW_BY_ROUTE["KNOWLEDGE"] = 0.20`
  gate never triggers because margin is always 1.0 > 0.20.
- **Adaptive STEM SC depth never adapts** — every STEM item looks "high margin"
  and gets n=3 instead of n=7 when it should be uncertain.
- The entire margin-based adaptive system is flying blind.

**This is the #1 blocker.** Fixing margin computation would auto-activate
KNOWLEDGE SC for uncertain items, covering the biggest error bucket (26 errors).

### Error breakdown by route (v02_gamma, 44 errors / 463 questions)

| Route | Errors | Total | Error Rate | SC fires? | Root cause |
|---|---|---|---|---|---|
| knowledge | **26** | 143 | **18.2%** | Never (margin bug) | No SC rescue; biggest leak |
| stem | 13 | 216 | 6.0% | Always | 3 had correct first_answer broken by SC vote |
| reading | 5 | 100 | 5.0% | Only 15 reason/purpose | Most reading errors get no SC |
| safety | 0 | 4 | 0% | Forced answer | Perfect |

### SC net impact

- SC rescued 12 wrong first-answers to correct: **+12**
- SC broke 3 correct first-answers to wrong: **-3**
- **Net: +9** (positive but the 3 breaks are real cost)

The 3 SC breaks are: `test_0222` (10-choice), `test_0227`, `test_0432`.
High-choice-count questions (>=8 options) are especially vulnerable to SC
option-shuffle confusion.

### 21 persistent failures (wrong across ALL versions)

These questions are wrong in v01 through v02_gamma. They likely need a
capability the model lacks, not a prompt tweak. See `reports/eval/persistent_failures.csv`.

## Improvement priorities for v3

**CRITICAL ORDER: fix margin first, then activate router-v2.**

1. **Fix margin computation** — investigate `src/batch_extract.py` and
   `src/extract.py` for why logprob margin is always 1.0 in the wave pipeline.
   Without real margins the adaptive system is blind. The router-v2 regression
   (-1.08 pts) confirmed this: moving items from STEM to KNOWLEDGE is safe only
   when KNOWLEDGE SC can rescue low-confidence items. Fix this first.

2. **After margins work: v03_alpha becomes v03_beta** — the new parser is
   already in `src/parser.py`. Once margins are real, KNOWLEDGE SC will fire
   for the 13 items reclassified from STEM, likely recovering the -1.08 pts
   and then some (26 knowledge errors currently get zero SC).

3. **Consider universal KNOWLEDGE SC as interim fallback** — if margin fix is
   not achievable before submission, run SC n=3 on ALL knowledge questions
   (~155 items) unconditionally. Cheap compute, directly covers the biggest
   error bucket.

3. **Protect first_answer from SC on high-choice questions** — for questions
   with >=8 options, option shuffle with SC is more likely to confuse the model.
   Skip SC or weight the first-pass answer higher in the vote.

5. **Expand reading SC** — only 15/100 reading questions currently get SC
   (reason/purpose keyword match). The 5 reading errors are all non-reason
   questions. Consider SC for all reading questions or those with long contexts.

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
