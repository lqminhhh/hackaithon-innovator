# Project Status

> Last updated: 2026-07-07. This file gives AI agents fast context on where
> the project stands. Read this before touching any code.

## What this project is

Vietnamese multiple-choice QA system for HackAIthon 2026 Bang C.
Input: JSON or CSV of questions with choices. Competition output:
`/output/pred.csv` (`qid,answer`).
Scored on a private set of ~2000 questions on a 16 GB VRAM GPU.

## Competition constraints (non-negotiable)

- One open LLM only: `Qwen/Qwen3.5-4B` (<=5B params)
- Offline inference, no internet, no external APIs
- No embedding model, reranker, RAG, or second LLM
- Target hardware: 16 GB VRAM (unknown judge card, may have desktop/browser overhead)

## Current final runner

**`v03_gamma`** — **85.96%** on the public 463-question set.

> `v03_gamma` keeps the hardened `v03_alpha` router, restores useful compute for
> hard KNOWLEDGE and READING cases, adds length-safe extraction so long-context
> reasoning does not overflow the 4096-token vLLM limit, and now wraps the
> shipped runner in a stronger always-emit safety layer.
>
> We are choosing `v03_gamma` as the final submission candidate not because the
> later real-margin idea was wrong, but because the later `v03_delta` /
> `v03_epsilon` experiments were much heavier operationally: about 4x slower on
> the public run and still OOM-prone on 16 GB VRAM. On a private set of ~2000
> questions, reliability matters more than squeezing out the last public-set
> accuracy point.

Architecture:
1. Parse input JSON/CSV via `src/parser.py` + `src/data_loader.py`
2. Route each question deterministically via `src/router.py` (READING / STEM / SAFETY / KNOWLEDGE)
3. Two-pass guided-choice: reason freely, then constrain to a valid letter via `src/batch_extract.py`
4. Wave-batched self-consistency escalation via `src/wave_solver.py` + `src/sc_policy.py`:
   - STEM: always SC; adaptive depth exists in the design, but in practice `v03_gamma`
     behaves mostly as a route-driven extra-compute path because the cheap margin
     proxy is not a faithful per-label confidence signal
   - KNOWLEDGE: SC n=5 when margin < 0.20
   - READING: SC n=3 for reason/purpose questions
   - SAFETY: force refusal label when harmful + refusal option present
5. Option shuffle de-bias across SC samples
6. Checkpoint per wave; fallback-prefilled output plus atomic always-emit on
   exception or signal
7. Additive deterministic vLLM warmup before the real run to reduce first-run
   Triton JIT latency spikes without changing route / SC policy

## Score progression

| Version | File | Score | s/question | Notes |
|---|---|---|---|---|
| v01_baseline | `src/v01_baseline.py` | 28.73% | 3.81 | |
| v02_alpha | `src/v02_alpha.py` | 60.48% | 0.09 | |
| v02_beta | `src/v02_beta.py` | 80.13% | 39.77 | |
| v02_gamma | `src/v02_gamma.py` | 85.31% | 12.77 | Original wave-batched best |
| v03_alpha | `src/v03_gamma.py` + new parser | 84.23% | 3.87 | Router regression; margin bug makes KNOWLEDGE SC dead |
| v03_gamma | `src/v03_gamma.py` + hardened parser + compute/context fixes | **85.96%** | 7.98 | Final submission candidate; best speed/reliability tradeoff |
| v03_delta | later experimental branch | **87.04%** | 27.53 | Higher public accuracy, but ~4x slower and not judge-safe on 16 GB |
| v03_epsilon | later experimental branch | pending final score | similar to delta | Delta-compatible safety attempt; still hit OOM in late Wave 2 on 16 GB |

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
- `src/v03_gamma.py` — current final runner, wave-batched (the one to use)
- `src/v02_gamma.py` — compatibility shim for older commands/imports
- `src/run.py` — S7 never-crash sequential fallback utility, not the submission default
- `src/v01_baseline.py`, `src/v02_alpha.py`, `src/v02_beta.py` — older versions, kept for eval comparison
- `src/main.py` — S0 fallback runner (writes all FALLBACK, no model)

### Core pipeline
- `src/wave_solver.py` — Wave1 (batch first passes) + Wave2 (batch SC escalations) + trace writer
- `src/batch_extract.py` — batched guided-choice extraction via vLLM
- `src/solve.py` — per-question solver (used by older runners, not the final `v03_gamma` path)
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
| `test_gamma_entrypoints.py` | Final runner rename shim + shipped entrypoint checks |
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

### Final choice: why `v03_gamma` over `v03_delta` / `v03_epsilon`

Later experiments after `v03_gamma` validated the original design idea:
**real continuation-scored margins do help accuracy**. On the public set, that
work reached `87.04%` in `v03_delta`.

However, we are **not** choosing that branch as the final runner.

Why:
- **Runtime blew up** — the real-margin extraction path expanded each question
  into one continuation-score request per legal answer label, pushing runtime to
  about **27.53 s/question**, versus about **7.98 s/question** for `v03_gamma`.
- **OOM risk remained real on 16 GB cards** — even after later safety work,
  the delta-compatible path still ran into OOM during late Wave 2 on
  judge-like hardware.
- **Private-set risk is multiplicative** — the final evaluation set is ~2000
  questions, so a branch that is already near the edge on 463 public questions
  is not the conservative submission choice.

How to frame `v03_gamma` correctly:
- `v03_gamma` is **not a wrong design**; it is a faster, cheaper approximation
  of the same broad route-aware compute-allocation idea.
- The later delta work showed that exact margins can improve the policy, but
  that exact version is substantially heavier and less reliable.
- So `v03_gamma` should be described as the **final efficiency-first,
  judge-safer operating point**, not as a failed or broken system.

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

Important framing:
- `v03_gamma` still spends extra compute on STEM through route-based SC.
- What it does **not** have is a fully faithful per-label margin like the later
  delta experiment. In practice, gamma behaves more like a fast confidence pass
  plus route-aware escalation than a true margin-calibrated adaptive system.
- That was acceptable for the final branch because the exact-margin alternative
  became too slow and too memory-fragile for judge deployment.

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

### Phase 0 speed instrumentation landed

We completed the first phase of `docs/speed_optimization_plan.md` without
changing answer policy:

- `src/llm.py` now records generated-token counts in `GenerationOutput`
  (`num_generated_tokens`) so token spend can be measured directly.
- `src/reasoning_agent.py` can now return structured generation outputs in
  addition to plain text, while keeping the old text-only path for existing
  callers.
- `src/wave_solver.py` now measures and writes per-question compute metadata:
  - `gen_tokens_wave1`
  - `gen_tokens_wave2`
  - `wave2_sample_tokens`
  - `sc_n`
  - `wave1_time_share`
  - `wave2_time_share`
  - `attributed_time_seconds`
  - backend/runtime metadata in `runtime_info`
- `src/v03_gamma.py` now logs the effective runtime environment at startup:
  safe mode, chosen engine settings, GPU name/VRAM, CUDA capability, vLLM
  version, prefix-caching status, backend loaded, and fallback reason if vLLM
  did not load.
- `predict.py` now writes per-question `time` values into
  `submission_time.csv` using trace-attributed runtime instead of a flat
  average copied across all rows.

This was intentionally a measurement-only pass: it should not change routing,
SC policy, prompts, or nominal accuracy. Its purpose is to make later speed
work diagnosable on judge hardware.

### Current speed-plan state

- **Phase 0:** done
- **Phase 1:** not landed yet in this branch
- Next planned work: dynamic free-VRAM sizing for vLLM, louder backend fallback
  visibility, and a headroom retry ladder before any HuggingFace fallback

## Known bugs and accuracy blockers

### LIMITATION: gamma margins are saturated (`~1.0`) in practice

In `v03_gamma`, the margin signal used by the wave pipeline is not a faithful
continuation-scored per-label margin. In practice many traces show
`margin=1.0`, which means:

- **KNOWLEDGE SC never fires** — the `MARGIN_LOW_BY_ROUTE["KNOWLEDGE"] = 0.20`
  gate never triggers because margin is always 1.0 > 0.20.
- **Adaptive STEM SC depth never adapts** — every STEM item looks "high margin"
  and gets n=3 instead of n=7 when it should be uncertain.
- The system is therefore better understood as **route-driven compute
  allocation with a lightweight confidence proxy**, not as a fully calibrated
  adaptive-margin design.

This limitation is real, but we are accepting it in the final branch because
the exact-margin fix was operationally too expensive for the 16 GB / 2000-question
submission target.

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

## Post-submission follow-ups

1. **If revisiting exact margins, treat it as a separate deployment track** —
   the delta/epsilon work proved that real continuation-scored margins can help,
   but they must first become genuinely safe on 16 GB cards before they can be
   reconsidered for the final path.
2. **Re-analyze gamma errors by route/path before any new policy work** —
   the remaining misses are narrower now, so future changes should be driven by
   fresh error slices rather than the older broad v2 assumptions.
3. **Tune only low-risk execution knobs first** — safe-mode defaults, entrypoint
   stability, and batching/throughput hygiene are better next steps than adding
   new accuracy heuristics late.
4. **Keep legacy comparison runners intact** — they are still useful for
   evaluation and regression analysis even though they are not the final branch.

## Remaining work

- Docker image / entrypoint should stay aligned with `src/v03_gamma.py` + `--safe-mode`
- Phase 1 of `docs/speed_optimization_plan.md` is still pending: dynamic VRAM sizing, headroom retry ladder, and higher safe batching without changing answer logic
- Private test set is ~2000 questions (~4.3x public set); conservative 16 GB judge-style runs can take far longer than the old public-set extrapolation. Recent safe-mode reports suggest planning for very long wall-clock runs, potentially on the order of 30+ hours on a slow 16 GB setup

## Docs index

| File | Purpose |
|---|---|
| `docs/status.md` | This file — current project state for AI context |
| `docs/version_results.md` | Score and runtime log per version |
| `docs/speed_optimization_plan.md` | Current throughput and judge-runtime optimization roadmap |
| `docs/research_v3.md` | Evidence map for architectural choices |
| `docs/faq.md` | Runtime setup notes and common environment issues |
| `docs/report/report_vi.md` | Main Vietnamese report for judges |
| `docs/report/report_en.md` | English report |
| `docs/report/presentation_slide.pdf` | Presentation deck |
| `docs/translations/README_en.md` | English README mirror of the main Vietnamese README |
