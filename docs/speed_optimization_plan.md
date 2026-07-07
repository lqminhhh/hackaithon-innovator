# Speed Optimization Plan — VietMind MCQ (v03_gamma)

> Goal: fix the 0.375 speed score by cutting per-question token spend and wall-clock
> time, with little to no accuracy loss (85.96% public / 91.58% proxy must hold).
>
> Organizer feedback addressed:
> 1. "Adjust token usage for each route so not too much token is spent on one problem."
> 2. "Log route + actual compute per question on the judge machine to detect degraded
>    fallback and adjust route thresholds instead of replacing the pipeline."

## Context

The submitted `v03_gamma` runs at ~7.98 s/question on our 24 GB dev GPU and slower on
the 16 GB judge card in safe mode. For ~2000 private questions that is hours of
wall-clock time. The accuracy design (router + wave-batched SC) is sound; the problem
is that compute is allocated far more bluntly than the design intends.

## Diagnosis (verified in code and traces)

1. **Per-route token budgets are dead config.**
   `TOKENS_BY_ROUTE` defines READING=512, STEM=3072, KNOWLEDGE=256, SAFETY=128
   (`configs/pipeline_config.yaml`), but `src/wave_solver.py` splits every generation
   batch into only two groups:
   - think-mode → `TOKENS_BY_ROUTE["STEM"]` = 3072 (`wave_solver.py:244`, `:365`)
   - no-think → `TOKENS_BY_ROUTE["READING"]` = 512 (`wave_solver.py:255`, `:377`)

   So a KNOWLEDGE question in think mode gets a 3072-token budget. This is exactly
   the organizer's "too much token on one problem" comment.

2. **The confidence gates are dead.**
   456/463 trace margins are exactly 1.0 (`softmax_margin` returns 1.0 when only one
   finite logprob comes back — `src/extract.py:58-59`). Consequences:
   - `stem_sc_n` always returns 3 (adaptive n=7 never fires)
   - `knowledge_low_margin < 0.20` never fires
   - Compute allocation is route-only, never confidence-aware.

3. **Most escalation compute is wasted re-confirmation.**
   From `data/traces/trace_v03_gamma.jsonl` (463 public questions):
   - 321/463 (69%) escalate to self-consistency
   - 232/321 (72%) of escalations return unanimous votes
   - 212/321 (66%) unanimously re-confirm the wave-1 answer (pure waste)
   - STEM = 201 questions, ALL get think-mode SC ×3 samples × up to 3072 tokens each,
     plus a 3072-token wave-1 think pass → the dominant token sink.

4. **Judge-run throughput is throttled.**
   `predict.py` defaults `safe_mode=True` → `max_num_seqs=4`,
   `gpu_memory_utilization=0.70`. Wave batching over ~2000 questions with only 4
   concurrent sequences wastes most of the batching design.

5. **No per-question compute logging exists.**
   `predict.py:52` writes the *same average time* for every row of
   `submission_time.csv`; traces contain no per-question time or token counts. The
   organizer's recommendation cannot currently be followed.

## Plan — 4 phases, ordered by risk

### Phase 0 — Instrumentation first (organizer's explicit recommendation)

Add per-question compute logging so budgets are set from measured data and a
judge-machine run can be diagnosed after the fact:

- `src/llm.py`: extend `GenerationOutput` with `num_generated_tokens`
  (length of `output.outputs[0].token_ids`).
- `src/reasoning_agent.py`: expose those counts through `generate_freeform`.
- `src/wave_solver.py`: record per question — route, think mode, sc_n, escalation
  reason, wave-1 reasoning token count, per-SC-sample token counts, per-wave wall
  time. Extend `Wave1Result` / `Wave2Result` and `write_traces` with new fields:
  `gen_tokens_wave1`, `gen_tokens_wave2`, `sc_n`, `think`, `wave1_time_share`,
  `wave2_time_share`.
- **Per-question time attribution:** attribute each wave's measured wall time to
  questions proportional to their generated-token share. `predict.py` reads this
  from the trace and writes real per-question `time` values into
  `submission_time.csv` (same `qid,answer,time` format) instead of a flat average.
- `src/v03_gamma.py`: at startup, log the effective engine config (safe mode,
  max_num_seqs, gpu_util, max_model_len, vLLM version, GPU name/VRAM via
  `torch.cuda`) and any fallback/degradation events into the trace — this is what
  identifies "degraded fallback" on the judge card.
- **Then run one instrumented public-set run** → per-route generated-token
  distributions (p50/p90/p95) that set the Phase 2 budgets.

### Phase 1 — Zero-accuracy-risk throughput (biggest wall-clock win)

- Raise safe-mode concurrency **with an OOM fallback ladder**:
  try `max_num_seqs=12`, `gpu_memory_utilization=0.78` first; on CUDA OOM at engine
  init or mid-wave, tear down and retry at `8 / 0.74`, then the current `4 / 0.70`.
  Implement in `src/v03_gamma.py` around engine construction and the wave loop —
  per-wave checkpointing already exists, so a mid-run retry resumes from checkpoint
  and never loses answers.
- Config ladder lives in `configs/pipeline_config.yaml` (`safe_vllm`).
- Keep `max_model_len=4096` unchanged.
- Answers should be identical (temperature-0 wave 1, unchanged SC seeds) — verify
  with an answer diff against `data/submissions/submission_v03_gamma.csv`.

### Phase 1b — More zero-accuracy runtime fixes (outputs stay byte-identical)

These remove pure mechanical overhead. None of them changes any prompt, sampling
parameter, or answer.

1. **Merge serialized generation batches (idle GPU tails).**
   Wave 1 and Wave 2 each run the think batch and the no-think batch as *separate
   sequential* engine calls (`wave_solver.py:239-259`, `:360-382`). While the last
   long stragglers of one batch finish, the GPU idles instead of starting the next
   group. After Phase 2 introduces more per-route groups, this cost multiplies.
   Fix: one merged `engine.generate` call per wave with **per-request
   `SamplingParams`** (vLLM accepts a params list), pre-rendering the chat template
   per prompt so think/no-think can coexist in one continuous batch. Same prompt +
   same params per item → identical outputs, but all groups share one continuous
   batch and stragglers overlap.

2. **Kill serial CPU tokenization between waves.**
   `_fit_extraction_prompt` (`wave_solver.py:156-196`) calls `tokenizer.encode` on
   every full extraction prompt one-by-one in Python — once per question in Wave 1
   and once per SC sample in Wave 2 (~5000+ serial encodes on the private set)
   while the GPU waits. Since BPE token count ≤ character count, add a fast path:
   if `len(prompt) + buffer < max_input_tokens` (characters), it provably fits —
   skip encoding. Only near-limit prompts (long reading passages) get real token
   counting. Byte-identical outputs.

3. **Raise the non-safe `max_num_seqs` and treat the safe ladder as a floor.**
   Non-safe mode hardcodes `max_num_seqs=16` (`v03_gamma.py:177`); vLLM's own
   default is 256 and it budgets KV-cache blocks under `gpu_memory_utilization`
   anyway, so the ceiling can go much higher without extra OOM exposure. The
   Phase 1 ladder values (12/8/4) are conservative floors — measure whether 16–32
   fits in safe mode on 16 GB.

4. **Enable chunked prefill.**
   Long reading-passage prefills (up to ~4k tokens) stall decode iterations for the
   whole running batch. `enable_chunked_prefill=True` in the engine kwargs
   (`src/llm.py`) interleaves prefill with decode; outputs are identical.

5. **Verify prefix caching actually engages.**
   `enable_prefix_caching=True` is configured, but some vLLM/GPU combinations
   silently disable it. Wave-2 extraction prompts share a long prefix with their SC
   generation prompts, so this materially affects Wave 2 cost. Add the
   engaged/disabled status to the Phase 0 startup log and confirm on the dev run.

### Phase 2 — True per-route token budgets (the organizer's core ask)

- In `src/wave_solver.py` (`run_wave1`, `run_wave2`): group generation calls by
  **(mode, route budget)** instead of the binary think/no-think split, so each group
  calls `batch_generate` with its own `max_tokens` from `TOKENS_BY_ROUTE`. Still a
  handful of batched vLLM calls — no per-prompt params needed.
- Set budgets from Phase 0 measurements (expected shape, confirm with data):
  | Route / mode | Current effective | Target |
  |---|---|---|
  | STEM think | 3072 | ~p95 of measured STEM reasoning length (likely 1024–1536) |
  | KNOWLEDGE think (8+ choice / ambiguous) | 3072 | ~1024 |
  | READING no-think | 512 | 512 (keep) |
  | KNOWLEDGE no-think | 512 | 256 |
  | SAFETY | n/a | forced answer, no generation |
- Truncation degrades gracefully: `_fit_extraction_prompt` plus constrained
  extraction still produce a valid letter from a truncated draft.
- Add `stop` sequences to reasoning generation (thread `stop` through
  `generate_freeform` → `llm.sampling_params(**extra)`, which already accepts
  extras): stop at the draft's conclusion marker (e.g. `"\nĐáp án:"`) so the model
  does not ramble to the token cap after concluding. Extraction re-derives the
  letter, so stopping at the conclusion is safe.
- Update both `tokens_by_route` blocks in `configs/pipeline_config.yaml`.

### Phase 3 — Cut wasted SC volume (small, controlled risk — validate before adopting)

- **Two-stage Wave 2** in `src/wave_solver.py`:
  - Wave 2a: run only n=2 SC samples per escalated question.
  - If both samples agree with the wave-1 answer → finalize (3-way agreement is
    stronger evidence than the dead margin proxy).
  - Only disagreeing questions proceed to Wave 2b with the remaining samples (up to
    the route's full sc_n).
  - Trace data predicts ~60–66% of SC sample volume disappears, concentrated in
    STEM think tokens.
  - Policy constant `SC_STAGE1_N = 2` in `src/sc_policy.py` / yaml;
    `escalation_reason` records `_early_consensus` vs `_full_sc` for auditability.
- **Stretch (only if time allows):** diagnose margin saturation in
  `src/batch_extract.py` / `src/extract.py` (why only one finite legal-label logprob
  returns from vLLM with `allowed_token_ids`). If cheaply fixable, use a
  high-precision margin threshold to skip Wave 2a entirely for very confident items.
  Not required — two-stage SC already captures the win using real votes.

### What we deliberately do NOT touch

Router rules, prompts, model choice, extraction mechanism, option-shuffle voting,
checkpoint/always-emit safety layer — all accuracy-bearing and organizer-praised.
No pipeline replacement, per the organizer's note.

## Files to modify

| File | Change |
|---|---|
| `src/wave_solver.py` | route-budget batch grouping, merged per-request-params generation, tokenization fast path, two-stage Wave 2, compute-count + time-share tracing |
| `src/llm.py` | token counts in `GenerationOutput`, `stop` passthrough, per-request `SamplingParams` list support, `enable_chunked_prefill` engine kwarg |
| `src/reasoning_agent.py` | thread `stop` / token counts / per-request params through `generate_freeform` |
| `src/v03_gamma.py` | OOM degrade-retry ladder, engine/GPU config + prefix-caching status logging, remove hardcoded `max_num_seqs=16` |
| `predict.py` | real per-question `time` in `submission_time.csv` from trace attribution |
| `configs/pipeline_config.yaml` + `src/config.py` | budget values, safe-mode ladder, stage-1 SC constants |
| `src/sc_policy.py` | two-stage SC constants/helpers |
| `tests/` | new unit tests: batch grouping picks correct budgets; two-stage consensus logic; time attribution sums to wave totals; OOM ladder retry |

## Verification

1. `python3.11 -m pytest tests/ -m "not slow"` after each phase.
2. Public-set run per phase (or batched: after Phase 1, then after Phases 2+3):
   - **Accuracy gate:** proxy vs `data/reference/reference_answers.csv` — baseline
     424/463 (91.58%). Accept Phases 1 and 1b only at zero answer diff (they must be
     byte-identical); accept Phases 2–3 if ≥ 423/463 (≤1 question drop).
   - **Speed:** s/question vs the 7.98 baseline. Rough expectations:
     - Phase 1 alone: ~1.5–2.5× faster wall-clock (concurrency 4 → 12)
     - Phase 2: think-token volume roughly halved
     - Phase 3: ~60% of SC samples removed
     - Combined target: **~2–3 s/question** on the dev GPU.
   - **Compute audit (new):** per-route token totals from the new trace fields —
     confirms no route exceeds its budget and shows exactly where remaining time
     goes. This is the artifact to show the organizers.
3. Smoke test: `python predict.py --input data/... --limit 20` to confirm
   `submission.csv` and the per-question `submission_time.csv` format.
