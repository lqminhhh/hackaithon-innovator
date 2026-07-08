# Speed Optimization Plan — VietMind MCQ (v03_gamma)

> Goal: fix the 0.375 speed score by cutting per-question token spend and wall-clock
> time, with little to no accuracy loss (85.96% public / 91.58% proxy must hold).
>
> Organizer feedback addressed:
>
> 1. "Adjust token usage for each route so not too much token is spent on one problem."
> 2. "Log route + actual compute per question on the judge machine to detect degraded
>    fallback and adjust route thresholds instead of replacing the pipeline."

## Context

The submitted `v03_gamma` runs at ~7.98 s/question on our 24 GB dev GPU and slower on
the 16 GB judge card in safe mode. For ~2000 private questions that is hours of
wall-clock time. The accuracy design (router + wave-batched SC) is sound; the problem
is that compute is allocated far more bluntly than the design intends.

## Judge hardware: RTX 5060 Ti (Blackwell), 32 GB system RAM

The judges' GPU is now known, which pins down several assumptions:

- **Architecture: Blackwell, SM 12.0.** Requires CUDA 12.8+ builds. Our image ships
  `nvidia/cuda:12.9.1` + `torch==2.11.0` + `vllm==0.23.0` (`Dockerfile:4`,
  `requirements.txt:7,19`), which support Blackwell — so a hard arch-incompatibility
  crash is unlikely. But this is exactly the failure class that would trigger the
  silent HF fallback (Diagnosis #6), so the Phase 0 backend/arch logging
  (`torch.cuda.get_arch_list()`, device name, a tiny CUDA op at startup) stays
  mandatory to prove vLLM actually ran on the judge machine.
- **VRAM: 16 GB dedicated (confirmed).** Exactly the target the safe-mode design
  assumes. If the judge machine runs a desktop session, ~1–2 GB is already taken by
  the OS/display — which is precisely why Phase 1 sizes VRAM **dynamically from
  free memory** (`torch.cuda.mem_get_info()`) instead of using a fixed fraction of
  total: the pipeline takes whatever is actually available minus a cushion, on any
  machine.
- **Memory bandwidth: 448 GB/s GDDR7 — roughly HALF of our 24 GB dev card**
  (3090/4090-class ≈ 930–1000 GB/s). Decode is bandwidth-bound, so expect judge
  s/question ≈ **~2× our dev numbers at identical settings**. This makes the
  token-volume cuts (Phases 2–3) and quantization (Phase 4) *more* valuable on the
  judge box than they look in dev measurements.
- **FP8 is now viable** (Blackwell ≥ SM89), widening Phase 4's options — see the
  updated Phase 4 note.
- **First-run kernel JIT:** a new architecture means Triton/vLLM compile caches are
  cold on the judge machine; the existing warmup pass matters and stays enabled.

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
6. **A silent catastrophic fallback exists: vLLM failure → sequential HuggingFace.**
   `_load_agent` (`v03_gamma.py:430-447`) wraps vLLM init in a try/except; on ANY
   failure (OOM, driver mismatch, CUDA version) it silently falls back to
   HuggingFace transformers — non-batched, sequential, with per-label scoring done
   one label at a time (`batch_extract.py:84-103`). That is easily 10–50× slower
   than the vLLM path. If this fired on the judge machine, it alone explains the
   speed score, and it matches the organizer's suspicion of "degraded fallback due
   to context/VRAM limits". Today nothing in the output reveals which backend ran.

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
  `torch.cuda`, prefix-caching status) and — most importantly — **which backend
  actually loaded (vLLM vs HuggingFace fallback) and why**, into both stdout and
  the trace. This is what identifies "degraded fallback" on the judge card
  (see Diagnosis #6).
- **Then run one instrumented public-set run** → per-route generated-token
  distributions (p50/p90/p95) that set the Phase 2 budgets.

### Phase 1 — Zero-accuracy-risk throughput (biggest wall-clock win)

- **Dynamic VRAM sizing — use everything actually free, minus a cushion.**
  vLLM's `gpu_memory_utilization` is a fraction of *total* VRAM, not *free* VRAM,
  so any fixed value both wastes headroom on a clean machine and risks OOM on a
  machine where the desktop already holds 1–2 GB. Instead, compute it at startup
  in `_load_agent`:
  1. `free, total = torch.cuda.mem_get_info()`
  2. `gpu_memory_utilization = (free − HEADROOM) / total`, with `HEADROOM = 1 GB`,
     clamped to `[0.50, 0.92]`
  3. Log free/total/chosen-util in the Phase 0 startup trace.
  The headroom cushion stays necessary for activation spikes during long prefills
  and for other processes grabbing VRAM after init (vLLM only measures once, at
  startup).
- **`max_num_seqs` needs no ladder — let the scheduler self-limit.** vLLM
  pre-allocates the KV cache inside its budget and only admits sequences while
  free KV blocks exist, so a generous `max_num_seqs=32` cannot itself cause OOM;
  actual concurrency adapts to whatever KV space the dynamic sizing produced.
- **OOM retry ladder becomes a headroom ladder (fallback only):** if engine init
  or a wave still OOMs, tear down and retry with `HEADROOM` = 2 GB, then 3 GB
  (i.e., progressively more conservative), instead of stepping through fixed util
  values. Implement in `src/v03_gamma.py` around engine construction and the wave
  loop — per-wave checkpointing already exists, so a mid-run retry resumes from
  checkpoint and never loses answers.
- **The ladder replaces the current single-shot vLLM try → silent HF fallback**
  (`v03_gamma.py:430-447`, Diagnosis #6). Today ONE vLLM failure sends the entire
  2000-question run to sequential HuggingFace. After this change, vLLM is retried
  down the whole ladder first; HuggingFace remains only as the very last resort and
  must announce itself loudly in stdout and the trace so a degraded judge run is
  diagnosable afterward.
- Config (headroom values, clamp range, `max_num_seqs=32`) lives in
  `configs/pipeline_config.yaml` (`safe_vllm` section repurposed).
- Keep `max_model_len=4096` unchanged.
- Answers are expected to be identical (temperature-0 wave 1, unchanged SC seeds) —
  verify with an answer diff against `data/submissions/submission_v03_gamma.csv`.
  Caveat: changing `max_num_seqs` changes batch composition, and GPU kernel
  reduction order can flip a rare borderline item even at temperature 0. A handful
  of diffs is acceptable if proxy accuracy does not drop (see Verification).

### Phase 1b — More zero-accuracy runtime fixes (outputs stay byte-identical)

These remove pure mechanical overhead. None of them changes any prompt, sampling
parameter, or answer.

1. **Merge serialized generation batches (idle GPU tails).**
   Wave 1 and Wave 2 each run the think batch and the no-think batch as *separate
   sequential* engine calls (`wave_solver.py:239-259`, `:360-382`). While the last
   long stragglers of one batch finish, the GPU idles instead of starting the next
   group. Fix: one merged `engine.generate` call per wave with **per-request
   `SamplingParams`** — a mechanism this codebase already uses successfully:
   `batch_extract.py:45-56` passes a params list to one `engine.generate` call.
   Think vs no-think is a chat template kwarg, not a sampling param, so pre-render
   each prompt through `tokenizer.apply_chat_template(..., enable_thinking=...)`
   individually, then submit the single merged batch. Same prompt + same params per
   item → identical outputs, but all groups share one continuous batch and
   stragglers overlap.
   **This is also the delivery mechanism for Phase 2:** each request carries its
   exact route-specific `max_tokens`, so no batch grouping is ever needed — one
   engine call per wave, per-question budgets, zero execution overhead.
2. **Kill serial CPU tokenization between waves.**
   `_fit_extraction_prompt` (`wave_solver.py:156-196`) calls `tokenizer.encode` on
   every full extraction prompt one-by-one in Python — once per question in Wave 1
   and once per SC sample in Wave 2 (~5000+ serial encodes on the private set)
   while the GPU waits. Two sound fixes, combined:
   - Fast path via **UTF-8 byte length** (not characters — a Vietnamese diacritic
     char can be 2–3 byte-level BPE tokens, so char count is NOT an upper bound;
     byte count is, since every token covers ≥1 byte): if
     `len(prompt.encode("utf-8")) + buffer < max_input_tokens`, it provably fits —
     skip encoding entirely.
   - For the remainder, use one **batched** `tokenizer(...)` call (HF fast
     tokenizers parallelize in Rust) instead of a Python loop.
   Byte-identical outputs either way.
3. **Remove the hardcoded non-safe `max_num_seqs=16`.**
   Non-safe mode hardcodes `max_num_seqs=16` (`v03_gamma.py:177`); vLLM's own
   default is 256 and it budgets KV-cache blocks under `gpu_memory_utilization`
   anyway, so the ceiling adds no OOM protection — it only caps throughput. With
   Phase 1's dynamic VRAM sizing + self-limiting scheduler, both modes converge on
   a generous `max_num_seqs=32` and let KV-block availability set real concurrency.
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

- In `src/wave_solver.py` (`run_wave1`, `run_wave2`): assign each question its own
  `max_tokens` from `TOKENS_BY_ROUTE` via the **per-request `SamplingParams`**
  mechanism from Phase 1b.1 — one merged engine call per wave, every request
  pre-configured with its exact route budget. No batch grouping needed at all.
- Set budgets from Phase 0 measurements (expected shape, confirm with data):

  | Route / mode                            | Current effective | Target                                                     |
  | --------------------------------------- | ----------------- | ---------------------------------------------------------- |
  | STEM think                              | 3072              | ~p95 of measured STEM reasoning length (likely 1024–1536) |
  | KNOWLEDGE think (8+ choice / ambiguous) | 3072              | ~1024                                                      |
  | READING no-think                        | 512               | 512 (keep)                                                 |
  | KNOWLEDGE no-think                      | 512               | 256                                                        |
  | SAFETY                                  | n/a               | forced answer, no generation                               |
- Truncation degrades gracefully: `_fit_extraction_prompt` plus constrained
  extraction still produce a valid letter from a truncated draft.
- **Stop sequences: deliberately NOT used.** The reasoning prompts
  (`solve.py:286-295`, `sc_policy.py:230-239`) end with "suy nghĩ ngắn gọn … trước
  khi chọn đáp án" and never ask the model to emit a fixed marker like
  `"Đáp án: X"`, so there is no reliable string to stop on — a stop string would
  cut drafts mid-conclusion unpredictably. Changing the prompts to add a marker is
  off-limits (prompts are accuracy-bearing). The per-route `max_tokens` budget is
  therefore the sole and sufficient length control.
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
- **Stretch (optional — the plan does NOT depend on it): fix margin saturation.**

  ⚠️ **Lesson from v03_delta:** real margins were already attempted there via
  **continuation scoring — one extra scoring request per legal answer label per
  question**. That multiplied request count and KV pressure, producing 27.5 s/q and
  Wave-2 OOM on 16 GB. **Do not repeat that approach.** Anything that adds extra
  requests per question to get confidence is out.

  This stretch item is mechanically different: **zero extra requests.** The same
  single 1-token extraction call that runs today already computes a full softmax;
  we only read more entries out of it. The extraction call currently requests
  `logprobs=min(len(token_map), 64)` (`batch_extract.py:47-53`) — with 4 labels
  that is top-4, and vLLM (notably the V1 engine) returns logprobs from the raw
  pre-mask distribution, so non-label tokens ("Answer", newline, …) swallow the
  top-4 and only the sampled label comes back finite → `softmax_margin` returns 1.0.

  Guardrails (informed by the delta failure):
  1. Request a fixed `logprobs=20` — **never above 20**, so the engine's default
     `max_logprobs` cap is never raised and logprob buffers do not grow.
  2. Validate on a small `--limit 50` run first: confirm ≥2 legal labels come back
     finite per question (log the finite-label count in the Phase 0 trace) and that
     runtime/VRAM are unchanged before any full run.
  3. If a label is still missing from top-k, treat the margin as **None/uncertain**
     instead of 1.0 (`extract.py:58-59`) so gates fail toward escalation, never
     toward false confidence. (Changes escalation behavior → gated like Phases 2–3.)
  4. If anything is unstable or the finite-label counts don't improve, **drop this
     item entirely** — Phase 3's two-stage SC already delivers the compute savings
     from observed votes, with no margin signal needed.

### Phase 4 — Weight quantization (gated — biggest single speed multiplier if accuracy holds)

Quantization changes numerics, so it is NOT accuracy-neutral — it goes through the
same accuracy gate as Phases 2–3. But the payoff is large enough to justify testing:

- **What:** AWQ INT4 weights for `Qwen3.5-4B`. FP16 weights are ~8 GB; INT4 ≈
  ~2.6–3 GB. On a 16 GB card that frees ~5 GB for KV cache, and 4B-scale decode is
  memory-bandwidth-bound, so expect roughly **1.5–2× faster decode** plus much
  higher safe concurrency — Phase 1's dynamic VRAM sizing reinvests the freed
  ~5 GB into KV cache automatically, no retuning needed.
- **AWQ vs FP8 on the known judge card:** the RTX 5060 Ti (Blackwell) supports
  both. **AWQ INT4 is still the better fit** because the 5060 Ti's bottleneck is
  its 448 GB/s memory bandwidth — INT4 weights cut weight traffic ~4× vs FP16
  (FP8 only ~2×). FP8 is the fallback if AWQ kernel quality on SM 12.0 in
  `vllm==0.23.0` turns out problematic; verify whichever is chosen inside the
  actual CUDA 12.9.1 container, not just on the dev box.
- **Existing hooks:** `src/models.py:105` already passes `quantization="awq"` to
  vLLM when "awq" appears in the model name, and `Dockerfile:41` already carries a
  comment marking where to swap in an AWQ repo — only the model id in
  `configs/pipeline_config.yaml` and the Dockerfile snapshot line need to change.
- **How:** prefer an official `Qwen3.5-4B-AWQ` checkpoint if published; otherwise
  quantize offline with AutoAWQ / llm-compressor using Vietnamese MCQ calibration
  data (e.g., the public-set questions). Bake the quantized weights into the Docker
  image — inference stays fully offline.
- **Compliance:** quantization does not change the parameter count — still one open
  LLM under 5B. Confirm with organizers if the rules text is ambiguous about
  quantized variants.
- **Accuracy gate:** full public-set run; accept only if proxy ≥ 423/463 (same bar
  as Phases 2–3). MCQ tasks typically lose 0–0.5 pt from AWQ INT4; if the drop is
  larger, ship without it — Phases 0–3 stand on their own.

### Phase 5 — Accuracy reinvestment (after speed phases land; each item gated)

Phase 3 frees ~60% of SC samples. Instead of pocketing all of it as speed, reinvest
part of that budget where the error data says it matters (v02_gamma error slice:
26 knowledge / 13 stem / 5 reading of 44 errors). All items are rule-compliant
(same single LLM, no RAG, no second model, offline) and exclude fine-tuning.
Ordered by expected value:

1. **Promote the margin fix from stretch to core (accuracy framing).**
   KNOWLEDGE is the biggest error bucket (18.2% error rate) precisely because the
   `margin < 0.20 → SC n=5` rescue gate has never fired (margins saturated at 1.0).
   With real margins (Phase 3 stretch, same guardrails — zero extra requests,
   `logprobs=20` cap), low-confidence knowledge questions finally get their rescue
   pass and STEM's adaptive n=7 activates for genuinely uncertain items.
2. **Reinvest early-consensus savings into deeper SC on disagreements.**
   In two-stage Wave 2, when the two probe samples DISAGREE with wave-1, escalate
   deeper than today (n=3 → 5–7 by route). Disagreement is the best uncertainty
   signal available; this targets compute exactly where votes are split, at
   roughly zero net cost vs the pre-plan baseline.
3. **Disable option shuffle for high-choice questions (≥8 labels).**
   SC broke 3 correct first answers (`test_0222`, `test_0227`, `test_0432`), and
   the project's own analysis flags ≥8-choice items as vulnerable to shuffle/remap
   confusion. Keep SC, skip the shuffle: `shuffle_options`
   (`sc_policy.py:184-205`) already returns an identity map when disabled — add an
   `n_choices >= 8` condition. Remap-confusion risk outweighs position-bias risk
   at 10 labels.
4. **Detect truncated think drafts and fall back to direct choice.**
   Protects Phase 2's budget cuts: when a think draft hits `max_tokens` without a
   closing `</think>`, extraction conditions on an incomplete draft. Detect
   truncation and use the plain direct guided-choice prompt for that question
   instead of the reasoning-conditioned extraction.
5. **Count the wave-1 answer as a full vote in `_vote`.**
   The temperature-0, unshuffled first pass is the most reliable single sample;
   today it is only a tie-breaker (`solve.py:200-242`). Counting it as a vote
   protects correct first answers from being outvoted by noisy SC samples — the
   exact "SC broke 3 correct answers" failure mode.
6. **Cheap A/B experiments (optional, each individually gated):**
   SC temperature 0.6 → ~0.8 (more diverse reasoning paths, which self-consistency
   theory prefers), and carefully validated route-specific few-shot exemplars —
   the ONE sanctioned exception to the "never touch prompts" rule, allowed only
   with its own full public-set gate, since it is the only lever that could move
   any of the 21 persistent failures that stem from format rather than knowledge.

Not worth pursuing: the 21 persistent failures as a group (wrong across all
versions v01→v03; almost certainly capability limits of a 4B model without
retrieval, and retrieval is banned).

Acceptance: each item lands separately and must improve or hold proxy accuracy
(≥ 424/463 to adopt an accuracy item — improvements are the goal, not breakeven)
without raising s/question above the post-Phase-3 level by more than ~10%.

### What we deliberately do NOT touch

Router rules, model choice, extraction mechanism, checkpoint/always-emit safety
layer — all accuracy-bearing and organizer-praised. No pipeline replacement, per
the organizer's note. Prompts stay untouched throughout the speed phases (0–4);
the only sanctioned prompt experiment is Phase 5 item 6, behind its own full
public-set gate. Option-shuffle voting stays, except the targeted ≥8-choice
exemption in Phase 5 item 3.

## Risk ratings per phase

| Phase | Risk | Why | Worst case |
|---|---|---|---|
| 0 (instrumentation) | Low | Additive logging only; no policy change | Slightly larger trace files |
| 1 (dynamic VRAM) | Medium-low | Sizing mistake is caught by the headroom retry ladder; floor = today's settings | No speedup, never a lost run |
| 1b (mechanical fixes) | Low-medium | Each is output-preserving *if implemented exactly*; chat-template pitfall below | Answer diffs caught by the zero-diff gate |
| 2 (route budgets) | Medium | Truncation can clip a draft that mattered; budgets set from measured p95, gated | ≤1 question drop or revert budgets |
| 3 (two-stage SC) | Medium | Changes escalation policy; evidence-based (66% wasted confirmations), gated | Revert to single-stage SC |
| 4 (AWQ) | Medium-high | Changes model numerics; biggest gated item | Ship without it; Phases 0–3 stand alone |
| 5 (accuracy reinvestment) | Medium | Each item changes answer policy; all individually gated | Drop any item that fails its gate |

Nothing in the plan is unrecoverable: every phase either preserves outputs exactly
or passes through the public-set accuracy gate before adoption, and the
checkpoint/always-emit layer is never touched.

## Implementation order, done criteria, and pitfalls (for the executing agent)

**Order:** Phase 0 → 1 → 1b → 2 → 3 → (4) → (5). Dependencies: Phase 2 requires
Phase 1b.1 (per-request params carry the budgets); Phase 2 budgets require
Phase 0's measured token distributions; Phase 3 is independent of Phase 2 but
should be measured after it so savings attribute cleanly. Phase 5 comes only
after the speed phases land (items 1–2 spend the budget Phase 3 frees; item 4
protects Phase 2's cuts) and its items land one at a time, each with its own
public-set gate.

**Done criteria per phase:**

- **Phase 0:** a public-set run produces a trace where every question has route,
  sc_n, think flag, gen-token counts, and time shares; per-question `time` values
  in `submission_time.csv` are non-uniform and sum ≈ total wall time; startup log
  shows backend, GPU name, free/total VRAM, chosen util, prefix-caching status.
- **Phase 1:** engine starts with dynamically computed util; forced-OOM test (set
  clamp max artificially high) triggers the headroom retry and completes from
  checkpoint; HF fallback only reachable after the full ladder and announces
  itself in stdout + trace.
- **Phase 1b:** public-set answers identical to the pre-1b run (see Verification
  for the numerics tolerance); wall-clock strictly lower; unit test proves the
  byte-length fast path never underestimates token counts on Vietnamese samples.
- **Phase 2:** trace shows no route exceeding its budget; proxy accuracy ≥ 423/463.
- **Phase 3:** trace shows `_early_consensus` vs `_full_sc` reasons; SC sample
  count drops ≥50%; proxy accuracy ≥ 423/463.
- **Phase 4:** model loads quantized inside the CUDA 12.9.1 container; proxy
  accuracy ≥ 423/463; s/question improves vs Phase 3 result.

**Known pitfalls:**

1. **Chat-template equivalence (Phase 1b.1 — the trap that silently breaks
   "byte-identical").** `llm.generate_text` currently calls `engine.chat(...)`,
   which applies the model's chat template internally. The merged path must
   pre-render with `tokenizer.apply_chat_template([{"role":"user",...}],
   tokenize=False, add_generation_prompt=True, enable_thinking=...)` and produce
   the EXACT same final prompt string. Add a startup assertion comparing both
   renderings on a sample prompt before trusting the merged path; fall back to
   the old two-call path if they differ.
2. **CUDA context before `mem_get_info` (Phase 1).** `torch.cuda.mem_get_info()`
   requires an initialized CUDA context on the right device — call
   `torch.cuda.init()` / select device first, or the reading can be wrong.
3. **Byte-length bound, not char-length (Phase 1b.2).** Vietnamese diacritics are
   multi-byte; only `len(text.encode("utf-8"))` upper-bounds byte-level BPE token
   count. Using `len(text)` reintroduces silent context overflows.
4. **Do not touch prompt text in any speed phase (0–4)** — prompts are
   accuracy-bearing (this is also why stop sequences were rejected). The single
   exception is the Phase 5 item 6 few-shot experiment, behind its own gate.
5. **`SC_SEED`-driven shuffles must not change in Phase 3.** Wave 2a must use
   `sample_idx` 0..1 and Wave 2b must continue 2..n-1 so shuffle sequences match
   the single-stage run for gate comparability.
6. **Confirm actual per-route budgets against `TOKENS_BY_ROUTE` keys** — the
   config uses uppercase route keys ("STEM") while `Wave1Result.route` is
   lowercase ("stem"); normalize once at the boundary or budgets silently miss.

## Files to modify

| File                                                 | Change                                                                                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/wave_solver.py`                               | merged per-request-params generation with per-route budgets, tokenization byte-length fast path + batched encode, two-stage Wave 2, compute-count + time-share tracing |
| `src/llm.py`                                       | token counts in `GenerationOutput`, per-request `SamplingParams` list support, `enable_chunked_prefill` engine kwarg |
| `src/reasoning_agent.py`                           | thread token counts / per-request params through `generate_freeform`                                                                |
| `src/v03_gamma.py`                                 | vLLM retry ladder replacing silent HF fallback, engine/GPU/backend + prefix-caching status logging, remove hardcoded `max_num_seqs=16`                                |
| `predict.py`                                       | real per-question`time` in `submission_time.csv` from trace attribution                                                                     |
| `configs/pipeline_config.yaml` + `src/config.py` | budget values, dynamic-VRAM headroom ladder + clamp range, stage-1 SC constants                                                                 |
| `src/sc_policy.py`                                 | two-stage SC constants/helpers; Phase 5: high-choice shuffle exemption, disagreement-depth policy                                               |
| `src/solve.py`                                     | Phase 5: `_vote` counts wave-1 answer as a full vote; truncated-draft detection helper                                                          |
| `src/models.py` + `configs/pipeline_config.yaml`   | Phase 4: quantized model id (AWQ hook at `models.py:105` already exists)                                                                        |
| `tests/`                                           | new unit tests: batch grouping picks correct budgets; two-stage consensus logic; time attribution sums to wave totals; OOM ladder retry         |

## Verification

1. `python3.11 -m pytest tests/ -m "not slow"` after each phase.
2. Public-set run per phase (or batched: after Phase 1, then after Phases 2+3):
   - **Accuracy gate:** proxy vs `data/reference/reference_answers.csv` — baseline
     424/463 (91.58%). Phases 1 and 1b change no policy, so expect ~0 answer diffs;
     accept them if proxy accuracy does not drop (≥ 424/463) — a couple of flips
     from batch-composition numerics are tolerable, but more than ~3 diffs means a
     bug, not noise. Accept Phases 2–4 (including quantization and the margin fix)
     if ≥ 423/463 (≤1 question drop).
   - **Speed:** s/question vs the 7.98 baseline. Rough expectations:
     - Phase 1 alone: ~1.5–2.5× faster wall-clock (concurrency 4 → 12)
     - Phase 2: think-token volume roughly halved
     - Phase 3: ~60% of SC samples removed
     - Combined target: **~2–3 s/question** on the dev GPU.
     - **Judge-box translation:** the 5060 Ti has ~half the dev card's memory
       bandwidth, so expect roughly 2× the dev s/question at identical settings —
       i.e., a 2–3 s/q dev result ≈ 4–6 s/q on the judge machine before Phase 4;
       AWQ (Phase 4) claws most of that factor back by shrinking weight traffic.
     - Separately: if the judge run had silently degraded to the HuggingFace
       backend (Diagnosis #6), the retry ladder + backend logging fixes a 10–50×
       slowdown on its own — the Phase 0 startup log is what will prove or rule
       this out on judge-like hardware.
   - **Compute audit (new):** per-route token totals from the new trace fields —
     confirms no route exceeds its budget and shows exactly where remaining time
     goes. This is the artifact to show the organizers.
3. Smoke test: `python predict.py --input data/... --limit 20` to confirm
   `submission.csv` and the per-question `submission_time.csv` format.
