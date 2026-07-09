# Post-submission changes

This note summarizes what changed after the last final submission candidate,
why we made those changes, and how they respond to BTC feedback.

## What this document is for

The project already had a clear final branch choice: `v03_gamma` at **85.96%**
on the public leaderboard. After that choice, we continued a limited branch to
improve deployment safety, judge diagnosability, and runtime hygiene without
changing the core one-model route-aware architecture.

This document is not a replacement for the report. It is a short engineering
memo for the branch work that happened after the main final path had already
been chosen.

## What BTC feedback we were reacting to

Two themes from BTC feedback mattered most:

1. **Token usage should be more controlled.**The system should not spend unnecessary compute on one problem without
   understanding where that compute goes.
2. **Judge-machine behavior must be diagnosable.**
   If a run becomes slower, falls back to another backend, or behaves
   differently on judge hardware, we need traces and runtime facts rather than
   guesswork.

Those comments pushed us toward instrumentation and deployment hardening first,
before any more aggressive accuracy experiments.

## What stayed unchanged

- Final architecture remained `v03_gamma`
- One open model only: `Qwen/Qwen3.5-4B`
- Offline inference only
- No second LLM
- No RAG
- No embedding model or reranker in final inference
- Official public leaderboard result stayed **85.96%**

In other words, this branch is not a new architecture. It is a more mature
operating version of the same final path.

## What changed

### 1. Phase 0 instrumentation

We added route-level and per-question runtime observability:

- generated-token counts for Wave 1 and Wave 2
- per-sample token counts for self-consistency
- attributed per-question runtime
- runtime backend metadata in traces
- startup runtime logging for GPU, CUDA, vLLM, and backend path

Why:

- to answer BTC's request for actual compute visibility
- to see where time really goes before trimming anything
- to detect degraded fallback on judge hardware

### 2. Phase 1 deployment hardening

We changed the runtime behavior so the pipeline sizes itself from **actual free
VRAM**, not only from fixed static assumptions:

- dynamic `gpu_memory_utilization` from `torch.cuda.mem_get_info()`
- retry ladder with increasing headroom before giving up
- wave-level OOM-like retry behavior
- chunked wave fallback before giving up on the main vLLM path
- louder backend/runtime reporting

Why:

- the final environment is 16 GB VRAM and hardware details are not fully under
  our control
- reliability on ~2000 questions matters more than a fragile best-case run on
  the public set

### 3. Phase 1b.4 chunked prefill

We enabled chunked prefill in the vLLM path.

Why:

- long reading/context prefills were one of the obvious runtime pain points
- this is a mechanical runtime improvement, not an answer-policy change

### 4. One conservative Phase 2 trim

Using the new trace data, we kept one low-risk token-budget intervention:

- READING Wave 2 think-mode SC now uses a smaller dedicated cap

Why:

- the trace showed that this branch was a safer trim target than STEM think or
  KNOWLEDGE think
- it was the one Phase 2-style change that looked low-risk enough to keep

What we deliberately did **not** do:

- we did not continue broad token-budget cuts across all route/stage pairs
- we did not keep pushing performance tuning after the user asked to stop
  changing performance behavior

So Phase 2 was closed conservatively after one measured change.

## What we learned from the new measurements

The new branch confirmed several things:

- the route structure remained healthy
- the system stayed on the intended vLLM path in the main branch tests
- the biggest compute sink remained think-mode STEM and other self-consistency
  heavy paths
- BTC's request for compute visibility was justified, because the original
  branch gave too little evidence about where tokens and time were spent

This made the branch easier to defend technically, even when we chose not to
continue aggressive optimization.

## Performance checkpoints in this branch

### Main branch checkpoints

Against the proxy reference `data/reference/reference_answers.csv`, the latest
intermediate high-water checkpoint reached:

- **426 / 463 = 92.01%**

After the later reliability hardening pass, one final sanity run was:

- **424 / 463 = 91.58%**

This is effectively back at the historical saved `submission_v03_gamma.csv`
proxy level. Compared with that saved run, the final sanity run was balanced:

- **8 improvements**
- **8 regressions**

The newest live local rerun now stored in `output/submission.csv` is:

- **423 / 463 = 91.36%**

Compared with the saved `submission_v03_gamma.csv`, that newest live run had:

- **13 answer diffs**
- **5 improvements**
- **6 regressions**
- **2 still wrong in both runs**

Its trace kept the same route/path structure as the main gamma branch:

- routes: `reading=100`, `stem=201`, `knowledge=155`, `safety=7`
- paths: `wave_reading_sc=42`, `wave_stem_sc=201`, `wave_direct=135`,
  `wave_knowledge_sc=78`, `forced_safety=7`

For reference, the older saved `submission_v03_gamma.csv` proxy was:

- **424 / 463 = 91.58%**

Important note:

- this is a **proxy-reference checkpoint**, not a new public leaderboard score
- the official public score remains **85.96%** unless re-submitted

### Quantized side experiments

We also tested quantized side paths, mainly to see whether a lighter model load
could improve deployment practicality.

Result:

- `cyankiwi/Qwen3.5-4B-AWQ-4bit` reached **409 / 463 = 88.34%** on the proxy
  reference

Conclusion:

- the quantized side experiment was not good enough to replace the main branch
- it may be faster, but the accuracy drop is too large for our current goal

FP8 was also explored, but hardware-path caveats on non-native FP8 GPUs made it
an exploratory path rather than a promotion candidate.

## Why these changes make sense as a response to BTC comments

BTC's comments were not asking for novelty for novelty's sake. They pointed to
two practical concerns:

- spend compute where it matters
- know what the system is actually doing on the judge machine

Our branch changes address that directly:

- instrumentation gave route-level and per-question evidence
- dynamic VRAM sizing and retry behavior reduced deployment fragility
- chunked wave fallback made late-wave OOM-like failures less likely to force
  immediate degradation
- chunked prefill improved runtime mechanics for long contexts
- conservative token trimming used measured data instead of intuition

This is also why we did **not** overreact by changing the architecture again.
The branch work is mostly about making the chosen architecture more explainable,
more stable, and more operationally credible.

## Final branch conclusion

After all post-submission branch work, our conclusion stayed the same:

- the preferred branch is still **`v03_gamma`**
- the branch is now better instrumented and more judge-diagnosable
- deployment safety is better than before
- the core architecture did not need to be replaced
- quantized side paths were not strong enough to promote

So the branch work strengthened the final submission story rather than changing
the final branch choice.
