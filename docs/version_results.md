# Version Results

> Scores are from the **HackAIthon Bảng C leaderboard** (public test set).
> Model: `Qwen/Qwen3.5-4B` on vLLM throughout.

| Version | Goal | Architecture | Key changes | Leaderboard | Inference time | s/question |
| --- | --- | --- | --- | --- | --- | --- |
| `v01_baseline` | Establish the simplest same-model baseline. | Single LLM free-form answer parsing; no routing, no guided-choice, no SC. | Updated baseline to use the same configured primary model as later versions. | 28.73% | 1761.9 s inference loop (2144.3 s total) | 3.81 s/q |
| `v02_alpha` | Test whether routing + constrained answer extraction fixes free-form parsing failures. | Rule router + two-pass guided-choice decoding. | Added READING / STEM / KNOWLEDGE / SAFETY routing and constrained final letter extraction via valid answer labels. No self-consistency. | 60.48% | 43.7 s inference loop (139.7 s total) | 0.09 s/q |
| `v02_beta` | Improve correctness with S4 escalation. | Rule router + two-pass guided-choice + per-question self-consistency. | Added STEM SC, low-margin KNOWLEDGE SC, and reason/purpose READING SC. Per-question loop; no wave batching. | 80.13% | 18411.7 s inference loop (18508.6 s total) | 39.77 s/q |
| `v02_gamma` | Keep beta-style accuracy gains while improving throughput and robustness. | Wave-batched router/guided-choice/SC pipeline. | Batches all first passes and all escalations; adaptive STEM SC depth; option shuffle de-bias; per-wave checkpointing. | **85.31%** | not separately recorded; 6424.4 s total | 12.77 s/q |
| `v03_alpha` | Harden router for 2000-question private set generalization. | Same wave pipeline as v02_gamma; router changes only. | Removed `n_choices >= 8 → STEM` rule; tightened harmful detection to actionable intent phrases; added STEM keywords. Routes: stem=201, knowledge=155, reading=100, safety=7. | 84.23% | 1790.3 s inference loop (2801.7 s total) | 3.87 s/q |
| `v03_gamma` | Keep the v3 router gains while recovering accuracy through compute policy. | Hardened router + targeted KNOWLEDGE/READING escalation + length-safe Wave 2 extraction. | Added high-choice KNOWLEDGE recovery without rerouting to STEM, broader READING detail SC, broader KNOWLEDGE ambiguity SC, and compacted Wave 2 extraction for long-context safety. | **85.96%** | not separately recorded | - |
| `v03_delta` | Convert the adaptive policy from “designed” to actually working. | Hardened router + targeted escalation + real continuation-scored margins + duplicate-safe SC handling. | Fixed wave margin extraction, repaired duplicate-option shuffle/remap, added duplicate/combination guidance in SC prompts, and made low-evidence margins conservative. Routes: safety=7, reading=100, stem=201, knowledge=155. Paths: wave_reading_sc=42, wave_stem_sc=201, wave_direct=134, wave_knowledge_sc=79, forced_safety=7. | **87.04%** | 12748.0 s inference loop (13066.0 s total) | 27.53 s/q |

## Key Notes

- Final-compliant inference uses one model only: `Qwen/Qwen3.5-4B`.
- `configs/pipeline_config.yaml` now marks final inference as offline, no external APIs, no embedding model, no reranker model, and no RAG.
- `src/config.py` holds shared Python constants such as `LLM_MODEL`, `GPU_MEM_UTIL`, `FALLBACK`, token budgets, and legacy S4 defaults.
- `v02_gamma` uses the refactored wave pipeline: `src/wave_solver.py`, `src/batch_extract.py`, and `src/sc_policy.py`.
- S5 semantic routing and RAG are not part of final-compliant runners because they require extra embedding/reranker models.
- Runtime numbers are from local/Colab runs and can vary by GPU, vLLM version, warmup, and `safe-mode` settings.
- `v02_gamma_router_v2` runtime was measured on a 24 GB VRAM card; judge hardware is 16 GB.
- `v03_delta` is the current best public-set score, but it makes extraction much heavier because each legal answer choice is scored as a separate continuation. Judge-safe deployment likely needs continuation microbatching.

## v03_delta Margin Recovery

`v03_delta` scored **+1.08 pts** above `v03_gamma` (87.04% vs 85.96%) and
**+2.81 pts** above `v03_alpha` (87.04% vs 84.23%).

**What changed:** the router stayed the same, but the adaptive confidence path
finally became real instead of decorative.
- Wave guided-choice extraction now scores each legal answer label directly via
  continuation logprobs, so margins are no longer stuck at `1.0`.
- Exact duplicate options are handled safely during SC option shuffle/remap, so
  votes no longer drift onto the wrong original label.
- SC prompts now explain how to treat duplicate options, near-duplicate wording,
  and combination answers like “cả A, B, C”.
- When too few legal label scores are visible, margins fall back to `0.0`
  instead of manufacturing false confidence.

**Why this matters:** `v03_gamma` had the right policy ideas, but the broken
margin path meant low-margin KNOWLEDGE rescue and adaptive STEM depth were not
actually being driven by trustworthy evidence. `v03_delta` converts those ideas
into working behavior and posts the best public-set score so far.

**Tradeoff:** throughput got worse, not better. Because extraction now expands a
question into one request per answer choice, large wave batches can OOM unless
continuation scoring is microbatched. That is now the main engineering risk for
judge deployment.

## v03_gamma Recovery

`v03_gamma` scored **+1.73 pts** above `v03_alpha` (85.96% vs 84.23%) and
**+0.65 pts** above `v02_gamma` (85.96% vs 85.31%).

**What changed:** the router stayed semantically strict, but compute policy got
smarter:
- 8+ choice KNOWLEDGE questions kept the `knowledge` route and received extra
  think-mode / SC treatment instead of being mislabeled as STEM.
- READING escalation expanded from only reason/purpose questions to
  detail-lookup questions that need exact evidence selection.
- KNOWLEDGE escalation expanded to ambiguous and combination-style options.
- Wave 2 extraction became length-safe, so long reading SC prompts no longer
  overflow the 4096-token context limit.

**Why this matters:** `v03_gamma` recovers the public-set accuracy that old
route hacks used to provide, but does it through compute allocation rather than
sloppier route labels. That is the more judge-safe path for the 2000-question
private set.

**Current best submission: `v03_delta` (87.04%).**

## Router v2 Regression Analysis

`v03_alpha` scored **-1.08 pts** below `v02_gamma` (84.23% vs 85.31%).

**Root cause:** The `n_choices >= 8 → STEM` rule removed 13 knowledge questions
from STEM routing. In v02_gamma those 13 items received think-mode reasoning +
always-on SC. In v02_gamma_router_v2 they were reclassified as KNOWLEDGE, but
because **margin computation is broken (all margins = 1.0)**, KNOWLEDGE SC never
fires. They fell back to a cheap no-think direct pass — worse than STEM treatment.

**Conclusion:** The router fix is correct in principle but premature. The right
sequence is: fix margin computation first → KNOWLEDGE SC activates → *then*
reclassify those 13 items. Until margins are fixed, the `n_choices >= 8` rule
was accidentally beneficial by forcing better compute on ambiguous items.

**`v03_alpha` by itself was not the final answer.** The cleaner router needed
targeted compute recovery, which is what `v03_gamma` now provides.
