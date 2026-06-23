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
| `v03_gamma` | Keep the v3 router gains while recovering accuracy through compute policy. | Hardened router + targeted KNOWLEDGE/READING escalation + length-safe Wave 2 extraction. | Added high-choice KNOWLEDGE recovery without rerouting to STEM, broader READING detail SC, broader KNOWLEDGE ambiguity SC, and compacted Wave 2 extraction for long-context safety. | **85.96%** | not separately recorded | 7.98 s/q |
| `v03_delta` | Validate real continuation-scored margins after gamma. | Hardened router + targeted escalation + real continuation-scored margins + duplicate-safe SC handling. | Fixed wave margin extraction, repaired duplicate-option shuffle/remap, added duplicate/combination guidance in SC prompts, and made low-evidence margins conservative. | **87.04%** | 12748.0 s inference loop (13066.0 s total) | 27.53 s/q |
| `v03_epsilon` | Try to keep delta logic while making deployment safer. | Delta-compatible runner with continuation microbatching defaults. | Added continuation microbatching and 16 GB deployment-oriented defaults, but the branch still encountered OOM in late Wave 2 on judge-like hardware. | not promoted | not promoted | not promoted |

## Key Notes

- Final-compliant inference uses one model only: `Qwen/Qwen3.5-4B`.
- `configs/pipeline_config.yaml` now marks final inference as offline, no external APIs, no embedding model, no reranker model, and no RAG.
- `src/config.py` holds shared Python constants such as `LLM_MODEL`, `GPU_MEM_UTIL`, `FALLBACK`, token budgets, and legacy S4 defaults.
- The final `v03_gamma` runner uses the refactored wave pipeline: `src/wave_solver.py`, `src/batch_extract.py`, and `src/sc_policy.py`.
- S5 semantic routing and RAG are not part of final-compliant runners because they require extra embedding/reranker models.
- Runtime numbers are from local/Colab runs and can vary by GPU, vLLM version, warmup, and `safe-mode` settings.
- `v02_gamma_router_v2` runtime was measured on a 24 GB VRAM card; judge hardware is 16 GB.
- Final branch choice is **`v03_gamma`**, even though `v03_delta` scored higher on the public set, because delta-like real-margin extraction was about 4x slower and remained OOM-prone on 16 GB judge-like runs.

## Final Branch Choice

We are choosing **`v03_gamma`** as the final submission branch.

Why not `v03_delta` / `v03_epsilon`, despite the higher public-set score?
- **Runtime cost:** `v03_delta` took about **27.53 s/question** versus about
  **7.98 s/question** for `v03_gamma`, roughly a 3.5-4x slowdown.
- **Deployment risk:** the delta-compatible path expanded extraction into many
  per-label continuation-score requests and still produced OOM failures on
  16 GB hardware, especially deep into Wave 2.
- **Private-set scale:** the final judge run is ~2000 questions, so reliability
  and completion probability matter more than a public-set gain that comes with
  real memory risk.

Important framing:
- `v03_gamma` was **not wrong in design**. It is better understood as a
  **fast-pass / efficiency-first approximation** of the later exact-margin
  system.
- The later delta experiment validated that real per-label margins can improve
  accuracy, but also showed that the exact method is too heavy for the final
  hardware target.
- So `v03_gamma` is the final operating point because it gives the best
  speed/reliability tradeoff while preserving the core route-aware architecture.

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

**Chosen final submission branch: `v03_gamma` (85.96%).**

Nuance on margins:
- `v03_gamma` still spends more compute on STEM through route-based
  self-consistency.
- But its confidence signal is a lightweight proxy, not the later exact
  continuation-scored per-label margin from `v03_delta`.
- So gamma should be described as a route-aware fast pass, not as a fully
  calibrated adaptive-margin system.

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
