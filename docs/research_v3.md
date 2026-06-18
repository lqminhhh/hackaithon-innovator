# Research & Evidence Map v3 — Vietnamese MCQ QA Agent

> Companion to `planning_v3.md` (build spec) and `note_v3.md` (rationale).
> This is the **traceability doc**: every design decision in the agent → the evidence that justifies it →
> the exact place it lives in the build. It is also our **creativity-score backbone** ("measured design,
> not fancy diagrams"): each lever was kept or killed by evidence, and this is the ledger.

## Evidence tiers (strongest → weakest)
- **[MEASURED]** — our own runs on the public-463. Strongest; load-bearing.
- **[FINDINGS]** — an external competitor benchmark/writeup. Treated as findings, not gospel.
- **[RESEARCH]** — published literature. Supporting rationale only (confirm exact citations before any public writeup).
- **[PENDING]** — believed beneficial but **not yet validated on our dev set**. Do not claim as proven.

---

## 1. ADOPTED — research/suggestions that are IN the agent, and where

| # | Lever | Evidence | Where in the agent |
|---|---|---|---|
| 1 | **Adaptive compute routing** (router decides effort; escalation catches misroutes) | [MEASURED] v2 route split 100/216/143/4; misroutes recovered by ladder · [RESEARCH] test-time-compute scaling, early-exit inference | `planning_v3` S3 + S4 per-route policy; INVARIANT 6 |
| 2 | **Self-consistency voting on STEM** (sample → majority) | [MEASURED] beta 60.48 (no SC) → gamma **79.91** (all-STEM SC); delta **−2.37** when SC skipped · [RESEARCH] Wang et al. 2022, *Self-Consistency Improves CoT* | S4 STEM policy (always SC); config `SC_N_STEM={high:3,low:7}` |
| 3 | **Two-pass guided choice** (reason freely → then constrain the letter) | [MEASURED] our Issue-1 (guided_choice suppresses thinking) · [RESEARCH] constrained/guided decoding; CoT-then-extract | S2; INVARIANT 5 |
| 4 | **Logprob-margin confidence gate** (escalate when unsure) | [MEASURED] gamma `low_margin_self_consistency=16` fired usefully · [RESEARCH] token-probability uncertainty / calibration | S2 (`margin`), S4 (`MARGIN_LOW`) |
| 5 | **Per-route margin thresholds** (margin distributions differ by route) | [MEASURED] our Issue-6 | config `MARGIN_LOW={READING:.10,STEM:.15,KNOWLEDGE:.20,SAFETY:.05}` |
| 6 | **Option-order shuffle / permutation de-bias** (10–11 options amplify slot bias) | [RESEARCH] Zheng et al. ICLR 2024, *LLMs Are Not Robust Multiple-Choice Selectors* (PriDe) · [MEASURED] Issue-6 · **[PENDING]** dev validation | S4; config `SHUFFLE_OPTIONS_IN_SC=True` |
| 7 | **Wave-batching** (batch all first passes, then all escalations) | [MEASURED] delta **2.45 s/q = 4.4×** gamma; Issue-2 · [RESEARCH] continuous batching (vLLM/Orca) | S4 wave structure; S7 runner |
| 8 | **Prefix caching** (shared system prompt + makes pass-2 nearly free) | [RESEARCH] vLLM prefix/KV cache · [MEASURED] Issue-1 note | S1 `enable_prefix_caching`; S2 |
| 9 | **Refusal-trap handling** (refuse only when truly harmful) | [MEASURED] our items: `test_0041` trap vs `test_0024/0294` harmful | S3 prompt line; S4 SAFETY |
| 10 | **Reading reason/purpose SC exception** | [MEASURED] `test_0005` fixed (distractor A → correct C via SC) | S4 READING policy (`n=3` on "lý do/tại sao/vì sao/mục đích") |
| 11 | **Never-crash + checkpoint/resume + deterministic SC seed** | [MEASURED] gamma_rag OOM crash risk · [RESEARCH] reproducibility | S7; INVARIANT 1; config `SC_SEED=1234` |
| 12 | **One strong model, no second model** | [MEASURED] note_v2 §4 (two small models share mistakes) · [FINDINGS] one-model benchmark hit 83.59 solo · [RULES] required | INVARIANT 3; model choice |
| 13 | **Single ≤5B reasoning model in think-mode for STEM** | [FINDINGS] competitor's weak cluster was math on a *non-thinking* model · [RESEARCH] reasoning checkpoints stronger on math | S1 (think/no_think), S4 STEM `mode=think` |

---

## 2. REJECTED — killed by a finding or a measurement (this list IS the creativity story)

| # | Lever | What killed it | Where recorded |
|---|---|---|---|
| 1 | **RAG / retrieval** | [MEASURED] gamma_rag **78.83 < 79.91** + reranker OOM · [FINDINGS] retrieval poisons small models · [RULES] illegal | planning_v3 "REMOVED S6"; note §2, §9 |
| 2 | **Reranker + embedding semantic router** | [MEASURED] OOM on shared card · [RULES] embedding/rerank banned | "REMOVED S5" |
| 3 | **Tool/code-execution reasoning (TIR)** | [FINDINGS] **−16.5** (4B writes buggy code, breaks correct items) | note §9; planning "Deliberately NOT doing" |
| 4 | **Naive fine-tuning** | [FINDINGS] FT **−4.44** / ±0.00 · [RESEARCH] catastrophic forgetting hits reasoning first | note §9 (last-resort, no-think path only) |
| 5 | **Letter-only / no-reasoning answering** | [FINDINGS] "letter-only → low" · [MEASURED] beta 60 ≪ gamma 80 (reasoning+SC is the lift) | INVARIANT 7; S2 two-pass |
| 6 | **STEM early-exit** (skip SC when confident) | [MEASURED] delta **−2.37** (38 STEM skipped) | config `STEM_DIRECT_MARGIN=1.01` (disabled) |
| 7 | **Trading accuracy for speed in general** | [MEASURED] delta receipt; [RESEARCH-of-the-rubric] 1 acc pt ≈ 0.8 final; speed axis = 10 total | INVARIANT 7 |

---

## 3. CONSIDERING / OPEN — research-backed, not yet validated on our dev set

| # | Lever | Evidence | Plan |
|---|---|---|---|
| 1 | **Adaptive SC depth (n=3/7)** vs flat n=5 | [PENDING] the `v02_epsilon` plan | Validate on S8 dev set; keep if it holds gamma accuracy at lower cost |
| 2 | **Quantization Pareto (F16 vs AWQ-4bit)** | [RESEARCH] AWQ near-lossless; faster decode on memory-bound 16GB · [PENDING] | Measure both on dev; ship the total-score winner (planning Open-items) |
| 3 | **STEM-specific accuracy levers** — self-verification pass, numeric-answer recompute, vote-aggregation tweaks | [RESEARCH] self-verification / SPOC-style single-pass check · [PENDING] | The main path past gamma → 83.59; prototype on the STEM bucket first |
| 4 | **STEM token-cap tuning (3072 → 2048/1536)** | [PENDING] speed lever; two-pass handles truncation | Sweep on dev; keep the fastest cap that doesn't drop STEM accuracy |
| 5 | **STaR / rejection-sampling self-distillation** (own model only → legal) | [RESEARCH] Zelikman et al. 2022, *STaR* · [FINDINGS] warn naive FT fails | Last resort if STEM still leaks and we're ahead of schedule; no-think path only |

---

## 4. Evidence ledger (the raw numbers + sources, in one place)

**[MEASURED] — our public-463 runs**
```
v02_alpha      54.43
v02_beta       60.48        (S0–S3 + margin, pre-SC)
v02_gamma      79.91   10.8 s/q   ← best accuracy; ALL 216 STEM run SC
v02_gamma_rag  78.83   18.0 s/q   ← RAG -1.08 + reranker OOM  → drop RAG
v02_delta      77.54    2.45 s/q  ← wave-batching 4.4×; STEM early-exit -2.37
Route split: reading 100 · stem 216 · knowledge 143 · safety 4
```

**[FINDINGS] — external competitor writeup (treated as findings, not verified by us)**
```
public-463 benchmark to beat:            83.59
fine-tune attempts:                      -4.44 / ±0.00
RAG:                                     noise + civics -5 ; illegal
tool/code reasoning (TIR):               -16.5
cluster acc:    math 73.9 · civics 78.7 · science 85.4
disqualified ~26B-total MoE:             88.55  (proves headroom exists, but >5B → illegal)
```

**[RESEARCH] — supporting literature (confirm exact citations before public submission)**
- Self-consistency decoding — Wang et al., 2022. → lever 1.2
- MCQ position/label bias & permutation debiasing (PriDe) — Zheng et al., ICLR 2024. → lever 1.6
- Constrained / guided decoding — guided generation literature. → lever 1.3
- Continuous batching for LLM serving — vLLM / Orca. → lever 1.7
- Self-Taught Reasoner (STaR) — Zelikman et al., 2022. → considering 3.5
- Single-pass self-verification / self-correction (SPOC-style) — recent self-correction literature. → considering 3.3

---

## 5. Honesty section — what is NOT yet backed by our own evidence

These are believed-good but **unproven on our data**; treat claims accordingly until S8 validates them:
- Option-shuffle de-bias gain (1.6) — research-backed, not yet measured by us.
- AWQ-4bit being "near-lossless" *for this task* (3.2) — measure, don't assume.
- That a thinking checkpoint beats non-thinking *for our exact STEM mix at our token budget* (1.13) — bake-off it.
- The ~83–84 ceiling estimate — a back-of-envelope from the bucket math, not a proof.

**Rule:** anything in §3 or flagged [PENDING] must pass the S8 tune/holdout split (and survive on an external VN set) before it ships or goes in the public writeup.
