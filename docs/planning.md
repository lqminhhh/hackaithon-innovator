# HackAIthon 2026 — Bảng C (Innovator) Agent: Full Engineering Plan

**Codename:** Entropy-Gated Jury (EGJ)
**Target:** ≥80 on accuracy + speed alone; ≥90 total with the creativity component
**Status:** Design locked — ready for implementation
**Last updated:** June 2026

---

## 1. Competition context

### 1.1 Scoring

| Component | Weight | How we win it |
|---|---|---|
| Public/private test accuracy | 80% | Adaptive two-tier inference, code execution, ensemble jury |
| Speed | 10% | One massively batched vLLM pass; adaptive compute only where needed |
| Creativity / innovation | 10% | Entropy-gated jury architecture, self-explaining audit log, ablation charts |

### 1.2 Hard constraints

- **Total network isolation:** container runs with `--network none`. All weights, tokenizers, packages, and indexes must be baked into the Docker image.
- **Model whitelist:** LLM must be from the **Qwen3.5** or **Gemma 4** series, **≤ 9B parameters**. Embedding/rerank: **BGE-M3** or **Qwen-Rerank** only.
- **Hardware caps:** fixed VRAM budget at runtime (verify exact GPU spec with organizers; design assumes 16–24 GB).

### 1.3 Dataset facts

- **463 questions**, Vietnamese, multiple choice, fields `qid` / `question` / `choices`.
- **Choice counts:** 4 (318), 10 (134), 3 (6), 2 (3), 5 (1), 11 (1). **29% of questions have 10+ choices** — all tooling must handle variable choice counts.
- **Question types:**
  1. **Context-grounded reading comprehension** — passage embedded in the question (`Đoạn thông tin:` … `Câu hỏi:` …). Answer from the passage; do NOT retrieve.
  2. **Quantitative reasoning** — math, physics, chemistry, economics (elasticity, GDP deflator, related rates, first-order decay, Hess's law, probability, Ohm's law). Requires real computation.
  3. **Standalone knowledge** — any field: Vietnamese law/politics, history, geography, civics, science, etiquette, general knowledge. The scope is unbounded ("every field").
  4. **Safety/refusal items** — questions asking how to violate laws or restrictions; one choice is a refusal (e.g. "Tôi không thể cung cấp thông tin…"). The refusal is the correct answer.

### 1.4 Lessons inherited from the 82.29% baseline post-mortem

- **A — Narrow beats wide:** indexing general Wikipedia cost −5.4 pp. RAG corpus must be small and surgical.
- **B — Models can't compute:** quantitative questions must be offloaded to a Python runtime.
- **C — No rigid routers:** keyword/regex routing fails silently on hybrid questions. Route by measured confidence, not rules.
- **D — Control output at the token layer:** mask the answer-token vocabulary; never regex-parse free text.

---

## 2. Model lineup (final)

| Role | Model | Quantization | VRAM est. | Rationale |
|---|---|---|---|---|
| Primary solver | **Qwen3.5-9B** | AWQ/GPTQ 4-bit (fallback FP8/BF16 if quant unsupported) | ~5–6 GB | Strongest legal model by a wide margin (GPQA Diamond 81.7; ~2× next-best sub-10B on Intelligence Index); strong Vietnamese; thinking mode with controllable budget |
| Second juror | **Gemma 4 E4B** (8B total / 4.5B effective) | 4-bit | ~3–4 GB | Different model family → decorrelated errors; agreement/disagreement is a strong difficulty signal |
| Fallback juror | Qwen3.5-4B | 4-bit | ~2.5 GB | Substituted if Gemma 4 proves flaky on the pinned vLLM commit (decide by Day 2) |
| Embedding | **BGE-M3** | FP16 | ~1.5 GB | Whitelisted; proven Vietnamese retrieval |
| Reranker | **Qwen-Rerank** | FP16 | ~1 GB | OPTIONAL — ships only if ablation shows net-positive accuracy per unit latency |

**Total resident footprint:** ~10–13 GB including KV cache → both LLMs stay loaded simultaneously; zero model-swap latency.

**Known landmine:** Qwen3.5 requires vLLM from the **main branch** (hybrid Gated DeltaNet attention). Pin the exact commit in the Dockerfile and verify on Day 1 that (a) both models co-load, (b) constrained decoding / `allowed_token_ids` works, (c) per-request logprobs are retrievable, (d) prefix caching is stable.

---

## 3. Architecture: Entropy-Gated Jury

```
                    [463 questions, one batch]
                              │
                    [Parse & flag (pure Python)]
                    passage split · type flags · choice normalization
                              │
                    [TIER 1 — fast pass, all 463]
                    Qwen3.5-9B · greedy · capped thinking
                    masked answer token · logprobs recorded
                              │
                    [CONFIDENCE GATE]
                    top1−top2 choice-logprob margin vs threshold τ
                       │                          │
                 margin ≥ τ                  margin < τ
                 (~60–70%)                   (~30–40%)
                       │                          │
                 [ACCEPT]            [TIER 2 — jury, escalated only]
                       │             1. Qwen3.5-9B self-consistency n=6
                       │             2. Gemma 4 E4B verdict
                       │             3. Tool: Python exec (quant) /
                       │                gated RAG (knowledge)
                       │                          │
                       │             [RESOLUTION]
                       │             agree → accept · disagree → tool
                       │             breaks tie · else margin-weighted
                       │             vote (Qwen default tiebreaker)
                       └──────────────┬───────────┘
                                      │
                    [ASSEMBLER → validated answers CSV + audit log JSON]
```

### 3.1 Parser / router (`parsing.py`)

Pure Python, effectively zero latency. For each question:

- **Passage split:** if `Câu hỏi:` (or `Đoạn thông tin:` / `Tiêu đề:` / `Nội dung:` markers) present → separate `context` from `query`.
- **Choice normalization:** map choices to letters `A…K` dynamically; strip any embedded letter prefixes; record `n_choices`.
- **Flags (hints, not routes):**
  - `has_context` — passage markers found
  - `is_quantitative` — digit/unit/formula-token density above threshold
  - `has_refusal_choice` — any choice matches refusal patterns ("Tôi không thể", "không thể cung cấp", "từ chối")
  - `is_legal` — legal vocabulary density (Luật, Nghị định, Điều, Khoản, Thông tư…)
- Flags select the **prompt template** and the **tier-2 tool**; the confidence gate does the actual routing (Lesson C).

### 3.2 Tier 1 — fast pass (`inference.py`, `prompts.py`)

- Single `LLM.generate()` call over all 463 prompts → vLLM continuous batching.
- **Three prompt templates** (Vietnamese system prompts):
  1. *Context-grounded:* "Trả lời CHỈ dựa trên đoạn thông tin sau…" — forbid outside knowledge.
  2. *Quantitative:* "Giải từng bước, kiểm tra lại phép tính trước khi chọn…"
  3. *General knowledge:* concise expert persona; if the question asks how to violate a law/restriction and a refusal choice exists, the refusal choice is correct.
- **Thinking budget:** capped per template (comprehension ≈ 400, knowledge ≈ 600, quantitative ≈ 1000 tokens). Tuned in Phase 5 sweep.
- **Constrained answer:** after thinking closes, generation is restricted via per-request `allowed_token_ids` built from that question's actual letter set (handles 2-, 3-, 4-, 5-, 10-, 11-choice). No free-text parsing exists anywhere in the system (Lesson D).
- **Logprob capture:** full distribution over the legal letter tokens at the answer position.

### 3.3 Confidence gate (`gate.py`)

- Score = logprob(top-1 letter) − logprob(top-2 letter).
- Margin ≥ τ → accept tier-1 answer. Margin < τ → escalate.
- τ calibrated on the dev set targeting ~60–70% tier-1 acceptance; **set conservatively** (over-escalate rather than under — tier-2 latency is affordable, wrong-but-confident 9B answers are not).
- Secondary signal (free): for 10+-choice questions, also compute the **discriminative score** — a single forward pass scoring each choice letter's likelihood — and escalate on generative/discriminative disagreement even if the margin passes.

### 3.4 Tier 2 — the jury (`jury.py`, `tools/`)

Run only on escalated questions (~140–180 expected), again as one batch:

1. **Self-consistency:** Qwen3.5-9B, n=6 parallel samples (shared prefill → cheap), temperature 0.7, top-p 0.95, majority vote over masked answers.
2. **Cross-family verdict:** Gemma 4 E4B, single greedy pass (n=3 vote if time budget allows), same masked-output discipline.
3. **Tool, chosen by flag:**
   - `is_quantitative` → **code execution** (`tools/code_exec.py`): prompt Qwen to emit a standalone Python script ending in `print(answer_value)`; run in `subprocess` with 5 s timeout, no network, restricted imports (math, fractions, datetime, itertools, statistics, numpy, sympy); numeric fuzzy-match the printed value to parsed choice values (rel. tolerance 1%, handle Vietnamese decimal commas, thousand separators, units). Crash/no-match → fall back to signals 1+2.
   - `is_legal` or standalone-knowledge → **gated RAG** (`tools/retrieval.py`): BGE-M3 over a small curated index (Hiến pháp, core bộ luật, key nghị định/thông tư likely in scope, curriculum summaries for chính trị/triết học). Inject top chunks ONLY if top-hit score spikes vs the mean of the next k hits (relative-delta gate, Lesson C); flat profile → answer parametrically with no context (Lesson A).
   - `has_context` → no tool; the passage IS the tool. Escalated comprehension questions get re-asked with an evidence-extraction prompt ("trích câu chứa bằng chứng trước, rồi chọn").

**Resolution rule:**
- Qwen vote == Gemma verdict → accept.
- Disagree + tool produced an answer → tool wins.
- Disagree + no tool answer → higher internal margin wins; ties → Qwen vote (stronger model).
- All three disagree → maximum-effort retry: n=8, doubled thinking budget, RAG forced on; then margin-weighted vote.

### 3.5 Assembler (`main.py`)

- Merge tier-1 accepts and tier-2 resolutions.
- **Hard invariants before write:** exactly 463 rows; every `qid` present once; every answer ∈ that question's legal letter set; no nulls. Any violation → loud failure in dev, safe-default (tier-1 answer or `A`) + log in production.
- Emit `submission.csv` + `audit_log.json` (per question: type flags, tier, tools used, jury votes, margins, 1-line rationale).

---

## 4. Speed engineering (the 10%)

Total wall-clock ≈ (one batched tier-1 pass over 463) + (one batched tier-2 pass over ~160). Levers in order of impact:

1. **Batch everything.** Never loop question-by-question. vLLM offline `LLM.generate()` with all prompts at once; tier-2 jury signals for all escalated questions submitted together (Qwen n=6 samples + Gemma verdicts interleave in one scheduler).
2. **Cap thinking budgets hard** per template (the dominant token cost at 9B scale).
3. **Prefix caching on.** Shared system prompt prefilled once across 463 requests.
4. **Quantize:** 4-bit weights + FP8 KV cache → bigger batches, faster decode.
5. **Discriminative scoring for 10+-choice questions:** one forward pass scoring all letters ≈ free; avoids long deliberation over 10 options.
6. **Lazy-load** BGE-M3/reranker only if any question actually trips the RAG path.
7. **Pre-warm:** one dummy generation per model before the timed run (CUDA graph capture / compile out of the timed window).
8. **Cheap parallelism:** code-exec subprocesses run concurrently with GPU batches (CPU and GPU work overlap).

**Budget target:** tier 1 ≤ 5 min, tier 2 ≤ 8 min, overhead ≤ 2 min → **≤ 15 min end-to-end** on a 16–24 GB GPU. Measure on every dev run; the eval harness reports wall-clock alongside accuracy.

---

## 5. Creativity component (the 10%)

The committee scores what they can see. Three deliberate, demo-able artifacts:

1. **The named architecture: "Entropy-Gated Jury."** The system measures its own uncertainty per question and buys extra compute only where uncertain. Present the confidence-distribution histogram, the escalation set, and tier-1 vs tier-2 accuracy on the dev set — a system that knows what it doesn't know.
2. **Cross-family cross-examination.** Two different model lineages (Qwen + Gemma) as independent jurors; disagreement = measured difficulty. Show the agreement matrix and how disagreement-triggered max-effort retries recovered questions.
3. **Self-explaining audit log.** `audit_log.json` gives a per-question machine-readable rationale: route taken, tools fired, votes, margins. Costs ~0 at inference; demonstrates engineering maturity and reproducibility.

Plus the structural innovations already in the design: logit-level answer control across variable choice counts (2→11), deterministic Python solving with numeric fuzzy-matching, and relative-delta gated retrieval. Write all of this up as a 2-page architecture brief with the charts — prepared in Phase 6, not the night before.

**Rule:** every creative component must survive ablation on the dev set. If it doesn't move accuracy or speed, it ships disabled and is mentioned honestly in the brief as a negative result (committees respect measured negative results more than decoration).

---

## 6. Repository layout

```
agent/
├── main.py                  # orchestrator: load → tier1 → gate → tier2 → assemble
├── config.yaml              # model paths, τ, thinking budgets, n-votes, toggles
├── parsing.py               # passage split, flags, choice normalization (A..K)
├── prompts.py               # 3 tier-1 templates + tier-2 variants (Vietnamese)
├── inference.py             # vLLM wrapper: batching, masking, logprob extraction
├── gate.py                  # margin computation, τ, discriminative cross-check
├── jury.py                  # tier-2 orchestration + resolution rules
├── tools/
│   ├── code_exec.py         # sandboxed subprocess + numeric fuzzy matcher
│   └── retrieval.py         # FAISS/BGE-M3 + relative-delta gate (+ optional rerank)
├── assemble.py              # invariant checks, submission.csv, audit_log.json
├── eval/
│   ├── dev_set.jsonl        # ~100–150 self-written questions w/ gold answers
│   ├── score.py             # accuracy overall + per-type + wall-clock report
│   └── ablate.py            # toggle components via config, diff the report
├── data/
│   ├── corpus/              # curated legal + curriculum texts (source of index)
│   └── index/               # prebuilt FAISS index (baked into image)
├── Dockerfile               # offline build; weights + index baked; vLLM commit pinned
└── scripts/
    ├── build_index.py       # one-off: corpus → chunks → BGE-M3 → FAISS
    ├── download_weights.sh  # one-off, build-time only
    └── dress_rehearsal.sh   # docker run --network none + timing + validation
```

---

## 7. Phased implementation plan

### Phase 0 — Day 1: compatibility spike (GO/NO-GO gates)
- [ ] Confirm exact GPU spec and time limit with organizers (in writing).
- [ ] vLLM main-branch commit: Qwen3.5-9B + Gemma 4 E4B co-load.
- [ ] Verify `allowed_token_ids` constrained decoding works on Qwen3.5's hybrid attention.
- [ ] Verify per-request logprobs at the answer position.
- [ ] Verify prefix caching stability; pin the commit in the Dockerfile.
- **Decision gate (Day 2):** Gemma 4 flaky → swap juror to Qwen3.5-4B. Masking broken → fallback = force 1-token completion and read the sampled letter (still no regex).

### Phase 1 — Days 2–3: walking skeleton
- [ ] `parsing.py` with full choice-count handling (validate on all 463, esp. the 134 ten-choice and the 11-choice item).
- [ ] One generic prompt; tier-1 batched pass; masked output; CSV emitter with invariants.
- [ ] Dockerfile builds; `dress_rehearsal.sh` passes with `--network none`.
- **Milestone:** a valid, submittable run exists. Expected ~70–76% on dev. This is the permanent safety net.

### Phase 2 — Days 4–5: measurement infrastructure
- [ ] Write/collect 100–150 dev questions mirroring the four types and the choice-count mix (include 10-choice items); gold answers verified.
- [ ] `score.py`: overall + per-type accuracy, escalation stats, wall-clock.
- [ ] `ablate.py`: every component behind a config toggle.
- **Rule from here:** no change merges without a dev-set delta. Measured, never vibed.

### Phase 3 — Days 6–9: gate + jury (the accuracy core)
- [ ] Logprob margin extraction; τ sweep; discriminative cross-check for 10+-choice.
- [ ] The three templates; per-template thinking caps (first pass).
- [ ] Self-consistency n=6 voting; Gemma juror; resolution rules.
- [ ] `code_exec.py` + numeric fuzzy matcher (Vietnamese number formats).
- **Expected:** quantitative subset jumps 15–25 pp; overall to ~82–86% on dev.

### Phase 4 — Days 10–11: RAG scalpel + safety handling
- [ ] Curate corpus (narrow: constitution, core codes, likely decrees, chính trị curriculum). Build index at image-build time.
- [ ] Relative-delta gate; optional Qwen-Rerank behind a toggle.
- [ ] Refusal-choice detection + prompt handling; verify near-100% on safety items.
- [ ] **Ablate RAG and rerank:** negative or flat → ship disabled (Lesson A).

### Phase 5 — Days 12–13: speed + robustness tuning
- [ ] Thinking-budget sweep per template (400/800/1200) → pick per-type budgets off the accuracy-latency curve.
- [ ] τ final calibration; n-votes vs latency trade.
- [ ] FP8 KV cache, max batch size, pre-warm, lazy loads.
- [ ] Failure-mode hardening: code-exec timeout storms, empty model output, OOM back-off (auto-reduce batch), single-question crash isolation (one bad question must never kill the run).

### Phase 6 — Days 14–15: creativity artifacts + dress rehearsals
- [ ] `audit_log.json` finalized; confidence histogram, agreement matrix, escalation-accuracy charts.
- [ ] 2-page architecture brief for the committee.
- [ ] ≥3 full dress rehearsals: clean machine, `--network none`, timed, CSV validated, deterministic seeds.
- [ ] Freeze. No changes in the last 24 h except disqualification-level bugs.

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | vLLM main-branch instability with Qwen3.5 / Gemma 4 | Med | Critical | Day-1 spike; pinned commit; juror fallback; masking fallback |
| 2 | Offline build breaks (weights not cached, package phones home) | Med | Critical (DQ) | `dress_rehearsal.sh` from Phase 1; HF_HUB_OFFLINE=1; repeated `--network none` runs |
| 3 | 10/11-choice questions mishandled | Med | High (29% of set) | Dedicated parser tests; discriminative scoring; dev set includes them |
| 4 | τ overfit to hand-made dev set | Med | Med | Conservative τ (over-escalate); sanity-check escalation rate on the real input distribution |
| 5 | GPU spec differs from dev box | Med | High | Confirm spec Day 1; OOM auto-back-off; 4-bit headroom |
| 6 | Code-exec emits unsafe/hanging code | Low | Med | Subprocess, 5 s timeout, import whitelist, no network, output-size cap |
| 7 | RAG hurts accuracy (Lesson A repeat) | Med | Med | Ablation-gated shipping; relative-delta gate; tiny corpus |
| 8 | Time-limit breach | Low | High | 15-min budget vs limit; per-phase wall-clock tracking; n-votes and budgets are the relief valves |
| 9 | Vietnamese numeric formats break the matcher | Med | Med | Normalizer for decimal commas, thousand dots, units; unit tests from dev set |

---

## 9. Score trajectory & definition of done

| Checkpoint | Expected dev accuracy | Wall-clock |
|---|---|---|
| Phase 1 skeleton | 70–76% | ~5 min |
| Phase 3 gate+jury | 82–86% | ~12 min |
| Phase 4–5 tuned | 86–89% | ≤15 min |

**Definition of done:** three consecutive clean dress rehearsals (offline, timed, validated CSV), dev accuracy ≥86%, wall-clock ≤15 min, audit log + architecture brief delivered. That profile yields 80+ on accuracy+speed alone, with the creativity artifacts carrying the total toward 90+.

