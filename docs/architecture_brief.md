# Entropy-Gated Jury — Architecture Brief

**Codename:** Entropy-Gated Jury (EGJ)  
**Event:** HackAIthon 2026 — Bảng C (Innovator)  
**Target:** ≥86% dev accuracy, ≤15 min wall-clock, ≥90 total score

---

## 1. Core Insight

Most questions in the 463-item set have an obvious answer. A standard
single-pass LLM gets those right cheaply. What costs accuracy is the ~30–40%
of questions where the model is genuinely uncertain — and treating them the
same as easy questions. 

**The Entropy-Gated Jury** measures its own uncertainty per question and
buys additional compute only where needed. It is not a router: question
classification determines the prompt template; logprob confidence determines
the compute allocation.

---

## 2. Architecture

```
[463 questions]
     │
     ▼
[Parser]  passage split · type flags · choice normalisation A..K
     │
     ▼
[Tier 1]  Qwen3.5-9B · greedy · capped thinking · masked answer token
          → letter + full logprob distribution over legal choices
     │
     ▼
[Confidence Gate]  margin = logprob(top1) − logprob(top2) vs τ
     │                        │
 margin ≥ τ (~65%)       margin < τ (~35%)
 ACCEPT                  ESCALATE
     │                        │
     │              [Tier 2 — Jury]
     │              1. Qwen3.5-9B  self-consistency n=6, temp=0.7
     │              2. Gemma 4 E4B verdict (different model family)
     │              3. Tool: code_exec (quantitative)
     │                      gated RAG (knowledge/legal)
     │                        │
     │              [Resolution]
     │              agree  → accept
     │              disagree + tool → tool wins
     │              else → margin-weighted vote, Qwen tiebreak
     │                        │
     └──────────────┬──────────┘
                    ▼
     [Assembler]  invariant checks → submission.csv + audit_log.json
```

---

## 3. Key Innovations

### 3.1 Entropy-Gated Compute Allocation

The tier-1 answer comes with a full logprob distribution over the legal
letter set. The margin `logprob(top1) − logprob(top2)` is a calibrated
proxy for model certainty. The threshold τ = 1.5 (configurable) was chosen
to target ~35% escalation — over-escalating rather than under, because
wrong-but-confident tier-1 answers cannot be recovered, while tier-2
latency is affordable.

For questions with ≥10 choices (29% of the set), a discriminative cross-check
is added: a forced single-token pass scoring each letter's likelihood. If the
discriminative and generative winners disagree, escalation is forced regardless
of the margin.

### 3.2 Cross-Family Cross-Examination

Two jurors from different model families (Qwen + Gemma) reduce correlated
errors. Disagreement is a measured difficulty signal — questions where
Qwen and Gemma disagree are systematically harder, and those are the ones
where the third tool signal matters most. The agreement rate and the
disagree-then-recover rate are reported in `audit_log.json`.

### 3.3 Logit-Level Answer Control

The answer token is masked at the logit level: only the legal letter tokens
for this specific question (e.g. A/B/C/D or A/B/C/D/E/F/G/H/I/J for
10-choice) are allowed. This eliminates all parsing failures — the model
cannot produce an illegal answer. Handles 2, 3, 4, 5, 10, and 11-choice
questions identically.

### 3.4 Deterministic Python Solver

For quantitative questions, the model emits a standalone Python script
ending in `print(answer)`. This is executed in a sandboxed subprocess
(5 s timeout, whitelisted imports, no network). The printed value is
fuzzy-matched to choice texts with 1% relative tolerance, handling
Vietnamese number formats (decimal comma, thousand dot, unit suffixes).
Computed answers are exact where model reasoning is approximate.

### 3.5 Relative-Delta Gated Retrieval

A narrow curated corpus (Hiến pháp, core bộ luật, key nghị định/thông tư,
curriculum summaries) replaces general Wikipedia. Context is injected
only when the top-hit cosine score significantly exceeds the mean of the
remaining hits — a relative-delta spike gate. A flat score profile means
the corpus doesn't contain useful information for this query; the model
answers parametrically without fabrication from irrelevant context.

---

## 4. Speed Engineering

Total target: ≤15 min on the competition GPU.

| Lever | Impact |
|---|---|
| Batch everything (one vLLM call per tier) | Dominant |
| Capped thinking budgets per template (400/600/1000 tokens) | High |
| Prefix caching across 463 requests | Medium |
| 4-bit AWQ weights + FP8 KV cache | Medium |
| Discriminative scoring (1 forward pass, not generative) | Low |
| Lazy-load BGE-M3 (only if RAG path fires) | Low |
| Pre-warm before timed run | Low |

---

## 5. Self-Explaining Audit Log

`audit_log.json` records per question:

```json
{
  "qid": "...",
  "answer": "B",
  "tier": 2,
  "flags": {"has_context": false, "is_quantitative": true, ...},
  "tier1": {"letter": "A", "margin": 0.42, "logprob_dist": {...}},
  "jury": {
    "qwen_vote": "B", "qwen_margin": 2.1,
    "gemma_vote": "B", "tool_answer": "B",
    "resolution": "agree:qwen==gemma"
  }
}
```

Cost at inference: zero. Demonstrates engineering maturity, enables
post-run ablation, and provides the committee with a verifiable rationale
trail.

---

## 6. Ablation Protocol

Every component ships with a config toggle. Before submission, run:

```bash
python eval/ablate.py \
    --input data/public-test_*.json \
    --gold  eval/dev_set.jsonl \
    --output data/ablation_results.json
```

Components that show flat or negative accuracy delta ship **disabled**.
This is the commitment to Lesson A (narrow beats wide) operationalised.

---

## 7. Negative Results (Honest)

The following components were designed conservatively:

- **General Wikipedia RAG** — validated negative in the v01 baseline
  (−5.4 pp). Removed entirely. Replaced by narrow corpus.
- **Qwen-Rerank** — ships disabled pending ablation. If reranking does
  not improve accuracy per unit latency, it is mentioned here as a
  tested-and-rejected optimisation.
- **Regex answer extraction** — eliminated. All answer extraction is
  logit-level (token masking). Regex parsing fails silently on edge
  cases; logit masking cannot fail.

---

## 8. Score Trajectory

| Checkpoint | Expected dev accuracy | Wall-clock |
|---|---|---|
| Phase 1 skeleton (current) | 70–76% | ~5 min |
| Phase 3 gate+jury | 82–86% | ~12 min |
| Phase 4–5 tuned | 86–89% | ≤15 min |

**Definition of done:** three consecutive clean dress rehearsals
(offline `--network none`, timed, validated CSV), dev accuracy ≥86%,
wall-clock ≤15 min, audit log + this brief delivered.
