# ISSUE LOG — Vietnamese MCQ QA Agent

<!-- Companion to planning_v2.md. Read before implementing any segment.
     Severity: CRITICAL > HIGH > MEDIUM > SMALL
     Each issue states exactly what segment it affects and what to fix. -->

## ▼ READ THIS FIRST

The architecture is right, but items 1–3 are genuine spec bugs/risks that would surface as
"why is STEM not improving" and "why is it slow/OOM" after days of confused debugging.
Items 5 and 7 change what you do first (textbook corpus; label early).

---

## 🔴 CRITICAL

### Issue 1 — guided_choice and thinking mode are incompatible as specified
**Affects:** S2, S4

In vLLM, `guided_choice=["A".."J"]` constrains the entire completion to be just that letter —
which means the model can't emit its reasoning chain first. S2/S4 as written would silently
suppress `/think` on the STEM route, gutting the very thing that fixes the STEM leak.
The correct pattern is two-step extraction:

- Pass 1: generate reasoning freely (think mode, no constraint).
- Pass 2: append the reasoning + `"Đáp án: "` and call again with `guided_choice` on just the
  letter. Prefix caching makes pass 2 nearly free (it reuses the KV cache of prompt+reasoning).

This also gives you a clean truncation policy for free: if thinking hits the token cap mid-chain,
you still run pass 2 and force a letter out. Without this fix, a truncated chain returns garbage.

---

### Issue 2 — the escalation ladder, implemented naively, destroys the speed win
**Affects:** S4, S7

The ladder is described per-question (answer → check margin → escalate), but vLLM's 10–20×
speedup comes from batching. A coding AI following the spec literally will write a per-question
loop. The implementation must be wave-based:

- Wave 1 = batch all first passes
- compute margins
- Wave 2 = batch all escalations together
- Wave 3 = batch all RAG re-answers

Checkpoint per wave-chunk, not per question. This is the single most important implementation
note that isn't in the spec.

---

### Issue 3 — VRAM contention on an unknown card
**Affects:** S1, S6, Dockerfile

`gpu_memory_utilization=0.85` tells vLLM to pre-allocate 85% of the GPU — then BGE-m3 and the
reranker need VRAM outside that pool. On your 24GB card it fits (~3.6GB slack); on a 16GB judge
card it's a coin flip; smaller, it OOMs on startup — which violates invariant #1 in the worst way.
Safe fix: run the embedder + reranker on CPU. The volume is tiny (a few hundred embeds, ~20 rerank
pairs per fired question) — minutes of CPU time, zero OOM risk. Load whatever does go on GPU
before vLLM claims its pool.

---

## 🟠 HIGH

### Issue 4 — sanity-check the math on where 29 points live
**Affects:** overall strategy

If reading is ~87%, STEM ~45% (plausible for 10-option items with the old setup), knowledge ~75%:
`0.30×87 + 0.28×45 + 0.42×75 ≈ 70.8` — almost exactly your score. That's reassuring (the plan
targets the right leak) and calibrating: STEM 45→75% ≈ +8.4 overall, reading +1.5, knowledge
with selective RAG +2 → realistic landing zone ~80–83 accuracy, not 90+. Corollary: never trade
accuracy for speed — 1 accuracy point ≈ 0.8 final points; the entire speed axis is worth 10.

---

### Issue 5 — the ~18% civics/HCM/law bucket is NOT unknowable niche — it's standard curriculum
**Affects:** S6 corpus

Tư tưởng HCM, Mác–Lênin, kinh tế chính trị, lịch sử Đảng all have official giáo trình
(textbooks) freely available. The plan's corpus says "legal codes + wikipedia" — add the
textbooks. This is the cheapest targeted lift for that bucket and directly addresses where
RAG-from-wikipedia would whiff. (The truly unknowable stuff — temple lineage, VNPT prices —
stays capped for everyone.)

---

### Issue 6 — the confidence net has a known blind spot: confidently-wrong recall
**Affects:** S4, S5

High margin on a hallucinated fact never escalates. `FORCE_RETRIEVE_DOMAINS` patches this only
where the centroid router catches the domain. Also, letter-token margins carry position bias
(LLMs favor certain option slots). Two mitigations worth dev-set-testing:

- Per-route thresholds: margin distributions differ wildly between READING and KNOWLEDGE —
  one global `MARGIN_LOW=0.15` is crude.
- Shuffling option order across the n=5 self-consistency samples — a free de-biasing trick since
  the samples are batched anyway.

---

## 🟡 MEDIUM

### Issue 7 — circular dependency: S5 needs labelled data but the dev set is S8, listed last
**Affects:** build order

In practice you must do a "S8-lite" (bucket-label the public 463, even roughly) before S5 — and
honestly before S4 tuning too. The dev set isn't the last segment; it's the second thing to build
after the MVP runs.

---

### Issue 8 — dev-set labels are least trustworthy exactly where you need them most
**Affects:** S8

Strong-model labels on niche Vietnam recall will be wrong at some rate. For law questions, verify
labels against the actual legal text you're putting in the corpus anyway. Include the refusal qids
both ways (harmful→refuse-correct, benign→refusal-trap) to measure that boundary.

---

### Issue 9 — timeline: 11 days
**Affects:** overall project

MVP 2–3 days, dev set 1–2, corpus+RAG 2–3 (time-box corpus work — it's unbounded by nature),
ablations+Docker 2. That leaves ~zero slack: feature-freeze around June 20, then only hardening
+ ≥2 full dress rehearsals of the image with networking disabled. An untested Docker image is the
most common way strong teams score zero.

---

### Issue 10 — the lever we're consciously not pulling: fine-tuning
**Affects:** overall strategy

A LoRA SFT on Vietnamese exam MCQ (VMLU-style, đề thi banks) could lift the civics bucket
further, and a top competitor might do it. With 11 days and no MVP yet, I'd still skip it — it
risks degrading the thinking mode (catastrophic forgetting hits reasoning first) and eats the
timeline. But it's the one deliberate omission; if the post-MVP dev set shows civics still
leaking badly and you're ahead of schedule, it's the next lever, applied only to the no-think
path.

---

## 🟢 SMALL

### Issue 11 — verify AWQ quant quality
**Affects:** S1

Verify an official/quality AWQ quant of Qwen3-8B exists for your pinned vLLM version (community
quants vary).

---

### Issue 12 — Qwen3 decoding settings
**Affects:** S1, S4

Qwen3 official guidance: don't greedy-decode in thinking mode (temp 0.6/top_p 0.95 — our SC
settings comply); letter-only greedy in no_think is fine.

---

### Issue 13 — submission format must match exactly
**Affects:** S0, S7

Keep `submission.csv` format byte-identical to the run that scored 70.84.
