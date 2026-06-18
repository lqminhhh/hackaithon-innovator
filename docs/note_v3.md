# Notes v3 — Vietnamese MCQ QA Agent (single model, ≤5B)

> Plain-language companion to `planning_v3.md` (the terse build spec for a coding AI).
> This file explains **what we're building, what changed from v2, why each choice was made, and
> what to watch out for** — including the things we learned the hard way from our own v2 runs.
> For the **evidence behind every choice** (which research/finding justifies it and where it lives in
> the build), see `research_v3.md`.

---

## 1. What we're building, in one breath

A program that, given a list of Vietnamese multiple-choice questions, picks the right letter for each
and writes a `submission.csv`. It runs entirely offline inside a Docker image (the model is packed in,
no internet). It's graded **80% accuracy, 10% speed, 10% creative design**. Our best version so far
scores **79.91 on accuracy**; an external **finding** puts the bar to beat at **83.59**. The job of v3
is to recover that 79.91 *at high speed* and then close the ~3.7-point gap.

---

## 2. What changed from v2 — and why it's mostly good news

The rules tightened: **one model, at most 5B parameters, 16 GB card, and no embedding/reranker models.**

That kills three pieces of our v2 system — **RAG (document lookup), the reranker, and the semantic
router** — because they were all extra models. That sounds bad, but it's a relief:

- **Our own numbers showed RAG was *hurting*:** 78.83 with RAG vs **79.91 without**. The findings say the
  same — looking things up poisons a small model with wrong-but-similar text.
- Dropping it **frees the whole graphics card** (the reranker used to run out of memory and crash the run),
  makes us **faster**, and makes us **simpler and harder to break**.

So v3 is v2 with the fragile, now-illegal parts removed — and we lose ~nothing on accuracy. The model
also moved to **Qwen3.5-4B** (smaller, newer, already in our code).

---

## 3. The core idea (unchanged — the one thing to remember)

We split the system into **two independent jobs**:

- **The router decides *how hard to think* about each question.** Easy reading question → answer fast.
  Hard 10-option math problem → think step-by-step and vote. This makes us both *fast* (don't over-think
  easy ones) and *accurate* (spend effort where it matters).
- **The escalation ladder decides *whether the answer is trustworthy*.** After the model answers, we look
  at how confident it was (the "margin"). If it's unsure, we make it think harder and vote.

Keep them separate so a routing mistake only costs a little *time*, never a wrong answer.
**Router = speed. Ladder = accuracy.**

### The pipeline at a glance

```
   ┌───────────────────────────────────────────────────────────────┐
   │                       ONE QUESTION COMES IN                   │
   └────────────────────────────────┬──────────────────────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  ROUTER (rules only) — "how hard should I think?"     │
        └────────────────────────────────┬──────────────────────┘
            ┌──────────────┬──────────────┼───────────────┐
            ▼              ▼              ▼               ▼
        READING          STEM          SAFETY         KNOWLEDGE
     answer from      think step    refusal is      general fact
     the passage      by step +     allowed         / recall
     · fast           always vote   here            (model only —
                      (n=3–7)                         NO lookup now)
            └──────────────┴──────────────┴───────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  ANSWER via TWO-PASS GUIDED CHOICE                    │
        │   pass 1: reason freely · pass 2: force a real letter │
        │  + a CONFIDENCE number (the "logprob margin")         │
        └────────────────────────────────┬──────────────────────┘
                                     ▼
                          ┌──────────────────────────┐
                          │   Confident enough?      │
                          └────────┬────────┬────────┘
                            YES     │        │     NO
                                    ▼        ▼
                                 accept   THINK HARDER: reason again
                                          several times + vote
                                                  │
                                                  ▼
        ┌──────────────────────────────────────────────────────────┐
        │  NEVER-CRASH WAVE RUNNER → always writes submission.csv  │
        └──────────────────────────────────────────────────────────┘
```

**Do we ever look up documents now?** No. Retrieval is gone (illegal, and it hurt us). Fact-recall
questions are answered from the model alone; if it's unsure, it re-reasons and votes instead of looking up.

---

## 4. Where we stand vs the target (the honest numbers)

| Run                          | Accuracy        | Speed              | What it tells us                                                             |
| ---------------------------- | --------------- | ------------------ | ---------------------------------------------------------------------------- |
| v02_gamma                    | **79.91** | 10.8 s/q           | our best — but too slow;*every* STEM question votes                       |
| v02_gamma + RAG              | 78.83           | 18.0 s/q           | RAG made it**worse** and slower → proof to drop it                    |
| v02_delta                    | 77.54           | **2.45 s/q** | wave-batching =**4.4× faster** …                                     |
|                              |                 |                    | … but it also skipped voting on 38 STEM items and**lost 2.37 points** |
| **findings (to beat)** | **83.59** | —                 | external benchmark                                                           |

The lesson is sharp: **the speed came from wave-batching, which is free. We then threw away points by
also cutting the voting — don't.** v3's first milestone is "gamma's accuracy at delta's speed."

Realistically a 4B model tops out around **83–84** on this test; the open leak is the **STEM block**.
Pushing past the findings' 83.59 means winning more STEM, not chasing the unanswerable niche questions
(those cap everyone equally).

---

## 5. What's actually in the test (we looked at all 463 public questions)

Roughly: **~30% reading comprehension** (a passage is printed in the question), **~28% STEM calculation**
(many with **10 options**, one with 11), **~18% Vietnam-specific recall** (HCM thought, Marxism, law/decree
numbers, admin procedures, niche local facts), **~12% economics/finance**, **~12% other** (CS, biology,
ethics, business).

Our router sorts the 463 into: **reading 100 · STEM 216 · knowledge 143 · safety 4.**

Three things this taught us:

1. **STEM is the leak.** With 10 options a guess is only 10% right; this is where the score bleeds → fix
   with step-by-step thinking + voting on *every* STEM item.
2. **There's a refusal trap.** Some questions plant an "I can't answer that" option. If the question is
   genuinely harmful, refusing is *correct*; if it's a normal academic question, refusing is a *trap*. A
   prompt line handles this so the model doesn't wrongly refuse.
3. **The niche Vietnam recall is a ceiling, not a bug.** We used to attack it with lookup; that's gone now.
   It caps every team equally, so it doesn't hurt our ranking — we don't need to win the unwinnable.

---

## 6. Where the points are — and where they leak

Plausible per-bucket picture behind the 79.91: reading ~87%, STEM ~70–75% (up from ~45% before
thinking+voting), knowledge ~75%. **STEM is both the biggest bucket and the biggest leak**, so it's where
extra effort pays. And the trade-off math to memorize: **1 accuracy point ≈ 0.8 final points; the entire
speed axis is worth only 10.** That's why we **never trade accuracy for speed** — we buy speed from
batching instead.

---

## 7. Two non-obvious traps we already hit (read before coding S2/S4)

These cost real debugging time in v2 — don't rediscover them:

- **Guided-choice silently kills thinking.** If you force the output to be a single letter, the model can't
  reason first. **Fix:** two passes — reason freely, *then* a second call forces the letter. Prefix caching
  makes the second pass nearly free. (And if reasoning runs long and gets cut off, the second pass still
  produces a valid letter — no garbage.)
- **A per-question loop destroys the speed win.** vLLM is fast because it processes a *batch* at once. Write
  the escalation ladder in **waves** (all first answers together → all the "think harder" retries together),
  not one question at a time. This is the difference between 10.8 s/q and 2.45 s/q.

---

## 8. The big risk: the graded test is private, and the container must never crash

We only see the public sample; the judge grades a hidden 2,000-question set. Defences:

- The router uses several signals + a safe default; whatever it gets wrong, the **escalation ladder**
  catches.
- We tune thresholds on *half* our dev set and check them on the other half, so we don't memorize the
  public sample; we also sanity-check on a different Vietnamese dataset.
- **Never let it crash. Always emit a complete CSV.** A missing file scores 0 — worse than any wrong answer.
  We checkpoint after each wave and resume if killed.

**Treat the public test as something to *tune* on, never to *depend* on.**

---

## 9. What we deliberately DON'T do (and why)

- **RAG / embeddings / reranker** — illegal now, *and* our own runs show it hurt (−1.08) while adding an
  OOM crash risk. Gone.
- **A second model** — illegal, and it never helped our real gap (niche recall); two small models make the
  same mistakes.
- **Tool / code-execution reasoning** — the findings measured **−16.5**: a 4B writes buggy code and breaks
  questions it would have gotten right.
- **Fine-tuning** — the findings show it *lost* points, and it tends to damage the thinking mode
  (the reasoning is the first thing to break). It's our one deliberate omission: only a last resort, only on
  the no-think path, only if we're ahead of schedule.

This "we built it, measured it, and removed what hurt" discipline **is** our creativity story (§10).

---

## 10. How we earn the 10 creativity points

We can't self-score the public test (no answer key). So we build our own **~150–200 question labelled set**
and produce an **ablation table** showing how each piece adds accuracy:

```
base model → + router → + two-pass guided-choice → + STEM adaptive voting → + option-shuffle de-bias
```

That table is both our tuning tool and our story: *"one model, adaptive effort — every question gets
exactly the reasoning it needs, and here's the measured proof, including the levers we removed because the
data said so."* Judges reward *measured* design over fancy diagrams.

---

## 11. Glossary (plain definitions)

- **vLLM** — a fast serving engine. Its superpower is **batching**: send all questions at once. Biggest
  speed win.
- **Wave batching** — running the pipeline in stages across the *whole* set at once (all first answers, then
  all retries) instead of one question at a time. Turns 10.8 s/q into 2.45 s/q.
- **Two-pass guided choice** — pass 1 lets the model reason; pass 2 forces the answer to be a real option
  letter (A–Z). It literally cannot output garbage, so we never parse text.
- **Logprob margin** — how much more confident the model was in its top choice vs the runner-up. Small
  margin = "unsure" = think harder.
- **Self-consistency / voting** — ask the same hard question several times (a little randomness) and take the
  majority. For STEM we always do this (n=3 if confident, n=7 if not).
- **Option shuffling** — re-order the choices between voting samples so the model can't be biased toward a
  particular slot. Free, since the samples are batched anyway.
- **Thinking mode** (`/think` vs `/no_think`) — Qwen can answer immediately or write step-by-step reasoning
  first. ON for STEM, OFF for easy questions (faster). In thinking mode use temperature 0.6 (don't greedy-decode).
- **RAG / reranker / semantic router** — *(removed in v3)* document lookup and the extra models that drove
  it. Illegal under the one-model rule, and they hurt accuracy. Listed here only so nobody re-adds them.

---

## 12. Quick reference

- **Model:** `Qwen3.5-4B` — one model, ≤ 5B, offline, fits ≤ 14 GB. No embedder, no reranker.
- **Build the MVP first:** skeleton → model → two-pass guided-choice → router → wave escalation →
  never-crash runner. That already scores ≈ 79.9 fast. Add the dev set + ablation after.
- **STEM always votes** (the leak). **Speed comes from waves, not from skipping votes.**
- **Never crash. Always emit a complete CSV. UTF-8 everywhere. Support A–Z. No internet at run time.**
- **Decide thresholds (margins, voting count, token caps) on the dev set — not by guessing.**
- **Do ≥ 2 offline dress rehearsals of the Docker image before submitting.**
