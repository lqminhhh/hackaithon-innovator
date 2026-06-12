# Notes — Vietnamese MCQ QA Agent

> Plain-language companion to `planning.md` (which is the terse build spec for a coding AI).
> This file explains **what we're building, why each choice was made, and what to watch out for** —
> including the things we worked through in chat that aren't obvious from the spec.

---

## 1. What we're building, in one breath

A program that, given a list of Vietnamese multiple-choice questions, picks the right letter for each
and writes a `submission.csv`. It runs entirely offline inside a Docker image (no internet — every
model and document is packed in). It's graded **80% on accuracy, 10% on speed, 10% on creative
design**. Our current version scores **70.84/100 on accuracy alone**; we want to raise accuracy and
also start earning the speed and creativity points.

---

## 2. The core idea (the one thing to remember)

We split the system into **two independent jobs**:

- **The router decides *how hard to think* about each question.** Easy reading question? Answer fast.
  Hard 10-option math problem? Think step-by-step and double-check. This is what makes us both *fast*
  (we don't over-think easy questions) and *accurate* (we spend effort where it matters).
- **The escalation ladder decides *whether the answer is trustworthy*.** After the model answers, we
  look at how confident it was. If it was unsure, we make it think harder or try several times and
  vote.

Why keep them separate? Because the router will sometimes guess wrong about a question. If routing
*also* decided correctness, a wrong guess would mean a wrong answer. By separating them, a routing
mistake only costs us a little *time* (the escalation ladder catches the hard question anyway) —
never a wrong answer. **Router = speed optimization. Ladder = accuracy guarantee.** That design choice
is what makes the whole thing robust.

---

## The pipeline at a glance

```
   ┌───────────────────────────────────────────────────────────────┐
   │                       ONE QUESTION COMES IN                      │
   └────────────────────────────────┬──────────────────────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  ROUTER — "how hard should I think about this?"          │
        │    Layer 1: fast keyword / structure rules               │
        │    Layer 2: match by meaning (only for the leftovers)    │
        └────────────────────────────────┬──────────────────────┘
            ┌──────────────┬──────────────┼───────────────┐
            ▼              ▼              ▼               ▼
        READING          STEM          SAFETY         KNOWLEDGE
     answer from      think step     refusal is      general fact
     the passage      by step +      allowed         or recall
     · fast           vote           here            (look up? see
     · no lookup      · no lookup    · no lookup       box below)
            └──────────────┴──────────────┴───────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  ANSWER via GUIDED CHOICE                                │
        │  (output is forced to be a real option letter, A–J)      │
        │  + a CONFIDENCE number (the "logprob margin")            │
        └────────────────────────────────┬──────────────────────┘
                                     ▼
                          ┌────────────────────────┐
                          │   Confident enough?      │
                          └────────┬────────┬────────┘
                            YES     │        │     NO
                                    ▼        ▼
                                 accept   THINK HARDER:
                                          · STEM → reason again + vote (n=5)
                                          · knowledge → look up docs,
                                            re-answer using them
                                                  │
                                                  ▼
        ┌───────────────────────────────────────────────────────┐
        │  NEVER-CRASH RUNNER → always writes submission.csv       │
        └───────────────────────────────────────────────────────┘
```

**When do we look up documents (retrieval)?** Three simple rules:

```
   reading · STEM · safety       →  NEVER   (answer's in the passage, or it's a calculation)
   law · decrees · local facts   →  ALWAYS  (model is unreliable here, even when it sounds sure)
   everything else               →  answer first, then:
                                       confident?  →  don't look up
                                       unsure?     →  look up + re-answer
```

This is how the system "knows what it doesn't know": it doesn't trust the model to say so — it
watches the confidence number, and force-looks-up the fact-heavy domains regardless.

---

## 3. What's actually in the test (we looked at all 463 public questions)

It's genuinely "every field." Roughly:

- **~30% reading comprehension** — a passage is *printed inside the question*, then it asks about it.
  (Markers like `Đoạn thông tin:`, `Title:`, `-- Document --`.)
- **~28% STEM calculation** — math, physics, chemistry. Many have **10 answer options (A–J)**, not 4.
- **~12% economics/finance** — elasticity, GDP, interest.
- **~18% Vietnam-specific recall** — Hồ Chí Minh thought, Marxism, law/decree numbers, admin
  procedures, temple history, VNPT telecom packages.
- **~12% other concepts** — CS, biology, ethics, business.

**Four things this taught us:**

1. **More than half the test needs NO document lookup.** Reading questions already contain their
   answer; searching a database for them just adds wrong-but-similar distractions. So we turn
   retrieval *off* for those.
2. **Our weak spot is the STEM block, not lookup.** With 10 options, a random guess is only 10%
   right. This is where the 70.84 is leaking points → fix with step-by-step "thinking" + voting.
3. **There's a refusal trap.** Some questions plant an option like *"I can't answer that."*
   - If the question genuinely asks how to do something illegal/harmful → that refusal **is** correct.
   - If it's a normal academic question → the refusal is a **trap** and a real option is correct.
   - We handle this with a prompt instruction so the model doesn't wrongly refuse normal questions.
4. **Retrieval only helps ~15% of questions, and unreliably.** A lot of the Vietnam-specific facts
   are too niche for any database to reliably answer. So retrieval is a *small, carefully-gated*
   add-on — not the heart of the system.

---

## 4. Why one model, not two (the question you pushed back on)

The honest version, because "use two models" is a reasonable instinct:

- **Two *general* models that vote** (like the old design): skip it. Under our 9B-per-model limit the
  second model is a peer, not a smarter checker. Two similar models make *similar* mistakes, so voting
  barely helps (~1–3%) while doubling memory, doubling latency (hurts the speed score), and doubling
  what can break in a one-shot-graded container. Sampling our *one* model several times ("self-
  consistency") gets most of the same benefit for far less.
- **A second model that's a *math specialist*, used only on STEM**: this is defensible — but two
  catches. (a) Qwen3's "thinking mode" already *is* a strong math reasoner; the old reason for separate
  math models has mostly closed. (b) Math-specialist models are often *worse at Vietnamese*, and our
  STEM questions are Vietnamese word problems. So it might not even help.
- **The decisive point:** a second model does **nothing** for our actual weak spot — the niche
  Vietnam recall. *Both* small models lack that knowledge equally. Adding a second model spends the
  whole complexity budget on a gap that's already mostly covered (math) and ignores the gap that's
  actually open (recall, which only retrieval — or accepting the ceiling — can address).

**Decision rule:** build a small labelled test set first, measure STEM accuracy with one model
(thinking + voting). If STEM is already good, a second model is wasted. If STEM is clearly the loser,
*then* add a math specialist **for STEM questions only** and confirm it doesn't hurt Vietnamese. Decide
with data, not in advance.

---

## 5. Can one 8B model really handle all of it? Honest map

No single 8B model "handles everything" — but the failures aren't where you'd expect:

| Question type                                                                  | How well an 8B does                   | Why                                                                                  |
| ------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------ |
| General concepts (econ, biology, CS, ethics, business)                         | **Good**                        | "explain/identify the concept" is exactly what instruct models do                    |
| Standard STEM calc (calculus, probability, circuits, stoichiometry, interest)  | **Good with thinking + voting** | these are "apply a known method carefully" problems; voting catches arithmetic slips |
| Long, tricky multi-step problems                                               | **Risky**                       | one slip → wrong option; voting is the defense                                      |
| **Vietnam niche recall** (decree numbers, local history, package prices) | **Weak — the real gap**        | the model was never trained on these facts                                           |
| Reading comprehension                                                          | **Good**                        | the answer is right there in the passage                                             |

So: concepts and reading are fine; standard STEM is fine once it thinks and votes; the genuinely open
gap is niche Vietnamese recall — and that's a *retrieval-or-accept-the-ceiling* problem, not a
"smarter model" problem. Importantly, that ceiling caps **every team** equally, so it doesn't hurt our
ranking — we don't need to win the unwinnable questions.

---

## 6. Example questions, and how to check if the model knows them

Representative items from the public test (qids), and the benchmark to look up for each:

- **STEM calc** — `test_0006` (exponential-decay differential equation, 10 options),
  `test_0009`/`test_0013` (related rates), `test_0016` (Hess's law), `test_0021` (resistor network),
  `test_0020` (probability), `test_0002`/`test_0008` (elasticity, GDP).
  → *Research:* Qwen3 scores on **MATH, GSM8K, AIME, GPQA** (English) and **VMLU STEM** (Vietnamese).
- **Vietnam knowledge / civics** — `test_0041` (HCM thought; refusal is a *trap* here),
  `test_0024`/`test_0294` (genuinely harmful → refusal is *correct*), `test_0022` (ID-card procedure),
  `test_0030` (temple lineage).
  → *Research:* **VMLU** social-science/political subjects, **ViMMRC**. For the hyper-niche ones, no
  benchmark helps — coverage = "is the source text in our corpus?"
- **Reading comprehension** — `test_0001` (law/religion), `test_0003`/`test_0004` (history),
  `test_0011` (biology), `test_0053`/`test_0104` (VNPT telecom — data is in the prompt).
  → *Research:* any reading benchmark; confirm Vietnamese long-context on VMLU reading.

**But the only real test is empirical:** benchmarks are proxies. Run the model on ~25–40 labelled
questions per bucket and read the per-bucket accuracy. *That* — not benchmark numbers — tells you
whether you need a second model, more voting, or retrieval.

---

## 7. The big risk: the test we're graded on is private

We only see the public sample; the judge grades a hidden set. Two risks:

- **Same distribution, different draw** → small effect; our score will be close to public.
- **Different wording or different mix** → the real risk. Our defenses:
  - The router uses *several* signals + a safe default, and the **semantic layer** (matching by meaning,
    not exact words) catches questions worded in ways our keyword rules don't expect.
  - The **escalation ladder** catches whatever the router gets wrong.
  - We tune our thresholds on *half* our dev set and check them on the other half — so we don't
    accidentally memorize the public sample.
  - We also test on a *different* Vietnamese dataset (VMLU/ViMMRC/old exams) to confirm it generalizes.

**Bottom line:** treat the public test as something to *tune* on, never to *depend* on for correctness.

---

## 8. How we prove it works and earn the 10 creativity points

We can't self-score the public test (no answer key — that's why we only have one accuracy number). So
we **build our own ~150–200 question labelled set**, and produce an **ablation table** showing how each
piece adds accuracy:

```
base model → + router (no lookup on reading) → + guided-choice → + STEM voting → + retrieval
```

That table is both our tuning tool *and* our creativity story: "one model, adaptive effort — every
question gets exactly the reasoning and tools it needs, and here's the measured proof of each step."
Judges reward *measured* design over fancy diagrams.

---

## 9. Glossary (plain definitions)

- **vLLM** — a fast serving engine for LLMs. Its superpower here is **batching**: send all questions at
  once and it processes them together far faster than a one-at-a-time loop. Biggest speed win.
- **AWQ 4-bit** — a way to shrink the model to ~¼ the memory with almost no accuracy loss. Safe on an
  unknown GPU. (FP8 is an alternative but only on newer GPUs — we don't gamble on it.)
- **Thinking mode** (`/think` vs `/no_think`) — Qwen3 can either answer immediately *or* write out
  step-by-step reasoning first. We turn thinking ON for math, OFF for easy questions (faster).
- **Guided choice** — we force the model's answer to be one of the actual option letters (A–J). It
  literally cannot output garbage, so we never need fragile text-parsing to find the answer.
- **Logprob margin** — how much more confident the model was in its top choice vs the runner-up. A
  small margin = "unsure" = escalate.
- **Self-consistency / voting** — ask the model the same hard question several times (with a little
  randomness) and take the majority answer. Cancels out one-off mistakes.
- **RAG (retrieval)** — look up relevant documents from our offline library and show them to the model
  before it answers. Useful for fact-recall questions, harmful for reading questions (distractions).
- **Reranker** — after retrieval pulls ~20 candidate passages, the reranker scores them and we keep
  only the best 3 (and only if they're good enough). Stops junk passages from misleading the model.
- **Centroid / semantic router** — we pre-compute, offline, an "average position in meaning-space" for
  each topic. At runtime we match a question to the nearest topic instantly — robust to rewording.

---

## 10. Quick reference

- **Models:** Qwen3-8B-AWQ (answers) · BGE-m3 (embeddings/retrieval) · BGE-reranker-v2-m3 (reranking).
  Each is under 9B; together they fit in ≤20 GB.
- **Build the MVP first:** skeleton → model → guided-choice → router → escalation → never-crash runner.
  That already scores. Add semantic router, retrieval, and the dev-set/ablation after.
- **Never let it crash. Always emit a complete CSV. UTF-8 everywhere. No internet at run time.**
- **Decide the two-model question, retrieval thresholds, and voting count with the dev set — not by guessing.**
