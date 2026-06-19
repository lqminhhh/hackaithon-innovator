# BUILD SPEC v3 — Vietnamese MCQ QA Agent (single-model, ≤5B)

<!-- Build contract. Terse on purpose. Companions: note_v3.md (rationale) · research_v3.md (evidence map / where every lever's research lives).
     v3 = the v2 architecture MINUS the now-illegal multi-model parts, re-tuned for the new rules
     and the measured v2 results. Read "WHAT CHANGED" before touching any segment. -->

## ▼ READ THIS FIRST (load before any segment)

**WHAT CHANGED v2 → v3 (rules tightened — this drives everything):**
- v2 rules: ≤9B *per* model, multiple models allowed, ≤20 GB VRAM.
- **v3 rules (confirmed): ONE LLM ≤ 5B TOTAL params · 16 GB VRAM · NO embedding/rerank · no external/API model · offline.**
- **REMOVED, now illegal:** `S5` semantic router (BGE-m3) and `S6` RAG (BGE-m3 + reranker + corpus). Both used extra models → banned.
- **This costs us nothing real:** measured `v02_gamma_rag` 78.83 < `v02_gamma` 79.91 — RAG was *hurting*. Findings agree retrieval poisons small models. Dropping it also frees the whole card (no more VRAM contention, Issue 3 dead) and removes the reranker-OOM failure mode.
- Model drift: `Qwen3-8B-AWQ` → **`Qwen3.5-4B`** (≤ 5B total; already in the code).

**What:** Offline Docker agent. Input JSON of Vietnamese MCQ → output `submission.csv` (`qid,answer`). Answer is a single letter.

**One-line architecture:** one LLM (`Qwen3.5-4B` on vLLM) + a rule router that decides *how much compute* each question gets + a **wave-batched** escalation ladder that decides *if the answer is good enough* + **two-pass** guided-choice decoding (reason freely, then constrain the letter) — inside a never-crash runner. **No RAG, no second model.**

**WHERE WE STAND (measured, public-463):**
| Run | Acc | Speed | Note |
|---|---|---|---|
| v02_gamma | **79.91** | 10.77 s/q | best accuracy; all STEM run SC; too slow |
| v02_gamma_rag | 78.83 | 18.04 s/q | RAG *hurt* −1.08 + reranker OOM → **proof to drop RAG** |
| v02_delta | 77.54 | **2.45 s/q** | wave-batching = 4.4× faster (the speed win) … |
| | | | …but STEM early-exit (skip SC on 38 items) cost −2.37 |
| **v3 target** | **≈ gamma** | **≈ delta** | recover 79.91 at 2.45 s/q, then attack the STEM leak |

**Findings to beat = 83.59 (public-463).** Gap from our 79.91 ≈ **3.7 pts**. Realistic 4B ceiling on this set ≈ 83–84 (Issue 4 math); every point past gamma is hard and lives in the STEM bucket.

**INVARIANTS (never violate):**
1. **Always write a complete `submission.csv`** (every qid → a valid letter; any per-question exception → `FALLBACK="A"` or best prior stage). A crash = score 0.
2. **UTF-8 everywhere.** Preserve Vietnamese diacritics.
3. **ONE model, ≤ 5B total params. NO embedding / rerank / second model.** Target **≤ 14 GB VRAM** (16 GB cap, leave slack for the unknown judge card).
4. **No internet at inference.** Model baked into the image.
5. **No regex answer parsing.** Answers come from vLLM `guided_choice`.
6. **Router = optimization. Escalation = correctness.** A misroute must only cost *speed*, never force a wrong/blank answer. Default route when unsure = `KNOWLEDGE` (full path).
7. **Speed comes from wave-batching, never from cutting reasoning.** `1 accuracy pt ≈ 0.8 final pts; the entire speed axis is worth 10.` The `v02_delta` regression is the receipt: skipping SC to go faster lost more than it saved.
8. **Support labels A–Z**, not just A–J (the public set has an 11-choice item).

**Data contract:**
```
question = {"qid": "test_0001", "question": str, "choices": [str, ...]}   # 2..~11 choices
output   = {"qid": "test_0001", "answer": "A"}                            # one valid letter
```

**Build order (each segment self-contained; build + verify one before the next):**

| Seg | Name | Priority | Gives you |
|---|---|---|---|
| S0 | Skeleton + config + I/O (A–Z) | **MUST** | runnable shell emitting a valid CSV |
| S1 | vLLM + Qwen3.5-4B | **MUST** | serving, think/no_think, prefix caching |
| S2 | **Two-pass** guided-choice + margin | **MUST** | reason-then-constrain extraction + confidence |
| S3 | Rule router + prompts (refusal-trap) | **MUST** | routing |
| S4 | **Wave-batched** escalation + adaptive SC | **MUST** | accuracy net *and* speed |
| S7 | Never-crash wave runner + checkpoint | **MUST** | the safety guarantee |
| S8 | Dev set + ablation harness | SHOULD | tuning + creativity-score evidence |
| ~~S5~~ | ~~Semantic router (BGE-m3)~~ | **REMOVED** | illegal (embedding model) |
| ~~S6~~ | ~~RAG (BGE-m3 + reranker + corpus)~~ | **REMOVED** | illegal (embed/rerank) + measured to hurt |

**MINIMAL VIABLE SUBMISSION = S0+S1+S2+S3+S4+S7.** That is essentially "`v02_gamma` logic on wave batching, no RAG" — it should land ≈ 79.9 at ≈ 2.5 s/q. Ship it, then tune on S8.

**Central config (`config.py`):**
```python
LLM_MODEL   = "Qwen/Qwen3.5-4B"     # ≤5B total, single model (needs transformers>=5.2, vllm>=0.17)
QUANT       = None                  # F16 default (4B≈8GB fits 16GB). Measure "awq" for speed on small/memory-bound cards.
GPU_MEM_UTIL= 0.80                  # 16GB × 0.80 = 12.8GB; leaves 3.2GB for OS/driver/desktop on unknown judge cards
MAX_MODEL_LEN = 4096

# Per-route low-margin thresholds (Issue 6: margin distributions differ wildly by route)
MARGIN_LOW  = {"READING":0.10, "STEM":0.15, "KNOWLEDGE":0.20, "SAFETY":0.05}

# STEM is the leak → never skip SC on it (the v02_delta lesson)
STEM_DIRECT_MARGIN = 1.01           # DISABLED: every STEM item runs self-consistency
SC_N_STEM   = {"high":3, "low":7}   # adaptive depth, not early-exit: n=3 if margin high, n=7 if low
SC_N        = 5                     # default SC for non-STEM escalation
SC_TEMP     = 0.6                   # Qwen3 thinking guidance — do NOT greedy-decode in think mode
SC_TOP_P    = 0.95
SC_SEED     = 1234                  # deterministic SC (reproducible score across judge re-runs)
SHUFFLE_OPTIONS_IN_SC = True        # free letter-position de-bias (samples are batched anyway)

TOK = {"READING":512, "STEM":3072, "KNOWLEDGE":256, "SAFETY":128}  # STEM cap tunable on dev (test 2048/1536 for speed)
FALLBACK = "A"
# REMOVED: EMBED_MODEL, RERANK_MODEL, RERANK_MIN, FORCE_RETRIEVE_DOMAINS  (no RAG in v3)
```

---

## S0 — Skeleton + config + I/O   [MUST]   Depends: none
**Goal:** runnable program that loads questions and writes a valid CSV.
**Deliver:** `config.py`, `io_utils.py`, `data_loader.py`, `main.py`.
**Spec:**
- `load_questions(path)`: UTF-8; optional `ftfy.fix_text` (guard: no-op on clean input). Accept CSV **and** JSON.
- `letters(n) -> list[str]`: `["A",...]` length n, **up to "Z"** (Issue: 11-choice item exists).
- `write_submission(rows, path)`: `qid,answer` UTF-8, one row per input qid, **even on partial runs**.
- `main.py`: load → answer `FALLBACK` for all → write CSV.
**Accept:** 5-question sample → CSV with 5 valid rows, UTF-8, no crash. An 11-choice item maps to A–K cleanly.

## S1 — vLLM + Qwen3.5-4B   [MUST]   Depends: S0
**Goal:** one vLLM engine serving Qwen3.5-4B with think/no_think control.
**Deliver:** `llm.py` (`class LLM`).
**Spec:**
- `LLM(model=LLM_MODEL, quantization=QUANT, gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True)`.
- `mode` switch: thinking via Qwen3.5 chat template (`enable_thinking=True/False`).
- `generate(prompts: list[str], sampling) -> list[output]` — **always batched** (pass the whole list).
- Expose logprobs via `SamplingParams(logprobs=…)`.
**Accept:** batch of 3 returns 3 completions; no_think short, think emits a reasoning block; **VRAM resident ≤ 14 GB** (single model, no RAG → comfortable now).

## S2 — Two-pass guided-choice + logprob margin   [MUST]   Depends: S1
**Goal:** extraction that can only return a valid letter, plus a confidence number — **without suppressing reasoning.**
**Deliver:** `extract.py`.
**Spec (the Issue-1 fix — do NOT collapse to one pass):**
- **Pass 1:** generate reasoning freely (think mode, no constraint), capped at `TOK[route]`.
- **Pass 2:** append the reasoning + `"\nĐáp án: "` and call again with `guided_choice=letters(n)` on **just the letter**. Prefix caching reuses Pass-1 KV → Pass 2 is nearly free.
- Request logprobs for the choice token. Return `ChoiceResult(letter, margin, per_letter_logprob)` where `margin = prob(top1) − prob(top2)` (softmax over the constrained letters).
- This also gives a free truncation policy: if Pass-1 thinking hits the cap mid-chain, Pass 2 still forces a valid letter (no garbage).
**Accept:** a 10-choice question returns exactly one of A–J + a float margin in [0,1]; never returns anything outside the letter set; a deliberately truncated chain still yields a letter.

## S3 — Rule router + prompts   [MUST]   Depends: S0
**Goal:** classify each question into READING / STEM / SAFETY, else None (→ KNOWLEDGE default). **Layer-1 rules only** (the semantic Layer-2 is removed — it needed an embedder).
**Deliver:** `router.py` (`route_l1`, `route_question`), `prompts.py`.
**Spec:**
```python
def route_l1(q) -> str | None:
    t = q["question"]; tl = t.lower(); n = len(q["choices"])
    if any(m in t for m in ("Đoạn thông tin","Đoạn văn","-- Document",
                            "Title:","Tiêu đề:","Nội dung:")) or len(t) > 600:
        return "READING"
    if "$" in t or n >= 8 or any(k in tl for k in STEM_KW):
        return "STEM"
    if any(k in tl for k in HARMFUL_KW):
        return "SAFETY"
    return None    # -> KNOWLEDGE default
```
- `STEM_KW = {"tính","bao nhiêu","phương trình","đạo hàm","tích phân","xác suất","vận tốc","điện trở","nồng độ","co giãn","lãi suất","ma trận","giá trị"}`. `HARMFUL_KW`: explicit illegal/harmful-intent phrasing.
- **All route prompts include the refusal-trap system line:**
  > Đây là bài thi trắc nghiệm. Chọn phương án đúng nhất. Chỉ chọn phương án "không thể trả lời / từ chối" khi câu hỏi thực sự yêu cầu hướng dẫn hành vi phạm pháp hoặc gây hại. Với mọi câu hỏi học thuật, khoa học, kiến thức thông thường, luôn chọn một phương án nội dung.
  - READING: "Chỉ dựa vào đoạn văn trong câu hỏi để trả lời."
  - STEM: "Giải từng bước rồi chọn đáp án."
**Accept:** on the public 463, route counts reproduce ≈ `reading=100, stem=216, knowledge=143, safety=4`. READING must **not** fire on STEM (misroutes elsewhere are fine — escalation covers them).

## S4 — Wave-batched escalation + adaptive self-consistency   [MUST]   Depends: S1,S2,S3
**Goal:** per-route compute + the correctness net, **implemented in waves so batching survives** (the Issue-2 fix).
**Deliver:** `solve.py`.
**Per-route policy:**
| Route | mode | max_tok | self-consistency |
|---|---|---|---|
| READING | no_think | 512 | direct; reason+SC `n=3` only for cause/purpose items ("lý do","tại sao","vì sao","mục đích") |
| STEM | think | 3072 | **always** SC; adaptive depth `n=3` (margin high) / `n=7` (margin low). **No early-exit.** |
| KNOWLEDGE | no_think | 256 | direct; if margin < `MARGIN_LOW["KNOWLEDGE"]` → SC `n=5` |
| SAFETY | no_think | 128 | direct; if harmful + refusal option present → force the refusal label |
**Wave structure (mandatory — a per-question loop kills the 4.4× batching win):**
```
Wave 1: batch ALL first passes (two-pass S2) → (letter, margin) per qid
        compute margins; partition by route + threshold
Wave 2: batch ALL escalations together (STEM SC, low-margin KNOWLEDGE SC, READING-reason SC)
        SC = sample SC_N completions (temp SC_TEMP, top_p SC_TOP_P, seed SC_SEED),
             shuffle option order per sample (de-bias), majority-vote the remapped letters
finalize: tie → highest-logprob letter. NEVER blank.
```
- Checkpoint per **wave-chunk**, not per question (S7).
- **No RAG branch exists** — low-margin KNOWLEDGE escalates to SC, not retrieval.
**Accept:** all STEM items run SC (none skipped); low-margin knowledge triggers a 2nd-pass SC; every qid yields exactly one letter; wall-clock ≈ delta (≈ 2.5 s/q), not gamma (10.8 s/q).

## S7 — Never-crash wave runner + checkpoint   [MUST]   Depends: S0,S4
**Goal:** the robustness guarantee.
**Deliver:** `run.py`.
**Spec:**
- Warmup pass before Wave 1 (trigger compile/caches).
- Per-question `try/except` inside each wave → on failure, answer = best-so-far or `FALLBACK`.
- **Checkpoint** the answer dict to `.ckpt` after each wave-chunk; on startup, resume from it.
- Always write the full `submission.csv` on completion **and** on signal/exception (atexit/finally).
- Order questions to maximize batching (group by route/mode) but keep qid mapping intact.
**Accept:** kill mid-run → restart → resumes, complete CSV. Force an exception on one qid → that qid gets `FALLBACK`, all others correct, no crash.

## S8 — Dev set + ablation harness   [SHOULD]   Depends: S4
**Goal:** measure accuracy and produce the creativity-score evidence. (Do an "S8-lite" bucket-label of the public 463 *before* tuning S4 — Issue 7.)
**Deliver:** `dev_set.jsonl` (labelled), `eval.py`, `ablation.md`.
**Spec:**
- ~150–200 labelled questions: hand-label STEM (verifiable) + reading-comp; strong-model-label the rest + spot-check. Cover every bucket (≥25–40 each). Include refusal qids both ways (harmful→refuse-correct, benign→refusal-trap) to measure that boundary (Issue 8).
- **Split: tune thresholds on half, report on the other half.** Drop any gain that doesn't survive the split.
- Ablation rows (no RAG row in v3): `base → +router → +two-pass guided-choice → +STEM adaptive SC → +option-shuffle de-bias`.
- Validate on an **external** set (VMLU / ViMMRC / old THPT) to confirm it generalizes — **as eval only** (no retrieval, no training).
**Accept:** per-bucket accuracy + ablation tables produced; `MARGIN_LOW`, `SC_N_STEM`, `TOK["STEM"]` chosen on the tune half, reported on the held-out half.

---

## Dockerfile / packaging   [MUST before submit]
- Base CUDA image; install vLLM (≥0.17) + transformers (≥5.2); **bake the model at build time** (offline at run).
- Entry: `python run.py --input <json|csv> --output submission.csv`.
- `HF_HUB_ENABLE_HF_TRANSFER=0` unless `hf_transfer` is installed (Issue: breaks downloads).
- **≥ 2 full dress rehearsals with networking disabled** before submit — an untested image is the most common way strong teams score 0.

## Open items to confirm (don't block MVP)
- Confirm `Qwen3.5-4B` total params ≤ 5B and an AWQ quant exists for the pinned vLLM (verify quant quality — Issue 11).
- vLLM `guided_choice` supports ≥ 11 letters in the pinned version.
- Qwen3.5 thinking-mode template flag name in the pinned vLLM.
- **GPU is unknown** → AWQ-4bit de-risks small/memory-bound cards; F16 if the card is ≥16 GB and fast. A llama.cpp GGUF fallback is a *stretch* hedge only if time permits (the stack is vLLM-native).
- Keep `submission.csv` byte-format identical to the run that scored 79.91.

## Deliberately NOT doing (see note_v3 §9)
RAG / embeddings / reranker (illegal + measured to hurt) · second model (illegal + doesn't help our real gap) · tool/code-execution reasoning (findings: −16.5) · fine-tuning (risks degrading thinking mode; last-resort, no-think path only, only if ahead of schedule).
