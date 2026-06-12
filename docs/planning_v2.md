# BUILD SPEC — Vietnamese MCQ QA Agent (for a coding AI)

<!-- This file is the build contract. Terse on purpose. Companion: note.md (human rationale). -->

## ▼ READ THIS FIRST (load before any segment)

**What:** Offline Docker agent. Input JSON of Vietnamese multiple-choice questions →
output `submission.csv` (`qid,answer`). Answer is a single letter.

**One-line architecture:** one LLM (Qwen3-8B-AWQ on vLLM) + a router that decides *how much
compute* each question gets + an escalation ladder that decides *if the answer is good enough* +
guided-choice decoding that *cannot* emit an invalid answer + reranker-gated RAG used only on
recall questions, all inside a never-crash runner.

**INVARIANTS (never violate):**
1. **Always write a complete `submission.csv`** for every qid. Any per-question exception →
   fallback letter `"A"` (or best prior stage). A crash = score 0.
2. **UTF-8 everywhere** (read/write). Preserve Vietnamese diacritics.
3. **Each model < 9B params.** AWQ 4-bit by default (judge GPU unknown). Target **≤20 GB VRAM**.
4. **No internet at inference.** All models/corpus baked into the image.
5. **No regex answer parsing.** Answers come from vLLM `guided_choice`.
6. **Router = optimization. Escalation = correctness.** A misroute must only cost speed, never
   force a wrong/blank answer. Default route when unsure = `KNOWLEDGE` (full path), never the cheap one.

**Data contract:**
```
question = {"qid": "test_0001", "question": str, "choices": [str, ...]}   # 2..10 choices
output   = {"qid": "test_0001", "answer": "A"}                            # one valid letter
```

**Build order (each segment is self-contained; build + verify one before the next):**

| Seg | Name | Priority | Gives you |
|---|---|---|---|
| S0 | Skeleton + config + I/O | **MUST** | runnable shell that emits a valid CSV |
| S1 | vLLM + Qwen3-8B-AWQ load | **MUST** | model serving, think/no_think |
| S2 | Guided-choice + logprob margin | **MUST** | unbreakable answer extraction |
| S3 | Layer-1 rule router + prompts | **MUST** | routing + refusal-trap handling |
| S4 | Compute budgets + escalation + self-consistency | **MUST** | accuracy net |
| S7 | Never-crash runner + checkpoint | **MUST** | the safety guarantee |
| S5 | Layer-2 semantic router (offline centroids) | SHOULD | better RAG gating, paraphrase-robust |
| S6 | RAG (BGE-m3 + reranker + corpus) | SHOULD | the recall slice |
| S8 | Dev set + ablation harness | SHOULD | tuning + creativity-score evidence |

**MINIMAL VIABLE SUBMISSION = S0+S1+S2+S3+S4+S7.** It scores without RAG/semantic. Ship that
first, then add S5/S6/S8. Do not start S5/S6 until the MVP passes its accept tests.

**Central config (one module, `config.py`):**
```python
LLM_MODEL   = "Qwen/Qwen3-8B-AWQ"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL= "BAAI/bge-reranker-v2-m3"        # or "Qwen/Qwen3-Reranker-0.6B"
GPU_MEM_UTIL= 0.85
MARGIN_LOW  = 0.15        # prob(top1) - prob(top2) below this -> escalate
SC_N        = 5           # self-consistency samples for STEM/escalation
SC_TEMP     = 0.6
RERANK_MIN  = 0.5         # inject RAG context only if top rerank score >= this
FORCE_RETRIEVE_DOMAINS = {"vn_law","vn_decree","vn_admin","local_facts"}  # always retrieve here
TOK = {"READING":512, "STEM":3072, "KNOWLEDGE":256, "SAFETY":128}
FALLBACK    = "A"
```

---

## S0 — Skeleton + config + I/O   [MUST]   Depends: none

**Goal:** runnable program that loads questions and writes a valid (dummy) CSV.
**Deliver:** `config.py`, `io_utils.py`, `main.py`.
**Spec:**
- `load_questions(path) -> list[dict]`: open with `encoding="utf-8"`. Optionally run `ftfy.fix_text`
  on each string (guard: only if it changes nothing on clean input).
- `letters(n) -> list[str]`: `["A","B",...]` length n (supports up to 10 = "J").
- `write_submission(rows, path)`: write `qid,answer` UTF-8, one row per input qid, **even on partial runs**.
- `main.py`: load → (for now) answer `FALLBACK` for all → write CSV.
**Accept:** run on a 5-question sample → CSV with 5 rows, valid letters, UTF-8, no crash.

---

## S1 — vLLM + Qwen3-8B-AWQ   [MUST]   Depends: S0

**Goal:** one vLLM engine serving Qwen3-8B-AWQ with think/no_think control.
**Deliver:** `llm.py` (`class LLM`).
**Spec:**
- Init vLLM `LLM(model=LLM_MODEL, quantization="awq", gpu_memory_utilization=GPU_MEM_UTIL,
  max_model_len=…, enable_prefix_caching=True)`.
- `mode` switch: thinking via Qwen3 chat template (`enable_thinking=True/False`) or `/think`,`/no_think`.
- `generate(prompts: list[str], sampling) -> list[output]` — always **batched** (pass the whole list).
- Expose logprobs in `SamplingParams(logprobs=…)`.
**Accept:** batch of 3 prompts returns 3 completions; `/no_think` returns short output, `/think` emits
a reasoning block; VRAM resident < 20 GB on the test card.

---

## S2 — Guided-choice + logprob margin   [MUST]   Depends: S1

**Goal:** extraction that can only ever return a valid letter, plus a confidence number.
**Deliver:** `extract.py`.
**Spec:**
- Build prompt ending in `"Đáp án: "`. Constrain with `guided_choice=letters(n)` (vLLM
  guided decoding / lm-format-enforcer). Works for 4 and 10 choices automatically.
- Request logprobs for the choice token. Return `(letter, margin, per_letter_logprob)` where
  `margin = prob(top1) - prob(top2)` (softmax over the constrained letters).
**Accept:** on a question with 10 choices, returns exactly one of A–J + a float margin in [0,1].
Never returns anything outside the letter set.

---

## S3 — Layer-1 rule router + prompts   [MUST]   Depends: S0

**Goal:** classify each question into READING / STEM / SAFETY, else None (→ KNOWLEDGE default).
**Deliver:** `router.py` (`route_l1`), `prompts.py`.
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
    return None    # -> KNOWLEDGE default (or S5 semantic)
```
- `STEM_KW = {"tính","bao nhiêu","phương trình","đạo hàm","tích phân","xác suất","vận tốc",
  "điện trở","nồng độ","co giãn","lãi suất","ma trận","giá trị"}`. `HARMFUL_KW`: explicit
  illegal/harmful-intent phrasing.
- Prompts per route in `prompts.py`. **All routes include the refusal-trap system line:**
  > Đây là bài thi trắc nghiệm. Chọn phương án đúng nhất. Chỉ chọn phương án "không thể trả lời /
  > từ chối" khi câu hỏi thực sự yêu cầu hướng dẫn hành vi phạm pháp hoặc gây hại. Với mọi câu hỏi
  > học thuật, khoa học, kiến thức thông thường, luôn chọn một phương án nội dung.
  - READING prompt: "Chỉ dựa vào đoạn văn trong câu hỏi để trả lời." (answer only from passage)
  - STEM prompt: "Giải từng bước rồi chọn đáp án." (think then choose)
**Accept:** on labelled samples, READING fires on context-bearing items, STEM on 10-choice/LaTeX items,
None on plain knowledge items. Misclassification is acceptable (escalation covers it) but READING must
not fire on STEM.

---

## S4 — Compute budgets + escalation + self-consistency   [MUST]   Depends: S1,S2,S3

**Goal:** per-route compute + the correctness net.
**Deliver:** `solve.py` (`solve_question`).
**Per-route policy:**
| Route | mode | max_tok | self-consistency | RAG |
|---|---|---|---|---|
| READING | no_think | 512 | no | off |
| STEM | think | 3072 | n=SC_N, temp SC_TEMP, majority vote | off |
| KNOWLEDGE | no_think | 256 | no (escalate if low margin) | S6 if recall |
| SAFETY | no_think | 128 | no | off |
**Escalation ladder:**
```
0. if domain in FORCE_RETRIEVE_DOMAINS -> fetch S6 context, include it in the first pass
1. single guided-choice pass -> (letter, margin)
2. margin >= MARGIN_LOW -> accept
3. margin <  MARGIN_LOW:
     STEM/other -> /think + self-consistency(SC_N) -> majority-vote letter
     KNOWLEDGE  -> fetch S6 context (if not already) -> second pass WITH that context
4. tie -> highest-logprob letter. NEVER blank.
```
- Self-consistency: sample SC_N completions (temp SC_TEMP), each ends in guided-choice, take the
  mode of the letters. Share the prompt so vLLM reuses KV cache.
**Accept:** STEM items run the n=SC_N vote; low-margin knowledge items trigger a second pass;
every question yields exactly one letter.

---

## S7 — Never-crash runner + checkpoint   [MUST]   Depends: S0,S4

**Goal:** the robustness guarantee.
**Deliver:** `run.py`.
**Spec:**
- For each question: `try: solve_question(...) except Exception: log + answer = best-so-far or FALLBACK`.
- **Checkpoint** the answer dict to disk every N questions; on startup, resume from checkpoint.
- Always write the full `submission.csv` at the end **and** on signal/exception (atexit/finally).
- Order questions to maximize vLLM batching (e.g., group by route/mode) but keep qid mapping intact.
**Accept:** kill the process mid-run → restart → it resumes and still produces a complete CSV.
Inject a forced exception on one question → that qid gets FALLBACK, all others correct, no crash.

---

## S5 — Layer-2 semantic router (offline centroids)   [SHOULD]   Depends: S3

**Goal:** classify the questions S3 abstained on, paraphrase-robustly; drive RAG gating.
**Deliver:** `semantic_router.py` + build script `build_centroids.py` + artifact `centroids.npz`.
**Spec:**
- **Build time (offline):** label a reference set by domain (calc / recall / concept + subject),
  embed with BGE-m3, store per-domain mean vectors → `centroids.npz`. Commit the artifact.
- **Inference:** embed question (BGE-m3), cosine to centroids, nearest = domain.
  - If best cosine < floor → return `KNOWLEDGE` default (no forced domain).
  - Output `(route, domain)`; `domain` decides RAG corpus + whether it's recall (→ S6) vs concept.
- Deterministic, no LLM call, microseconds.
**Accept:** abstained items get a domain or safe default; rewording a question keeps its domain stable.

---

## S6 — RAG (BGE-m3 + reranker + corpus)   [SHOULD]   Depends: S5

**Goal:** retrieval, fired **only** on KNOWLEDGE recall questions.
**Deliver:** `rag.py` + offline `build_index.py` + corpus artifacts.
**Spec:**
- **Corpus (general, not cherry-picked):** VN legal codes (Hiến pháp, Dân sự, Hình sự, Lao động),
  civics/HCM-tư-tưởng, Vietnamese Wikipedia. Build a BGE-m3 dense (+ optional sparse) index offline.
- **Retrieve:** top-20. **Rerank:** RERANK_MODEL. **Inject:** top-3 only if score ≥ RERANK_MIN.
- **When to retrieve (3 filters, cheapest first — implemented in S4):**
  1. *Route exclude:* READING/STEM/SAFETY -> never retrieve.
  2. *Domain force:* domain in FORCE_RETRIEVE_DOMAINS -> always retrieve (model's confidence is
     untrustworthy here), even on the first pass.
  3. *Confidence gate:* otherwise answer first with NO retrieval; retrieve + re-answer only when
     margin < MARGIN_LOW. High margin -> trust the model, skip retrieval (saves time, avoids distraction).
- Keep `exclude_qid` decontamination.
- **Time-box** retrieval so it can't stall the run.
**Accept:** recall question retrieves relevant passages; concept/STEM questions never trigger retrieval;
empty/low-score retrieval falls back to no-context generation (never errors).

---

## S8 — Dev set + ablation harness   [SHOULD]   Depends: S4 (S5/S6 optional)

**Goal:** measure accuracy and produce the creativity-score evidence.
**Deliver:** `dev_set.jsonl` (labelled), `eval.py`, `ablation.md` (results).
**Spec:**
- Build ~150–200 labelled questions: hand-label STEM (verifiable) + reading-comp; strong-model-label
  the rest + spot-check. Cover every bucket (≥25–40 each).
- **Split: tune on half, report on the other half.** Drop any gain that doesn't survive the split.
- Ablation rows: `base → +router(no-RAG on context) → +guided-choice → +STEM self-consistency → +RAG`.
- Also validate on an **external** set (VMLU / ViMMRC / old THPT) to confirm it generalizes.
**Accept:** per-bucket accuracy table + ablation table produced; thresholds (MARGIN_LOW, RERANK_MIN,
SC_N) chosen on the tune half, reported on the held-out half.

---

## Dockerfile / packaging   [MUST before submit]
- Base CUDA image; install vLLM + deps; **download all models + corpus at build time** (offline at run).
- Entry: `python run.py --input <json> --output submission.csv`.
- Verify the image runs with no network and emits a complete CSV.

## Open items to confirm (don't block MVP)
- vLLM `guided_choice` supports up to 10 letters in the pinned version.
- Qwen3-8B-AWQ thinking-mode template flag name in your vLLM version.
- Real input encoding is UTF-8 (the public paste looked mojibake).
