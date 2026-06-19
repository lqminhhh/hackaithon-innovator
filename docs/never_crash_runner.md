# DESIGN NOTE — Building the Never-Crash Runner (S7)

<!-- Scope: the run loop that turns an input JSON into a COMPLETE submission.csv no matter what
     happens — crashes, kills, bad questions, OOM. This is the safety guarantee, not an optimization.
     Read planning.md S0 and S4 first. -->

---

## 0. Purpose and where this fits

The **never-crash runner** is the outermost layer. It reads the questions, drives them through the
escalation ladder, and **always writes a complete `submission.csv`** — on success, on a single bad
question, on a kill signal, on an exception, on out-of-memory. In a one-shot-graded container there
are no retries, so this guarantee is the whole point.

Restating the project's first invariant in operational terms:

> **A crash is a score of 0.** Every input `qid` must end up in the CSV with a valid letter, even if
> the run dies halfway. Robustness beats cleverness here.

**Prerequisites:**

| Needs | From | Why |
|---|---|---|
| `load_questions(path)`, `write_submission(rows, path)`, `letters(n)`, `FALLBACK` | S0 (`io_utils.py`, config) | I/O contract; UTF-8; the safe default |
| `solve_question(q, route, domain)` + batchable stages `first_pass` / `run_action` | S4 (the ladder note) | the per-question work the runner schedules |
| `route_l1` / `route` + `domain` | S3 / S5 | needed to group questions for batching |

**Relationship to an existing `pipeline.py`:** if the project already has an orchestrator loop,
`run.py` (this note) either **replaces** it or **wraps** it. It almost certainly adds requirements the
existing loop lacks — checkpoint/resume, atomic write, always-emit on signal — so treat those as the
deliverable even if a loop already exists. Decide explicitly; don't leave two loops fighting.

---

## 1. The contract

```
run(input_path: str, output_path: str) -> None
CLI:  python run.py --input <questions.json> --output submission.csv
```

Produces `submission.csv` with **exactly one row per input qid** (`qid,answer`), UTF-8, valid letters,
**resumable** and **crash-proof**. Returns nothing; its only effect is a complete CSV on disk.

---

## 2. What the runner must guarantee

Four hard guarantees, plus throughput. Tests in §12 map one-to-one to these.

- **G1 — Completeness.** Every input `qid` has a row, even on a partial or failed run.
- **G2 — Fault isolation.** One bad question cannot break the others; it gets `FALLBACK`.
- **G3 — Durability.** Progress survives a kill/crash via checkpoint + resume.
- **G4 — Always-emit.** A complete CSV is written at normal end **and** on signal/exception.
- **Throughput.** Order/batch questions to maximise vLLM utilisation **without breaking qid mapping**
  (this is where the 10% speed score lives — but never at the cost of G1–G4).

---

## 3. Principles (treat as invariants)

1. **`qid` is the only key.** Never map answers to questions by list position. All state is
   `{qid: answer}`. Batching reorders things — position-based mapping is how you silently ship wrong
   answers with no crash to warn you.
2. **Pre-fill `FALLBACK` for every qid before any work starts.** A crash at question 1 then still
   yields a complete CSV of fallbacks (G1).
3. **Writes are atomic** — temp file + `os.replace`, so a kill mid-write can't corrupt the CSV.
4. **The checkpoint is the source of truth for resume**, and writing the CSV is **idempotent**
   (writing twice is fine; last write wins).
5. **Batch failures degrade to per-question, never to a crash** (G2).
6. **UTF-8 everywhere; preserve Vietnamese diacritics.**

---

## 4. State model

```python
answers: dict[str, str]   # {qid: letter}, PRE-FILLED with FALLBACK for every input qid
status:  dict[str, str]   # {qid: "todo" | "done" | "failed"}  (drives resume; optional but recommended)
```

- **Init:** `answers = {q["qid"]: FALLBACK for q in questions}`, `status = {qid: "todo"}`.
- **Checkpoint** (`checkpoint.json`): persist `answers` + `status` every `checkpoint_every`
  questions, with `flush()` + `os.fsync()` for durability.
- **Resume on startup:** if a checkpoint exists, load it; process only qids whose status is not
  `"done"`. Re-attempt `"failed"` qids once (the failure may have been a transient crash), but never
  loop on them.
- **CSV is always derivable from `answers`** at any moment — that's what makes always-emit trivial.

---

## 5. Execution strategy — two modes

Both modes are keyed by `qid`, both checkpoint, both always-emit. Start with the MVP; move to batched
only once correctness is locked.

### 5a. MVP — sequential (simplest; satisfies G1–G4)

```python
def run(input_path, output_path):
    questions = load_questions(input_path)
    answers = {q["qid"]: FALLBACK for q in questions}
    status  = {q["qid"]: "todo"  for q in questions}
    load_checkpoint(answers, status)            # resume if present
    install_always_emit(answers, output_path)   # atexit + signal + finally (see §6)

    try:
        for i, q in enumerate(questions):
            if status[q["qid"]] == "done":
                continue
            try:
                r, d = route(q), domain(q)
                answers[q["qid"]] = solve_question(q, r, d)
                status[q["qid"]]  = "done"
            except Exception as e:
                log_error(q["qid"], e)          # answers[qid] stays FALLBACK
                status[q["qid"]] = "failed"      # (retry once on a later resume)
            if i % CHECKPOINT_EVERY == 0:
                checkpoint(answers, status)
    finally:
        write_submission_atomic(answers, output_path)   # G4
```

This alone passes every robustness accept test. It does not batch, so it's slow — acceptable for a
first correct version.

### 5b. Optimised — phased batch (for the speed score)

The escalation ladder exposes batchable stages exactly so the runner can do this. Process in phases,
isolating faults at the **group** level and degrading to per-question on a batch error:

1. **Route all questions** (cheap) and **group by `(route, thinking-mode)`** for KV-cache-friendly
   batches. Also fetch step-0 context for forced-domain qids.
2. **Phase A — batched first pass** per group → `{qid: (letter, logprob)}`. If a group's `generate`
   raises (e.g. OOM), retry that group **per-question**; any still-failing qid stays `FALLBACK`.
3. **Phase B — gate per qid** (pure, cheap) → `{qid: decision}`.
4. **Phase C — group qids by decision** and batch each action: all `ESCALATE_THINK_VOTE` together
   (each via `n=SC_N`), all `ESCALATE_RETRIEVE` together, etc. Same per-group → per-question fallback
   on error.
5. **Assemble** `answers[qid]`, **checkpoint after each phase**, then `write_submission_atomic`.

Keep a single dict keyed by `qid` throughout; results from any batched call are re-keyed to `qid`
**immediately** (zip with the qids passed *into that same call*, never with a global ordering).

**Recommendation:** ship 5a, lock correctness with the §12 tests, then add 5b behind a
`runner.batch: true` flag. The qid-keying discipline is what makes 5b safe.

---

## 6. Always-emit wiring (G4)

Write the CSV from **three** places; all idempotent:

- **`finally:`** around the main loop (normal end and uncaught exception).
- **`atexit.register(lambda: write_submission_atomic(answers, output_path))`**.
- **Signal handlers** for `SIGTERM`/`SIGINT`: write, then exit. Keep the handler **minimal**
  (CUDA/vLLM state is fragile mid-signal) — write the CSV and call `os._exit(0)` rather than running
  more Python. Setting a flag the loop checks is an alternative, but a direct write is safer against a
  hard kill.

Because every write derives from the same `answers` dict and is atomic, writing more than once is
harmless.

---

## 7. Atomic write

```python
def write_submission_atomic(answers, path):
    tmp = path + ".tmp"
    write_submission(rows_from(answers), tmp)   # S0 writer, UTF-8, one row per qid
    os.replace(tmp, path)                       # atomic on the same filesystem
```

A process killed mid-write leaves the **previous** good CSV intact; the partial write lands only in
`.tmp`.

---

## 8. Checkpoint format and cadence

- Format: a single `checkpoint.json` = `{"answers": {...}, "status": {...}}`, rewritten atomically
  (same temp+replace trick), or an append-only JSONL of `{qid, answer, status}` if you prefer.
- Cadence: every `checkpoint_every` questions (config), plus once after each phase in 5b.
- Durability: `flush()` + `os.fsync()` so a power/kill loss keeps the last checkpoint.
- Corrupt checkpoint on load → **ignore it and start fresh**, log a warning; never crash on a bad
  checkpoint.

---

## 9. Config (`pipeline_config.yaml`)

```yaml
runner:
  checkpoint_path: checkpoint.json
  checkpoint_every: 25
  batch: false          # false = sequential MVP (5a); true = phased batch (5b)
  group_by_route: true  # KV-cache-friendly batching in 5b
# `fallback: "A"` already exists in config (reused as FALLBACK).
```

---

## 10. Edge cases and failure-safety

- **Empty input** → write a header-only CSV, no crash.
- **Duplicate qids in input** → keep one row per unique qid (last wins); log the duplicates.
- **Malformed question** (missing `choices`, etc.) → `FALLBACK`, log; never abort the run.
- **OOM / CUDA error mid-batch (5b)** → catch at the group level, retry that group per-question with a
  smaller effective batch; a single question that still OOMs gets `FALLBACK`.
- **Output directory missing** → create it before writing.
- **Killed mid-write** → atomic rename protects the prior good CSV (§7).
- **Corrupt checkpoint** → start fresh (§8).
- **`solve_question` hangs** (e.g. a stalled retrieval) → rely on S6's retrieval time-box; consider a
  per-question soft timeout in 5a so one stuck question can't stall the whole run.

---

## 11. Phased build order

- **v1:** sequential loop (5a) + `FALLBACK` pre-fill + per-question `try/except` + atomic write +
  `finally`/`atexit`. **Satisfies G1, G2, G4.**
- **v2:** checkpoint + resume + signal handlers. **Adds G3** and completes always-emit.
- **v3:** phased-batch execution (5b) behind `runner.batch: true`, with group→per-question fallback.
  **Adds throughput** for the speed score.

Get G1/G2/G4 before anything else — a correct slow runner beats a fast one that can zero the score.

---

## 12. Accept tests

1. **Resume (G3):** kill the process mid-run → restart → it resumes from the checkpoint and produces a
   complete CSV with the already-solved answers intact.
2. **Fault isolation (G2):** force `solve_question` to raise on exactly one qid → that qid gets
   `FALLBACK`, every other qid is correct, no crash, complete CSV.
3. **Atomic write:** kill the process during the CSV write → the previous good CSV is intact and
   uncorrupted.
4. **qid integrity (the misalignment risk):** run on the input, then run on a **shuffled** copy of the
   input → every qid maps to the **same** answer in both runs. (This is the test that catches
   position-based mapping bugs in 5b.)
5. **Completeness (G1):** kill before any question finishes → CSV still has one `FALLBACK` row per
   input qid.
6. **Signal (G4):** send `SIGTERM` → a complete CSV is written before the process exits.
7. **Empty input** → header-only CSV, no crash.
8. **Smoke:** the S0 5-question sample still runs end-to-end and emits 5 valid rows, UTF-8.

---

## 13. Metrics to report

- **Wall-clock total** and **throughput** (questions/sec) — the speed-score inputs; compare 5a vs 5b.
- **Checkpoint overhead** (time spent writing checkpoints).
- **Fallback count** — should be ~0 on a clean run. **Any `FALLBACK` in a normal run is a real error to
  investigate**, not noise; surface it.

> The runner's job is boring on purpose: make it impossible to score 0. Optimise throughput only after
> every robustness test in §12 is green.