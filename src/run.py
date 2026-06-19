"""Never-crash runner + checkpoint

The outermost layer of the agent. It reads an input file, drives every question
through the solve policy, and **always writes a complete ``submission.csv``** —
on success, on a single bad question, on a kill signal, on an exception, on OOM.

Guarantees:
    G1 Completeness    — every input qid has a row, even on a partial/failed run.
    G2 Fault isolation — one bad question gets FALLBACK; the rest are unaffected.
    G3 Durability      — progress survives a kill/crash via checkpoint + resume.
    G4 Always-emit     — a complete CSV is written at normal end AND on signal/exception.

This module implements the sequential MVP plus checkpoint/resume and signal handlers.
Phased-batch execution is a later throughput optimization and is intentionally not wired here;
``--batch`` is accepted but currently runs sequentially with a warning so the safety guarantees
are never at risk.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
import traceback
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import FALLBACK
from src.data_loader import load_questions, write_submission

# Default runner settings (overridable via CLI / config)
DEFAULT_CHECKPOINT_EVERY = 30
DEFAULT_CHECKPOINT_NAME = "checkpoint.json"

SolveFn = Callable[[dict], Any]


@dataclass(slots=True)
class RunnerState:
    """All mutable run state, keyed exclusively by ``qid``.

    ``answers`` is pre-filled with FALLBACK for every input qid so that a crash
    before any work still yields a complete CSV (G1). ``status`` drives resume.
    """

    answers: dict[str, str]
    status: dict[str, str]  # qid -> "todo" | "done" | "failed"
    output_path: str
    checkpoint_path: str | None = None
    _emitted: bool = field(default=False, repr=False)

    def rows(self) -> list[dict[str, str]]:
        return [{"qid": qid, "answer": answer} for qid, answer in self.answers.items()]


# ─────────────────────────────────────────────────────────────────────────────
# Atomic write + checkpoint 
# ─────────────────────────────────────────────────────────────────────────────

def write_submission_atomic(state: RunnerState) -> None:
    """Write the CSV via temp-file + ``os.replace`` so a kill mid-write is safe.

    A process killed mid-write leaves the previous good CSV intact; the partial
    write only ever lands in the ``.tmp`` file.
    """
    path = Path(state.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_submission(state.rows(), tmp)
    os.replace(tmp, path)
    state._emitted = True


def write_checkpoint(state: RunnerState) -> None:
    """Persist answers + status atomically with fsync for durability (G3)."""
    if not state.checkpoint_path:
        return
    payload = {"answers": state.answers, "status": state.status}
    ckpt = Path(state.checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_name(ckpt.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ckpt)


def _load_checkpoint(checkpoint_path: str) -> tuple[dict[str, str], dict[str, str]] | None:
    """Load a checkpoint, or return None if missing/corrupt (never crash, §8)."""
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"[runner] ignoring corrupt checkpoint {checkpoint_path}: {exc}", flush=True)
        return None
    if not isinstance(data, dict):
        print(f"[runner] ignoring malformed checkpoint {checkpoint_path}", flush=True)
        return None
    answers = data.get("answers", {})
    status = data.get("status", {})
    if not isinstance(answers, dict) or not isinstance(status, dict):
        print(f"[runner] ignoring malformed checkpoint {checkpoint_path}", flush=True)
        return None
    return answers, status


def _resume_from_checkpoint(checkpoint_path: str, state: RunnerState) -> int:
    """Adopt previously-solved answers for qids present in the current input.

    Only ``done`` qids are trusted; ``failed``/``todo`` qids are re-attempted on
    this run. Returns the number of resumed (done) qids.
    """
    loaded = _load_checkpoint(checkpoint_path)
    if loaded is None:
        return 0
    ck_answers, ck_status = loaded
    resumed = 0
    for qid in state.answers:
        if ck_status.get(qid) == "done":
            state.answers[qid] = ck_answers.get(qid, state.answers[qid])
            state.status[qid] = "done"
            resumed += 1
    return resumed


# ─────────────────────────────────────────────────────────────────────────────
# Always-emit wiring (§6, G4)
# ─────────────────────────────────────────────────────────────────────────────

def _install_always_emit(state: RunnerState) -> Callable[[], None]:
    """Register atexit + SIGTERM/SIGINT handlers that emit a complete CSV.

    Returns a cleanup callable that removes the handlers again (used so repeated
    in-process ``run()`` calls and tests don't leak global state).
    """

    def _safe_emit() -> None:
        try:
            write_submission_atomic(state)
        except Exception as exc:  # never let the safety net itself crash
            print(f"[runner] always-emit failed: {exc}", flush=True)

    def _signal_handler(signum, _frame):
        # Keep this minimal: vLLM/CUDA state is fragile mid-signal. Write the CSV
        # from the already-populated dict, then hard-exit without more Python.
        _safe_emit()
        os._exit(0)

    atexit.register(_safe_emit)
    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Not on the main thread (e.g. inside some test harnesses) — skip.
            pass

    def _cleanup() -> None:
        atexit.unregister(_safe_emit)
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, TypeError):
                pass

    return _cleanup


# ─────────────────────────────────────────────────────────────────────────────
# Input prep
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_questions(questions: list[dict]) -> list[dict]:
    """Keep one entry per unique qid (last wins); log duplicates (§10)."""
    by_qid: "OrderedDict[str, dict]" = OrderedDict()
    duplicates: list[str] = []
    for q in questions:
        qid = str(q.get("qid", ""))
        if qid in by_qid:
            duplicates.append(qid)
        by_qid[qid] = q
    if duplicates:
        print(
            f"[runner] dropped {len(duplicates)} duplicate qid(s) (last wins): "
            f"{sorted(set(duplicates))}",
            flush=True,
        )
    return list(by_qid.values())


def _extract_answer(result: Any) -> str:
    """Normalise a solve result (a letter string or an object with ``.answer``)."""
    if isinstance(result, str):
        return result
    answer = getattr(result, "answer", None)
    if isinstance(answer, str) and answer:
        return answer
    raise ValueError(f"solve_fn returned an unusable result: {result!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Default solver (real model path) — built lazily so tests can inject a fake
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_solve_fn(model_id: str | None = None) -> SolveFn:
    """Load the primary model once and return a per-question solve closure.

    Uses the v3 path: rule router + two-pass guided choice + escalation, no RAG
    and no semantic router (both removed in v3).
    """
    import torch  # heavy import; only needed for the real run

    from src.models import load_primary_model, load_vllm_primary
    from src.parser import parse_question
    from src.reasoning_agent import ReasoningAgent
    from src.solve import solve_question

    chosen_model = model_id or "Qwen/Qwen3.5-4B"
    t0 = time.time()
    if torch.cuda.is_available():
        try:
            print(f"[runner] loading vLLM model: {chosen_model}", flush=True)
            agent = ReasoningAgent(llm=load_vllm_primary(model_id=chosen_model))
        except Exception as exc:
            print(f"[runner] vLLM unavailable ({exc}); falling back to HuggingFace", flush=True)
            model, tokenizer = load_primary_model(model_id=chosen_model)
            agent = ReasoningAgent(model=model, tokenizer=tokenizer)
    else:
        print(f"[runner] loading HuggingFace model: {chosen_model}", flush=True)
        model, tokenizer = load_primary_model(model_id=chosen_model)
        agent = ReasoningAgent(model=model, tokenizer=tokenizer)
    print(f"[runner] model ready in {time.time() - t0:.1f}s", flush=True)

    def solve_fn(question: dict) -> Any:
        parsed = parse_question(question)
        return solve_question(agent, parsed)

    return solve_fn


# ─────────────────────────────────────────────────────────────────────────────
# The runner (5a sequential MVP + resume + always-emit)
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_path: str,
    *,
    solve_fn: SolveFn | None = None,
    model_id: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    resume: bool = True,
    limit: int | None = None,
    install_handlers: bool = True,
    batch: bool = False,
) -> None:
    """Drive ``input_path`` to a complete ``submission.csv`` that can never crash.

    Parameters
    ----------
    solve_fn:
        Optional ``question_dict -> answer`` callable. When omitted, the primary
        model is loaded and the v3 solve policy is used. Injecting a fake here is
        how the accept tests run without a GPU.
    checkpoint_path:
        Where to persist progress. Defaults to ``<output_dir>/checkpoint.json``.
    install_handlers:
        Install atexit + signal always-emit handlers (G4). Disable in in-process
        unit tests so global handlers don't leak between cases.
    batch:
        Accepted for forward-compatibility; phased-batch (5b) is not yet wired,
        so a truthy value logs a warning and runs sequentially.
    """
    if batch:
        print(
            "[runner] --batch requested but phased-batch (5b) is not implemented; "
            "running sequentially (5a).",
            flush=True,
        )

    questions = _dedup_questions(load_questions(input_path))
    if limit is not None:
        questions = questions[:limit]

    if checkpoint_path is None:
        checkpoint_path = str(Path(output_path).parent / DEFAULT_CHECKPOINT_NAME)

    # G1: pre-fill FALLBACK for every qid before any work begins.
    state = RunnerState(
        answers={q["qid"]: FALLBACK for q in questions},
        status={q["qid"]: "todo" for q in questions},
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    if not questions:
        print("[runner] empty input — writing header-only CSV.", flush=True)
        write_submission_atomic(state)
        return

    resumed = _resume_from_checkpoint(checkpoint_path, state) if resume else 0
    if resumed:
        print(f"[runner] resumed {resumed}/{len(questions)} answers from checkpoint.", flush=True)

    cleanup = _install_always_emit(state) if install_handlers else (lambda: None)

    if solve_fn is None:
        solve_fn = _build_default_solve_fn(model_id=model_id)

    route_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    failed: list[str] = []
    run_start = time.time()
    total = len(questions)

    try:
        for i, q in enumerate(questions):
            qid = q["qid"]
            if state.status.get(qid) == "done":
                continue

            q_start = time.time()
            try:
                result = solve_fn(q)
                state.answers[qid] = _extract_answer(result)
                state.status[qid] = "done"
                route_counts[str(getattr(result, "route", "?"))] += 1
                path_counts[str(getattr(result, "path", "?"))] += 1
            except Exception as exc:
                # G2: fault isolation. The qid keeps its pre-filled FALLBACK.
                state.status[qid] = "failed"
                failed.append(qid)
                print(f"[runner] qid={qid} failed -> FALLBACK: {exc}", flush=True)
                traceback.print_exc()

            if (i + 1) % checkpoint_every == 0:
                write_checkpoint(state)

            avg = (time.time() - run_start) / (i + 1)
            eta = avg * (total - i - 1)
            print(
                f"[runner] [{i + 1}/{total}] {qid} "
                f"answer={state.answers[qid]} status={state.status[qid]} "
                f"({time.time() - q_start:.2f}s, avg {avg:.2f}s/q, ETA {eta / 60:.1f}min)",
                flush=True,
            )
    finally:
        # G4: always-emit on normal end and on any uncaught exception.
        write_checkpoint(state)
        write_submission_atomic(state)
        cleanup()

    done = sum(1 for s in state.status.values() if s == "done")
    elapsed = time.time() - run_start
    print(f"[runner] wrote {len(state.answers)} answers to {output_path}", flush=True)
    print(f"[runner] solved={done} failed={len(failed)} (FALLBACK count={len(failed)})", flush=True)
    if failed:
        # §13: any FALLBACK on a normal run is a real error to investigate.
        print(f"[runner] FAILED qids (investigate): {failed}", flush=True)
    if route_counts:
        print(f"[runner] route counts: {dict(route_counts)}", flush=True)
        print(f"[runner] path counts: {dict(path_counts)}", flush=True)
    print(
        f"[runner] total {elapsed:.1f}s "
        f"({elapsed / max(total, 1):.2f}s/q, {total / max(elapsed, 1e-9):.2f} q/s)",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="S7 never-crash runner (v3 single-model path)")
    parser.add_argument("--input", required=True, help="Path to input questions (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    parser.add_argument("--model-id", default=None, help="Optional HF model override")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: <output_dir>/checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint cadence in questions",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore any existing checkpoint and start fresh",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions to process",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Reserved for phased-batch (5b); currently runs sequentially",
    )
    args = parser.parse_args()

    run(
        args.input,
        args.output,
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        limit=args.limit,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
