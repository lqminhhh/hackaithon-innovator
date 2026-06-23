"""Shared runners for v01_baseline and v02_beta. One primary LLM, deterministic routing, same-model SC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FALLBACK,
    GPU_MEM_UTIL,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    SAFE_GPU_MEM_UTIL,
    SAFE_MAX_MODEL_LEN,
    SAFE_MAX_NUM_SEQS,
)
from src.extract import ChoiceResult
from src.models import load_primary_model, load_vllm_primary
from src.normaliser import normalise_answer
from src.parser import ParsedQuestion, parse_question
from src.reasoning_agent import ReasoningAgent
from src.router import Route, get_forced_answer, route_question
from src.solve import SolveResult, solve_question

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "public-test_1780368312.json"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: str,
    default_trace: str,
    include_sc_batch: bool = False,
) -> None:
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to input file (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--trace-output",
        default=default_trace,
        help="Path to per-question JSONL trace output",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model override. Defaults to project config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions to process",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=False,
        help="Use conservative vLLM settings for a 16GB VRAM machine.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM max_num_seqs.",
    )
    if include_sc_batch:
        parser.add_argument(
            "--sc-batch-size",
            type=int,
            default=None,
            help="Maximum self-consistency generation batch size.",
        )


def run_v01_baseline(
    *,
    input_path: str,
    output_path: str,
    trace_output: str,
    model_id: str | None = None,
    limit: int | None = None,
    safe_mode: bool = False,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
) -> None:
    """Run the LLM-only baseline with trace logging."""
    t_start = time.time()
    agent = _load_agent(
        model_id=model_id,
        safe_mode=safe_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        t_start=t_start,
    )
    questions = _load_limited_questions(input_path, limit)
    print(f"Processing {len(questions)} questions (v01_baseline)...", flush=True)

    results: list[dict[str, str]] = []
    run_start = time.time()
    with _trace_writer(trace_output) as write_trace:
        for i, question in enumerate(questions):
            q_start = time.time()
            raw = agent.infer_no_context(question["question"], question["options"])
            valid_labels = tuple(sorted(question["options"].keys()))
            answer = normalise_answer(raw, valid_labels)
            elapsed = time.time() - q_start

            results.append({"qid": question["qid"], "answer": answer})
            write_trace(
                _baseline_trace(
                    qid=question["qid"],
                    answer=answer,
                    raw_output=raw,
                    elapsed_seconds=elapsed,
                )
            )
            _print_progress(
                i=i,
                total=len(questions),
                qid=question["qid"],
                answer=answer,
                route=None,
                path="llm_only",
                margin=None,
                votes=[],
                error=None,
                q_elapsed=elapsed,
                run_start=run_start,
            )

    _finish_run(
        results=results,
        output_path=output_path,
        t_start=t_start,
        run_start=run_start,
        route_counts=Counter({"llm_only": len(results)}),
        path_counts=Counter({"llm_only": len(results)}),
    )


def run_v02_alpha(
    *,
    input_path: str,
    output_path: str,
    trace_output: str,
    model_id: str | None = None,
    limit: int | None = None,
    safe_mode: bool = False,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
) -> None:
    """Run route-aware direct guided choice without S4 self-consistency."""
    _run_parsed_solver(
        version="v02_alpha",
        input_path=input_path,
        output_path=output_path,
        trace_output=trace_output,
        solver=_solve_direct_route,
        model_id=model_id,
        limit=limit,
        safe_mode=safe_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def run_v02_beta(
    *,
    input_path: str,
    output_path: str,
    trace_output: str,
    model_id: str | None = None,
    limit: int | None = None,
    safe_mode: bool = False,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    sc_batch_size: int | None = None,
) -> None:
    """Run S4 self-consistency/escalation."""
    resolved_sc_batch_size = _resolve_sc_batch_size(sc_batch_size, safe_mode)

    def solver(agent: ReasoningAgent, parsed: ParsedQuestion) -> SolveResult:
        return solve_question(
            agent,
            parsed,
            sc_batch_size=resolved_sc_batch_size,
        )

    _run_parsed_solver(
        version="v02_beta",
        input_path=input_path,
        output_path=output_path,
        trace_output=trace_output,
        solver=solver,
        model_id=model_id,
        limit=limit,
        safe_mode=safe_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def _run_parsed_solver(
    *,
    version: str,
    input_path: str,
    output_path: str,
    trace_output: str,
    solver: Callable[[ReasoningAgent, ParsedQuestion], SolveResult],
    model_id: str | None,
    limit: int | None,
    safe_mode: bool,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> None:
    t_start = time.time()
    agent = _load_agent(
        model_id=model_id,
        safe_mode=safe_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        t_start=t_start,
    )
    questions = _load_limited_questions(input_path, limit)
    print(f"Processing {len(questions)} questions ({version})...", flush=True)

    results: list[dict[str, str]] = []
    route_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    run_start = time.time()

    with _trace_writer(trace_output) as write_trace:
        for i, question in enumerate(questions):
            q_start = time.time()
            parsed = parse_question(question)
            solved = solver(agent, parsed)
            elapsed = time.time() - q_start

            results.append({"qid": solved.qid, "answer": solved.answer})
            route_counts[str(solved.route)] += 1
            path_counts[solved.path] += 1
            write_trace(_solve_trace(solved, elapsed))
            _print_progress(
                i=i,
                total=len(questions),
                qid=solved.qid,
                answer=solved.answer,
                route=solved.route,
                path=solved.path,
                margin=solved.margin,
                votes=solved.votes,
                error=solved.error,
                q_elapsed=elapsed,
                run_start=run_start,
            )

    _finish_run(
        results=results,
        output_path=output_path,
        t_start=t_start,
        run_start=run_start,
        route_counts=route_counts,
        path_counts=path_counts,
    )


def _solve_direct_route(agent: ReasoningAgent, parsed: ParsedQuestion) -> SolveResult:
    route = route_question(parsed)
    try:
        forced = get_forced_answer(parsed, route)
        if forced is not None:
            return SolveResult(
                qid=parsed.qid,
                answer=forced,
                route=route,
                margin=None,
                path="forced_safety",
                layer1_route=route,
                first_answer=forced,
            )

        first: ChoiceResult = agent.predict_route_choice_result(
            route=route,
            question=parsed.query,
            options=parsed.options,
            context=parsed.context if route == "reading" else None,
        )
        return SolveResult(
            qid=parsed.qid,
            answer=first.letter,
            route=route,
            margin=first.margin,
            path="direct",
            layer1_route=route,
            first_answer=first.letter,
        )
    except Exception as exc:
        return SolveResult(
            qid=parsed.qid,
            answer=FALLBACK,
            route=route,
            margin=None,
            path="fallback",
            layer1_route=route,
            error=str(exc),
        )


def _load_agent(
    *,
    model_id: str | None,
    safe_mode: bool,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
) -> ReasoningAgent:
    import torch

    chosen_gpu_util = gpu_memory_utilization
    chosen_max_len = max_model_len
    chosen_max_seqs = max_num_seqs

    if safe_mode:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else SAFE_GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else SAFE_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else SAFE_MAX_NUM_SEQS
    else:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else MAX_NUM_SEQS

    if torch.cuda.is_available():
        try:
            print("Loading primary model with vLLM...", flush=True)
            llm = load_vllm_primary(
                model_id=model_id,
                gpu_memory_utilization=chosen_gpu_util,
                max_model_len=chosen_max_len,
                max_num_seqs=chosen_max_seqs,
            )
            print(f"Primary model loaded with vLLM in {time.time() - t_start:.1f}s", flush=True)
            return ReasoningAgent(llm=llm)
        except Exception as exc:
            print(f"vLLM unavailable ({exc}), falling back to HuggingFace", flush=True)

    print("Loading primary model with HuggingFace...", flush=True)
    model, tokenizer = load_primary_model(model_id=model_id)
    print(f"Primary model loaded in {time.time() - t_start:.1f}s", flush=True)
    return ReasoningAgent(model=model, tokenizer=tokenizer)


def _load_limited_questions(input_path: str, limit: int | None) -> list[dict]:
    from src.data_loader import load_questions

    questions = load_questions(input_path)
    if limit is not None:
        return questions[:limit]
    return questions


def _resolve_sc_batch_size(sc_batch_size: int | None, safe_mode: bool) -> int | None:
    if sc_batch_size is not None:
        return sc_batch_size
    if safe_mode:
        return 1
    return None


def _trace_writer(path: str):
    class _Writer:
        def __enter__(self):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.path, "w", encoding="utf-8")
            return self.write

        def write(self, record: dict) -> None:
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.handle.flush()

        def __exit__(self, exc_type, exc, tb):
            self.handle.close()

    return _Writer()


def _baseline_trace(
    *,
    qid: str,
    answer: str,
    raw_output: str,
    elapsed_seconds: float,
) -> dict:
    return {
        "qid": qid,
        "answer": answer,
        "route": None,
        "path": "llm_only",
        "margin": None,
        "first_answer": None,
        "votes": [],
        "layer1_route": None,
        "semantic_route": None,
        "route_override": False,
        "override_blockers": [],
        "rag_used": False,
        "rag_top_score": None,
        "error": None,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "raw_output": raw_output,
    }


def _solve_trace(solved: SolveResult, elapsed_seconds: float) -> dict:
    return {
        "qid": solved.qid,
        "answer": solved.answer,
        "route": solved.route,
        "path": solved.path,
        "margin": solved.margin,
        "first_answer": solved.first_answer,
        "votes": solved.votes,
        "error": solved.error,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def _print_progress(
    *,
    i: int,
    total: int,
    qid: str,
    answer: str,
    route: str | None,
    path: str,
    margin: float | None,
    votes: list[str],
    error: str | None,
    q_elapsed: float,
    run_start: float,
) -> None:
    avg = (time.time() - run_start) / (i + 1)
    eta = avg * (total - i - 1)
    margin_text = f"{margin:.3f}" if margin is not None else "n/a"
    votes_text = f" votes={''.join(votes)}" if votes else ""
    error_text = f" error={error}" if error else ""
    route_text = route or "none"
    print(
        f"  [{i + 1}/{total}] {qid} "
        f"route={route_text} path={path} answer={answer} "
        f"margin={margin_text}{votes_text}{error_text} "
        f"({q_elapsed:.1f}s, avg {avg:.1f}s/q, ETA {eta / 60:.0f}min)",
        flush=True,
    )


def _finish_run(
    *,
    results: list[dict[str, str]],
    output_path: str,
    t_start: float,
    run_start: float,
    route_counts: Counter[str],
    path_counts: Counter[str],
) -> None:
    from src.data_loader import write_submission

    write_submission(results, output_path)
    total = time.time() - t_start
    infer_only = time.time() - run_start
    print(f"Written {len(results)} predictions to {output_path}", flush=True)
    print(f"Route counts: {dict(route_counts)}", flush=True)
    print(f"Path counts: {dict(path_counts)}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer_only:.1f}s, {infer_only / max(len(results), 1):.2f}s/question)",
        flush=True,
    )
