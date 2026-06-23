"""v03 gamma: final wave-batched submission runner.

Wave 1: batch ALL first passes (S2 two-pass guided-choice) -> (letter, margin) per qid.
Wave 2: batch ALL escalations together - STEM SC, low-margin KNOWLEDGE SC, READING-reason SC.

Key traits of the final submission path:
- vLLM sees all questions at once -> large wall-clock speedup from wave batching.
- STEM always runs SC, never skipped.
- Route-aware compute recovery keeps hard KNOWLEDGE / READING cases from falling
  back to the cheapest direct path.
- Per-wave checkpointing plus atexit emergency write preserves outputs on crash.
- `--safe-mode` constrains vLLM for 16 GB judge-like cards.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SAFE_GPU_MEM_UTIL, SAFE_MAX_MODEL_LEN, SAFE_MAX_NUM_SEQS
from src.sc_policy import (
    GAMMA_GPU_MEM_UTIL,
    GAMMA_MAX_MODEL_LEN,
    SC_N_STEM,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "public-test_1780368312.json"


def add_common_args(parser: argparse.ArgumentParser, *, default_output: str, default_trace: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v03_gamma wave-batched pipeline")
    add_common_args(
        parser,
        default_output="data/submissions/submission_v03_gamma.csv",
        default_trace="data/traces/trace_v03_gamma.jsonl",
    )
    parser.add_argument(
        "--no-adaptive-sc",
        dest="adaptive_sc",
        action="store_false",
        default=True,
        help=(
            "Disable adaptive SC depth for STEM. "
            "When set, STEM always uses SC_N_STEM['high'] "
            f"(n={SC_N_STEM['high']}) regardless of margin. "
            "Default: adaptive (n=3 if margin high, n=7 if margin low)."
        ),
    )
    args = parser.parse_args()

    run_v03_gamma(
        input_path=args.input,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        limit=args.limit,
        safe_mode=args.safe_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        adaptive_sc=args.adaptive_sc,
    )


def run_v03_gamma(
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
    adaptive_sc: bool = True,
) -> None:
    """Run the v03_gamma wave-batched pipeline."""
    from src.data_loader import load_questions
    from src.parser import parse_question
    from src.wave_solver import (
        finalize_answers,
        path_counts,
        run_wave1,
        run_wave2,
        write_results,
        write_traces,
    )

    t_start = time.time()

    chosen_gpu_util = gpu_memory_utilization
    chosen_max_len = max_model_len
    chosen_max_seqs = max_num_seqs
    if safe_mode:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else SAFE_GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else SAFE_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else SAFE_MAX_NUM_SEQS
    else:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else GAMMA_GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else GAMMA_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else 16

    agent = _load_agent(
        model_id=model_id,
        gpu_memory_utilization=chosen_gpu_util,
        max_model_len=chosen_max_len,
        max_num_seqs=chosen_max_seqs,
        t_start=t_start,
    )

    questions = load_questions(input_path)
    if limit is not None:
        questions = questions[:limit]
    parsed_list = [parse_question(q) for q in questions]
    print(f"Processing {len(parsed_list)} questions (v03_gamma)...", flush=True)

    output_path_obj = Path(output_path)
    ckpt_path = output_path_obj.with_suffix(".ckpt")
    state: dict[str, Any] = {
        "answers": _load_ckpt(ckpt_path),
        "parsed_list": parsed_list,
        "output_path": output_path,
    }

    def _emergency_write() -> None:
        try:
            write_results(state["answers"], state["parsed_list"], state["output_path"])
        except Exception:
            pass

    atexit.register(_emergency_write)

    run_start = time.time()

    print("Wave 1: batching all first passes...", flush=True)
    wave1 = run_wave1(agent, parsed_list, set(state["answers"]))
    state["answers"].update({r.qid: r.answer for r in wave1.values()})
    _save_ckpt(ckpt_path, state["answers"])
    print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

    print("Wave 2: batching all escalations...", flush=True)
    wave2 = run_wave2(agent, parsed_list, wave1, adaptive_sc=adaptive_sc)
    state["answers"].update({qid: w2.answer for qid, w2 in wave2.items()})
    _save_ckpt(ckpt_path, state["answers"])
    print(f"Wave 2 complete ({len(wave2)} escalated).", flush=True)

    final = finalize_answers(parsed_list, wave1, wave2, state["answers"])
    write_results(final, parsed_list, output_path)
    write_traces(trace_output, parsed_list, wave1, wave2, final)

    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    route_counts: Counter[str] = Counter(r.route for r in wave1.values())
    path_counts_summary = path_counts(parsed_list, wave1, wave2)

    total = time.time() - t_start
    infer = time.time() - run_start
    n = max(len(parsed_list), 1)
    print(f"Written {len(final)} predictions to {output_path}", flush=True)
    print(f"Route counts: {dict(route_counts)}", flush=True)
    print(f"Path counts:  {dict(path_counts_summary)}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer:.1f}s, {infer / n:.2f}s/question)",
        flush=True,
    )


def _save_ckpt(path: Path, answers: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_ckpt(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_agent(
    *,
    model_id: str | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
) -> Any:
    from src.models import load_primary_model, load_vllm_primary
    from src.reasoning_agent import ReasoningAgent
    import torch

    if torch.cuda.is_available():
        try:
            print("Loading primary model with vLLM...", flush=True)
            llm = load_vllm_primary(
                model_id=model_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
            print(f"Model loaded (vLLM) in {time.time() - t_start:.1f}s", flush=True)
            return ReasoningAgent(llm=llm)
        except Exception as exc:
            print(f"vLLM unavailable ({exc}), falling back to HuggingFace", flush=True)

    print("Loading primary model with HuggingFace...", flush=True)
    model, tokenizer = load_primary_model(model_id=model_id)
    print(f"Model loaded (HF) in {time.time() - t_start:.1f}s", flush=True)
    return ReasoningAgent(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
