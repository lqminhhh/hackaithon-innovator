"""v03 delta: v03_gamma wave pipeline with a live vLLM LoRA adapter.

This runner avoids merging LoRA weights into a new model directory. vLLM loads
the clean base model and applies the adapter at inference time, which bypasses
the Qwen3.5 merged-metadata issues seen with ``*_merged`` checkpoints.
"""

from __future__ import annotations

import atexit
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_MODEL
from src.data_loader import load_questions
from src.models import load_vllm_primary
from src.parser import parse_question
from src.reasoning_agent import ReasoningAgent
from src.sc_policy import GAMMA_GPU_MEM_UTIL, GAMMA_MAX_MODEL_LEN, SC_N_STEM
from src.version_runner import add_common_args
from src.wave_solver import (
    finalize_answers,
    path_counts,
    run_wave1,
    run_wave2,
    write_results,
    write_traces,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LORA_ADAPTER = PROJECT_ROOT / "outputs" / "finetune" / "qwen35_4b_lora_2_5k_v1"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run v03_delta: v03_gamma wave pipeline with vLLM LoRA adapter"
    )
    add_common_args(
        parser,
        default_output="data/submissions/submission_v03_delta.csv",
        default_trace="data/traces/trace_v03_delta.jsonl",
        include_sc_batch=False,
    )
    parser.add_argument(
        "--lora-adapter",
        default=str(DEFAULT_LORA_ADAPTER),
        help=(
            "Path to the trained PEFT LoRA adapter directory. "
            "Use outputs/finetune/qwen35_4b_lora_2_5k_v1, not a merged model."
        ),
    )
    parser.add_argument(
        "--lora-name",
        default="v03_delta_lora",
        help="Stable name for the vLLM LoRA request.",
    )
    parser.add_argument(
        "--lora-int-id",
        type=int,
        default=1,
        help="Stable integer id for the vLLM LoRA request.",
    )
    parser.add_argument(
        "--no-adaptive-sc",
        dest="adaptive_sc",
        action="store_false",
        default=True,
        help=(
            "Disable adaptive SC depth for STEM. "
            "When set, STEM always uses SC_N_STEM['high'] "
            f"(n={SC_N_STEM['high']}) regardless of margin."
        ),
    )
    args = parser.parse_args()

    run_v03_delta(
        input_path=args.input,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        lora_adapter=args.lora_adapter,
        lora_name=args.lora_name,
        lora_int_id=args.lora_int_id,
        limit=args.limit,
        safe_mode=args.safe_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        adaptive_sc=args.adaptive_sc,
    )


def run_v03_delta(
    *,
    input_path: str,
    output_path: str,
    trace_output: str,
    model_id: str | None = None,
    lora_adapter: str,
    lora_name: str = "v03_delta_lora",
    lora_int_id: int = 1,
    limit: int | None = None,
    safe_mode: bool = False,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    adaptive_sc: bool = True,
) -> None:
    """Run v03_delta with base model + adapter loaded directly by vLLM."""
    t_start = time.time()

    adapter_path = Path(lora_adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}. "
            "Pass --lora-adapter pointing to the PEFT adapter directory, "
            "not the merged model directory."
        )

    chosen_gpu_util = gpu_memory_utilization
    chosen_max_len = max_model_len
    chosen_max_seqs = max_num_seqs
    if safe_mode:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else 0.70
        chosen_max_len = chosen_max_len if chosen_max_len is not None else GAMMA_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else 4
    else:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else GAMMA_GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else GAMMA_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else 16

    agent = _load_lora_agent(
        model_id=model_id or LLM_MODEL,
        lora_adapter=str(adapter_path),
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        gpu_memory_utilization=chosen_gpu_util,
        max_model_len=chosen_max_len,
        max_num_seqs=chosen_max_seqs,
        t_start=t_start,
    )

    questions = load_questions(input_path)
    if limit is not None:
        questions = questions[:limit]
    parsed_list = [parse_question(q) for q in questions]
    print(f"Processing {len(parsed_list)} questions (v03_delta + LoRA)...", flush=True)

    output_path_obj = Path(output_path)
    ckpt_path = output_path_obj.with_suffix(".ckpt")

    _state: dict[str, Any] = {
        "answers": _load_ckpt(ckpt_path),
        "parsed_list": parsed_list,
        "output_path": output_path,
    }

    def _emergency_write() -> None:
        try:
            write_results(_state["answers"], _state["parsed_list"], _state["output_path"])
        except Exception:
            pass

    atexit.register(_emergency_write)

    run_start = time.time()

    print("Wave 1: batching all first passes...", flush=True)
    wave1 = run_wave1(agent, parsed_list, set(_state["answers"]))
    _state["answers"].update({r.qid: r.answer for r in wave1.values()})
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

    print("Wave 2: batching all escalations...", flush=True)
    wave2 = run_wave2(agent, parsed_list, wave1, adaptive_sc=adaptive_sc)
    _state["answers"].update({qid: w2.answer for qid, w2 in wave2.items()})
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 2 complete ({len(wave2)} escalated).", flush=True)

    final = finalize_answers(parsed_list, wave1, wave2, _state["answers"])
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


def _load_lora_agent(
    *,
    model_id: str,
    lora_adapter: str,
    lora_name: str,
    lora_int_id: int,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
) -> ReasoningAgent:
    if not torch.cuda.is_available():
        raise RuntimeError("v03_delta requires CUDA because vLLM LoRA is the intended path.")

    print("Loading base model with vLLM + LoRA adapter...", flush=True)
    llm = load_vllm_primary(
        model_id=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        lora_adapter_path=lora_adapter,
        lora_name=lora_name,
        lora_int_id=lora_int_id,
    )
    print(
        f"Model loaded (vLLM + LoRA) in {time.time() - t_start:.1f}s "
        f"base={model_id} adapter={lora_adapter}",
        flush=True,
    )
    return ReasoningAgent(llm=llm)


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


if __name__ == "__main__":
    main()
