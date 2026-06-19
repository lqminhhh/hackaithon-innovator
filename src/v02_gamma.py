"""v02 gamma: wave-batched escalation + adaptive self-consistency.

Wave 1: batch ALL first passes (S2 two-pass guided-choice) → (letter, margin) per qid.
Wave 2: batch ALL escalations together — STEM SC, low-margin KNOWLEDGE SC, READING-reason SC.

Key differences from v02_beta (per-question loop):
- vLLM sees all questions at once → 4–10× wall-clock speedup.
- STEM always runs SC, never skipped — the v02_delta lesson.
- STEM SC depth is adaptive: n=3 (margin high) / n=7 (margin low).
- Option shuffle de-biases SC majority votes.
- Per-wave checkpoint; atexit writes submission on crash.
- gpu_memory_utilization=0.80 → 12.8 GB on a 16 GB card (leaves 3.2 GB headroom).
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

from src.data_loader import load_questions
from src.models import load_primary_model, load_vllm_primary
from src.parser import parse_question
from src.reasoning_agent import ReasoningAgent
from src.sc_policy import (
    GAMMA_GPU_MEM_UTIL,
    GAMMA_MAX_MODEL_LEN,
    SC_N_STEM,
)
from src.wave_solver import (
    finalize_answers,
    path_counts,
    run_wave1,
    run_wave2,
    write_results,
    write_traces,
)
from src.version_runner import add_common_args

# 16 GB × 0.80 = 12.8 GB — leaves 3.2 GB for OS/driver/desktop on unknown judge cards.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "public-test_1780368312.json"


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run v02_gamma wave-batched pipeline")
    add_common_args(
        parser,
        default_output="data/submissions/submission_v02_gamma.csv",
        default_trace="data/traces/trace_v02_gamma.jsonl",
        include_sc_batch=False,
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

    run_v02_gamma(
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


def run_v02_gamma(
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
    """Run the v02_gamma wave-batched pipeline.

    Parameters
    ----------
    adaptive_sc:
        When True (default) STEM SC depth adapts to margin:
        n=SC_N_STEM["high"]=3 if margin high, n=SC_N_STEM["low"]=7 if low.
        When False, STEM always uses SC_N_STEM["high"]=3 (faster, less thorough).
        KNOWLEDGE and READING escalation thresholds are unaffected by this flag.
    """
    t_start = time.time()

    # Resolve vLLM settings.  safe_mode imposes conservative VRAM limits so
    # the runner is safe on any 16 GB card regardless of unknown driver overhead.
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
    print(f"Processing {len(parsed_list)} questions (v02_gamma)...", flush=True)

    output_path_obj = Path(output_path)
    ckpt_path = output_path_obj.with_suffix(".ckpt")

    # Mutable container so the atexit closure always sees the latest answers.
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

    # ── Wave 1: batch ALL first passes ────────────────────────────────────────
    print("Wave 1: batching all first passes...", flush=True)
    wave1 = run_wave1(agent, parsed_list, set(_state["answers"]))
    _state["answers"].update({r.qid: r.answer for r in wave1.values()})
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

    # ── Wave 2: batch ALL escalations ─────────────────────────────────────────
    print("Wave 2: batching all escalations...", flush=True)
    wave2 = run_wave2(agent, parsed_list, wave1, adaptive_sc=adaptive_sc)
    _state["answers"].update({qid: w2.answer for qid, w2 in wave2.items()})
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 2 complete ({len(wave2)} escalated).", flush=True)

    # ── Finalize ──────────────────────────────────────────────────────────────
    final = finalize_answers(parsed_list, wave1, wave2, _state["answers"])
    write_results(final, parsed_list, output_path)
    write_traces(trace_output, parsed_list, wave1, wave2, final)

    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Summary
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


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save_ckpt(path: Path, answers: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic rename on POSIX and most Windows filesystems


def _load_ckpt(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ── Model loading ─────────────────────────────────────────────────────────────

def _load_agent(
    *,
    model_id: str | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
) -> ReasoningAgent:
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
