"""v03 gamma: final wave-batched submission runner.

Wave 1: batch ALL first passes (S2 two-pass guided-choice) -> (letter, margin) per qid.
Wave 2: batch ALL escalations together - STEM SC, low-margin KNOWLEDGE SC, READING-reason SC.

Key traits of the final submission path:
- vLLM sees all questions at once -> large wall-clock speedup from wave batching.
- STEM always runs SC, never skipped.
- Route-aware compute recovery keeps hard KNOWLEDGE / READING cases from falling
  back to the cheapest direct path.
- Per-wave checkpointing plus always-emit best-effort output preserves results
  on exception or kill signal.
- `--safe-mode` uses conservative vLLM settings for the final 16 GB target.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FALLBACK, SAFE_GPU_MEM_UTIL, SAFE_MAX_MODEL_LEN, SAFE_MAX_NUM_SEQS
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
        help="Use conservative vLLM settings for the final 16GB VRAM target.",
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
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        default=True,
        help="Disable the additive vLLM pre-run warmup step.",
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
        warmup=args.warmup,
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
    install_handlers: bool = True,
    warmup: bool = True,
) -> None:
    """Run the v03_gamma wave-batched pipeline."""
    from src.data_loader import load_questions
    from src.parser import ParsedQuestion, parse_question
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

    questions = load_questions(input_path)
    if limit is not None:
        questions = questions[:limit]
    parsed_list: list[ParsedQuestion] = []
    for question in questions:
        try:
            parsed_list.append(parse_question(question))
        except Exception as exc:
            qid = str(question.get("qid", question.get("id", "")))
            print(f"[v03_gamma] qid={qid} parse failed -> FALLBACK shape: {exc}", flush=True)
            options = question.get("options")
            if not isinstance(options, dict) or not options:
                options = {"A": ""}
            raw_question = str(question.get("question", ""))
            parsed_list.append(
                ParsedQuestion(
                    qid=qid,
                    original_question=raw_question,
                    query=raw_question,
                    context=None,
                    options=options,
                    refusal_labels=(),
                    n_choices=len(options),
                    has_context=False,
                    is_quantitative=False,
                    is_legal=False,
                    has_refusal_choice=False,
                    is_harmful=False,
                )
            )
    print(f"Processing {len(parsed_list)} questions (v03_gamma)...", flush=True)

    output_path_obj = Path(output_path)
    ckpt_path = output_path_obj.with_suffix(".ckpt")
    ckpt_answers = _load_ckpt(ckpt_path)
    restored_qids = set(ckpt_answers)
    prefilled_answers = {parsed.qid: FALLBACK for parsed in parsed_list}
    prefilled_answers.update(ckpt_answers)
    state: dict[str, Any] = {
        "answers": prefilled_answers,
        "parsed_list": parsed_list,
        "output_path": output_path,
    }

    def _emergency_write() -> None:
        try:
            _write_results_atomic(state["answers"], state["parsed_list"], state["output_path"])
        except Exception:
            pass

    cleanup = _install_always_emit(_emergency_write) if install_handlers else (lambda: None)

    run_start = time.time()
    wave1: dict[str, Any] = {}
    wave2: dict[str, Any] = {}
    failed = False
    succeeded = False
    runtime_info = _build_runtime_info(
        safe_mode=safe_mode,
        gpu_memory_utilization=chosen_gpu_util,
        max_model_len=chosen_max_len,
        max_num_seqs=chosen_max_seqs,
    )

    try:
        agent = _load_agent(
            model_id=model_id,
            gpu_memory_utilization=chosen_gpu_util,
            max_model_len=chosen_max_len,
            max_num_seqs=chosen_max_seqs,
            t_start=t_start,
        )
        runtime_info.update(getattr(agent, "_backend_info", {}))
        _log_runtime_info(runtime_info)
        if warmup:
            _warmup_agent(agent)

        print("Wave 1: batching all first passes...", flush=True)
        wave1 = run_wave1(agent, parsed_list, restored_qids)
        state["answers"].update({r.qid: r.answer for r in wave1.values()})
        _save_ckpt(ckpt_path, state["answers"])
        print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

        print("Wave 2: batching all escalations...", flush=True)
        wave2 = run_wave2(agent, parsed_list, wave1, adaptive_sc=adaptive_sc)
        state["answers"].update({qid: w2.answer for qid, w2 in wave2.items()})
        _save_ckpt(ckpt_path, state["answers"])
        print(f"Wave 2 complete ({len(wave2)} escalated).", flush=True)

        final = finalize_answers(parsed_list, wave1, wave2, state["answers"])
        state["answers"].update(final)
        _write_results_atomic(final, parsed_list, output_path)
        write_traces(
            trace_output,
            parsed_list,
            wave1,
            wave2,
            final,
            runtime_info=runtime_info,
        )

        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass
        succeeded = True

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
    except Exception as exc:
        failed = True
        print(f"[v03_gamma] run degraded to checkpoint/fallback output: {exc}", flush=True)
    finally:
        if not succeeded:
            try:
                _save_ckpt(ckpt_path, state["answers"])
            except Exception:
                pass
        _emergency_write()
        cleanup()
        if failed:
            print(
                f"[v03_gamma] wrote best-effort submission to {output_path} "
                f"using checkpointed/fallback answers.",
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


def _write_results_atomic(answers: dict[str, str], parsed_list: list[Any], output_path: str) -> None:
    from src.wave_solver import write_results

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path_obj.with_name(output_path_obj.name + ".tmp")
    write_results(answers, parsed_list, str(tmp))
    os.replace(tmp, output_path_obj)


def _install_always_emit(emitter) -> Any:
    def _safe_emit() -> None:
        try:
            emitter()
        except Exception:
            pass

    def _signal_handler(_signum, _frame) -> None:
        _safe_emit()
        os._exit(0)

    atexit.register(_safe_emit)
    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    def _cleanup() -> None:
        atexit.unregister(_safe_emit)
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, TypeError):
                pass

    return _cleanup


def _warmup_agent(agent: Any) -> None:
    """Best-effort deterministic warmup for common vLLM inference shapes.

    This is intentionally additive: it runs before the real pipeline, uses
    deterministic generation only, and swallows failures so normal inference
    still proceeds unchanged.
    """
    if not getattr(agent, "is_vllm", False):
        return

    from src.batch_extract import batch_extract

    long_context = " ".join(["chi tiet"] * 900)
    reasoning_prompts = [
        "Cau hoi: 2 + 2 bang bao nhieu?\nA) 3\nB) 4\nTra loi ngan gon.",
        (
            "Doan thong tin:\n"
            f"{long_context}\n\n"
            "Cau hoi: Theo doan thong tin, chi tiet nao duoc nhac den?\n"
            "A) Lua chon 1\nB) Lua chon 2\nC) Lua chon 3\nD) Lua chon 4"
        ),
        "Giai bai toan ngan gon tung buoc va chon dap an dung nhat.\nA) 1\nB) 2\nC) 3\nD) 4",
    ]
    extract_prompts = [
        "Chon dung mot dap an hop le.\nA) 3\nB) 4\nDap an: ",
        (
            "Chon dung mot dap an hop le.\n"
            + "\n".join(f"{chr(ord('A') + i)}) Lua chon {i + 1}" for i in range(10))
            + "\nDap an: "
        ),
    ]
    extract_options = [
        {"A": "3", "B": "4"},
        {chr(ord("A") + i): f"Lua chon {i + 1}" for i in range(10)},
    ]

    print("[v03_gamma] Warmup: priming vLLM kernels...", flush=True)
    t0 = time.time()
    try:
        agent.generate_freeform(
            [reasoning_prompts[0], reasoning_prompts[1]],
            mode="no_think",
            max_tokens=16,
            temperature=0.0,
        )
        agent.generate_freeform(
            [reasoning_prompts[2]],
            mode="think",
            max_tokens=32,
            temperature=0.0,
        )
        batch_extract(agent, extract_prompts, extract_options)
    except Exception as exc:
        print(f"[v03_gamma] Warmup skipped after error: {exc}", flush=True)
        return
    print(f"[v03_gamma] Warmup complete in {time.time() - t0:.1f}s", flush=True)


def _build_runtime_info(
    *,
    safe_mode: bool,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> dict[str, Any]:
    runtime_info: dict[str, Any] = {
        "safe_mode": safe_mode,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching_requested": True,
    }
    try:
        import torch

        runtime_info["torch_version"] = getattr(torch, "__version__", None)
        runtime_info["cuda_available"] = bool(torch.cuda.is_available())
        runtime_info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            runtime_info["gpu_name"] = props.name
            runtime_info["gpu_total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            runtime_info["gpu_capability"] = f"{props.major}.{props.minor}"
            runtime_info["cuda_arch_list"] = list(torch.cuda.get_arch_list())
            try:
                probe = torch.zeros(1, device=f"cuda:{device_index}")
                runtime_info["cuda_smoke_test"] = float((probe + 1).item()) == 1.0
            except Exception as exc:
                runtime_info["cuda_smoke_test"] = False
                runtime_info["cuda_smoke_test_error"] = str(exc)
    except Exception as exc:
        runtime_info["runtime_probe_error"] = str(exc)

    try:
        import vllm

        runtime_info["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception as exc:
        runtime_info["vllm_version"] = None
        runtime_info["vllm_probe_error"] = str(exc)

    return runtime_info


def _log_runtime_info(runtime_info: dict[str, Any]) -> None:
    print(
        "[v03_gamma] Runtime info: "
        + json.dumps(runtime_info, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


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
            agent = ReasoningAgent(llm=llm)
            agent._backend_info = {
                "backend": "vllm",
                "backend_reason": "loaded_vllm",
                "model_id": getattr(llm, "model", model_id),
                "enable_prefix_caching_effective": getattr(llm, "enable_prefix_caching", None),
            }
            return agent
        except Exception as exc:
            print(f"vLLM unavailable ({exc}), falling back to HuggingFace", flush=True)
            fallback_reason = str(exc)
    else:
        fallback_reason = "cuda_unavailable"

    print("Loading primary model with HuggingFace...", flush=True)
    model, tokenizer = load_primary_model(model_id=model_id)
    print(f"Model loaded (HF) in {time.time() - t_start:.1f}s", flush=True)
    agent = ReasoningAgent(model=model, tokenizer=tokenizer)
    agent._backend_info = {
        "backend": "huggingface",
        "backend_reason": fallback_reason,
        "model_id": model_id,
        "enable_prefix_caching_effective": False,
    }
    return agent


if __name__ == "__main__":
    main()
