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
import gc
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    FALLBACK,
    MAX_NUM_SEQS,
    SAFE_DYNAMIC_HEADROOM_GB,
    SAFE_HEADROOM_LADDER_GB,
    SAFE_MAX_MODEL_LEN,
    SAFE_MAX_NUM_SEQS,
    SAFE_UTILIZATION_CLAMP_MAX,
    SAFE_UTILIZATION_CLAMP_MIN,
)
from src.sc_policy import (
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
        chosen_max_len = chosen_max_len if chosen_max_len is not None else SAFE_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else SAFE_MAX_NUM_SEQS
    else:
        chosen_max_len = chosen_max_len if chosen_max_len is not None else GAMMA_MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else (MAX_NUM_SEQS or 32)

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
            safe_mode=safe_mode,
        )
        runtime_info.update(getattr(agent, "_backend_info", {}))
        _log_runtime_info(runtime_info)
        if warmup:
            _warmup_agent(agent)

        print("Wave 1: batching all first passes...", flush=True)
        wave1, agent = _run_wave_with_retry(
            "wave1",
            lambda active_agent: run_wave1(active_agent, parsed_list, restored_qids),
            agent=agent,
            model_id=model_id,
            gpu_memory_utilization=chosen_gpu_util,
            max_model_len=chosen_max_len,
            max_num_seqs=chosen_max_seqs,
            t_start=t_start,
            safe_mode=safe_mode,
            runtime_info=runtime_info,
        )
        state["answers"].update({r.qid: r.answer for r in wave1.values()})
        _save_ckpt(ckpt_path, state["answers"])
        print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

        print("Wave 2: batching all escalations...", flush=True)
        wave2, agent = _run_wave_with_retry(
            "wave2",
            lambda active_agent: run_wave2(active_agent, parsed_list, wave1, adaptive_sc=adaptive_sc),
            agent=agent,
            model_id=model_id,
            gpu_memory_utilization=chosen_gpu_util,
            max_model_len=chosen_max_len,
            max_num_seqs=chosen_max_seqs,
            t_start=t_start,
            safe_mode=safe_mode,
            runtime_info=runtime_info,
        )
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


def _run_wave_with_retry(
    wave_name: str,
    runner,
    *,
    agent: Any,
    model_id: str | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
    safe_mode: bool,
    runtime_info: dict[str, Any],
) -> tuple[Any, Any]:
    try:
        return runner(agent), agent
    except Exception as exc:
        if not _should_retry_wave(agent, exc):
            raise

        current_index = int(getattr(agent, "_backend_info", {}).get("vllm_headroom_index", -1))
        current_headroom = getattr(agent, "_backend_info", {}).get("gpu_memory_headroom_gb")
        print(
            f"[v03_gamma] {wave_name} hit OOM-like failure at headroom={current_headroom} GB: {exc}",
            flush=True,
        )
        _dispose_agent(agent)
        retry_agent = _load_agent(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            t_start=t_start,
            safe_mode=safe_mode,
            min_headroom_index=current_index + 1,
            retry_context=f"{wave_name}_oom_retry",
        )
        runtime_info.update(getattr(retry_agent, "_backend_info", {}))
        _log_runtime_info(runtime_info)
        return runner(retry_agent), retry_agent


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


def _should_retry_wave(agent: Any, exc: Exception) -> bool:
    backend_info = getattr(agent, "_backend_info", {})
    if backend_info.get("backend") != "vllm":
        return False
    if backend_info.get("gpu_memory_utilization_requested") is not None:
        return False
    if not _is_probable_oom(exc):
        return False
    headroom_index = int(backend_info.get("vllm_headroom_index", -1))
    headroom_ladder = backend_info.get("vllm_headroom_ladder_gb") or []
    return headroom_index + 1 < len(headroom_ladder)


def _is_probable_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass

    message = str(exc).lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "oom",
        "enginedeaderror",
    )
    return any(needle in message for needle in needles)


def _dispose_agent(agent: Any) -> None:
    llm = getattr(agent, "_llm", None)
    engine = getattr(llm, "engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    del agent
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def _build_vllm_attempts(
    *,
    requested_utilization: float | None,
    headroom_ladder_gb: tuple[float, ...],
    clamp_min: float,
    clamp_max: float,
    min_headroom_index: int,
) -> list[dict[str, Any]]:
    if requested_utilization is not None:
        return [
            {
                "headroom_gb": None,
                "headroom_index": 0,
                "gpu_memory_utilization": requested_utilization,
                "dynamic_vram_sizing": False,
            }
        ]

    attempts: list[dict[str, Any]] = []
    for idx, headroom_gb in enumerate(headroom_ladder_gb):
        if idx < min_headroom_index:
            continue
        attempts.append(
            _dynamic_vllm_attempt(
                headroom_gb=headroom_gb,
                headroom_index=idx,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
        )
    return attempts


def _dynamic_vllm_attempt(
    *,
    headroom_gb: float,
    headroom_index: int,
    clamp_min: float,
    clamp_max: float,
) -> dict[str, Any]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = float(1024 ** 3)
    free_gb = free_bytes / gib
    total_gb = total_bytes / gib
    requested_util = (free_bytes - (headroom_gb * gib)) / total_bytes
    effective_util = max(clamp_min, min(clamp_max, requested_util))
    return {
        "headroom_gb": round(headroom_gb, 3),
        "headroom_index": headroom_index,
        "gpu_memory_utilization": round(effective_util, 6),
        "gpu_memory_utilization_unclamped": round(requested_util, 6),
        "gpu_memory_free_gb": round(free_gb, 3),
        "gpu_memory_total_gb": round(total_gb, 3),
        "gpu_memory_free_bytes": int(free_bytes),
        "gpu_memory_total_bytes": int(total_bytes),
        "utilization_clamp_min": clamp_min,
        "utilization_clamp_max": clamp_max,
        "dynamic_vram_sizing": True,
    }


def _load_agent(
    *,
    model_id: str | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_seqs: int | None,
    t_start: float,
    safe_mode: bool,
    min_headroom_index: int = 0,
    retry_context: str = "initial_load",
) -> Any:
    from src.models import load_primary_model, load_vllm_primary
    from src.reasoning_agent import ReasoningAgent
    import torch

    clamp_min = SAFE_UTILIZATION_CLAMP_MIN
    clamp_max = SAFE_UTILIZATION_CLAMP_MAX
    headroom_ladder_gb = tuple(SAFE_HEADROOM_LADDER_GB) or (SAFE_DYNAMIC_HEADROOM_GB,)

    if torch.cuda.is_available():
        attempt_records: list[dict[str, Any]] = []
        attempts = _build_vllm_attempts(
            requested_utilization=gpu_memory_utilization,
            headroom_ladder_gb=headroom_ladder_gb,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            min_headroom_index=min_headroom_index,
        )
        for attempt_number, attempt in enumerate(attempts, start=1):
            effective_util = float(attempt["gpu_memory_utilization"])
            headroom_gb = attempt.get("headroom_gb")
            label = (
                f"dynamic util={effective_util:.3f} (free={attempt['gpu_memory_free_gb']:.2f} GB, "
                f"headroom={headroom_gb:.2f} GB)"
                if attempt.get("dynamic_vram_sizing")
                else f"explicit util={effective_util:.3f}"
            )
            try:
                print(
                    f"Loading primary model with vLLM ({retry_context}, attempt {attempt_number}/{len(attempts)}; {label})...",
                    flush=True,
                )
                print(
                    f"[v03_gamma] vLLM sizing probe: free={attempt.get('gpu_memory_free_gb')} GB, "
                    f"total={attempt.get('gpu_memory_total_gb')} GB, "
                    f"chosen_util={effective_util:.3f}",
                    flush=True,
                )
                llm = load_vllm_primary(
                    model_id=model_id,
                    gpu_memory_utilization=effective_util,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                )
                print(f"Model loaded (vLLM) in {time.time() - t_start:.1f}s", flush=True)
                agent = ReasoningAgent(llm=llm)
                agent._backend_info = {
                    "backend": "vllm",
                    "backend_reason": "loaded_vllm",
                    "model_id": getattr(llm, "model", model_id),
                    "safe_mode": safe_mode,
                    "gpu_memory_utilization_requested": gpu_memory_utilization,
                    "gpu_memory_utilization_effective": effective_util,
                    "gpu_memory_headroom_gb": headroom_gb,
                    "gpu_memory_free_gb": attempt.get("gpu_memory_free_gb"),
                    "gpu_memory_total_gb": attempt.get("gpu_memory_total_gb"),
                    "gpu_memory_utilization_unclamped": attempt.get("gpu_memory_utilization_unclamped"),
                    "utilization_clamp_min": clamp_min,
                    "utilization_clamp_max": clamp_max,
                    "dynamic_vram_sizing": attempt.get("dynamic_vram_sizing", False),
                    "vllm_headroom_index": attempt.get("headroom_index", 0),
                    "vllm_headroom_ladder_gb": list(headroom_ladder_gb),
                    "vllm_attempt_count": attempt_number,
                    "vllm_attempts": attempt_records + [dict(attempt, status="loaded")],
                    "enable_prefix_caching_effective": getattr(llm, "enable_prefix_caching", None),
                    "enable_chunked_prefill_effective": getattr(llm, "enable_chunked_prefill", None),
                }
                return agent
            except Exception as exc:
                failure_record = dict(attempt)
                failure_record["status"] = "failed"
                failure_record["error"] = str(exc)
                attempt_records.append(failure_record)
                print(
                    f"[v03_gamma] vLLM attempt {attempt_number}/{len(attempts)} failed: {exc}",
                    flush=True,
                )
                _dispose_agent_locals(locals().get("llm"))
                if attempt_number == len(attempts):
                    fallback_reason = f"vllm_retry_exhausted:{exc}"
                    break
        else:
            fallback_reason = "vllm_attempts_unavailable"
    else:
        fallback_reason = "cuda_unavailable"

    print(
        f"Loading primary model with HuggingFace... (reason={fallback_reason})",
        flush=True,
    )
    model, tokenizer = load_primary_model(model_id=model_id)
    print(f"Model loaded (HF) in {time.time() - t_start:.1f}s", flush=True)
    agent = ReasoningAgent(model=model, tokenizer=tokenizer)
    agent._backend_info = {
        "backend": "huggingface",
        "backend_reason": fallback_reason,
        "model_id": model_id,
        "enable_prefix_caching_effective": False,
        "enable_chunked_prefill_effective": False,
    }
    return agent


def _dispose_agent_locals(llm: Any) -> None:
    engine = getattr(llm, "engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
