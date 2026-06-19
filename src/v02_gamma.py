"""v02 gamma: wave-batched escalation + adaptive self-consistency.

Wave 1: batch ALL first passes (S2 two-pass guided-choice) → (letter, margin) per qid.
Wave 2: batch ALL escalations together — STEM SC, low-margin KNOWLEDGE SC, READING-reason SC.

Key differences from v02_beta (per-question loop):
- vLLM sees all questions at once → 4–10× wall-clock speedup.
- STEM always runs SC, never skipped — the v02_delta lesson.
- STEM SC depth is adaptive: n=3 (margin high) / n=7 (margin low).
- Option shuffle de-biases SC majority votes.
- Per-wave checkpoint; atexit writes submission on crash.
- gpu_memory_utilization=0.85 → ≈13.6 GB on a 16 GB card (well under target).
"""

from __future__ import annotations

import atexit
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FALLBACK
from src.data_loader import load_questions, write_submission
from src.extract import (
    ChoiceResult,
    best_label,
    build_label_token_map,
    softmax_margin,
)
from src.models import load_primary_model, load_vllm_primary
from src.parser import ParsedQuestion, parse_question
from src.router import get_forced_answer, route_question
from src.reasoning_agent import ReasoningAgent
from src.solve import (
    _build_extraction_from_reasoning,
    _build_reasoning_prompt,
    _is_reason_purpose_question,
    _vote,
)
from src.version_runner import add_common_args, _trace_writer

# ── v3 local config ───────────────────────────────────────────────────────────

# Per-route low-margin thresholds — differ significantly by route (Issue 6).
_MARGIN_LOW = {"READING": 0.10, "STEM": 0.15, "KNOWLEDGE": 0.20, "SAFETY": 0.05}

# STEM SC depth is adaptive, not an early-exit toggle.
_SC_N_STEM = {"high": 3, "low": 7}

_SC_N = 5        # KNOWLEDGE escalation n
_SC_TEMP = 0.6   # Qwen3 thinking guidance — do NOT greedy-decode in think mode
_SC_TOP_P = 0.95
_SC_SEED = 1234  # deterministic SC across judge re-runs
_SHUFFLE_OPTIONS = True  # letter-position de-bias
_TOK = {"READING": 512, "STEM": 3072, "KNOWLEDGE": 256, "SAFETY": 128}

# 16 GB × 0.85 ≈ 13.6 GB — leaves 2.4 GB for OS/driver/fragmentation.
# The spec says 0.90 but also says "target ≤14 GB"; 0.85 gives real headroom.
_GPU_MEM_UTIL = 0.85
_MAX_MODEL_LEN = 4096

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "public-test_1780368312.json"


# ── internal dataclass ────────────────────────────────────────────────────────

@dataclass
class _W1Result:
    """Per-question result from Wave 1 (first pass)."""
    qid: str
    route: str                 # "stem" | "reading" | "knowledge" | "safety"
    answer: str
    margin: float | None
    forced: bool = False       # SAFETY forced-answer path
    error: str | None = None
    reasoning_prompt: str = ""
    per_letter_logprob: dict[str, float] = field(default_factory=dict)


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
            "When set, STEM always uses SC_N_STEM['high'] (n=3) regardless of margin. "
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
        chosen_max_len = chosen_max_len if chosen_max_len is not None else _MAX_MODEL_LEN
        chosen_max_seqs = chosen_max_seqs if chosen_max_seqs is not None else 4
    else:
        chosen_gpu_util = chosen_gpu_util if chosen_gpu_util is not None else _GPU_MEM_UTIL
        chosen_max_len = chosen_max_len if chosen_max_len is not None else _MAX_MODEL_LEN

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
            _write_results(_state["answers"], _state["parsed_list"], _state["output_path"])
        except Exception:
            pass

    atexit.register(_emergency_write)

    run_start = time.time()

    # ── Wave 1: batch ALL first passes ────────────────────────────────────────
    print("Wave 1: batching all first passes...", flush=True)
    wave1 = _run_wave1(agent, parsed_list, set(_state["answers"]))
    _state["answers"].update({r.qid: r.answer for r in wave1.values()})
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 1 complete ({len(wave1)} items).", flush=True)

    # ── Wave 2: batch ALL escalations ─────────────────────────────────────────
    print("Wave 2: batching all escalations...", flush=True)
    wave2 = _run_wave2(agent, parsed_list, wave1, adaptive_sc=adaptive_sc)
    _state["answers"].update(wave2)
    _save_ckpt(ckpt_path, _state["answers"])
    print(f"Wave 2 complete ({len(wave2)} escalated).", flush=True)

    # ── Finalize ──────────────────────────────────────────────────────────────
    final = _finalize(parsed_list, wave1, wave2, _state["answers"])
    _write_results(final, parsed_list, output_path)
    _write_traces(trace_output, parsed_list, wave1, wave2, final)

    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Summary
    route_counts: Counter[str] = Counter(r.route for r in wave1.values())
    path_counts: Counter[str] = Counter()
    for p in parsed_list:
        qid = p.qid
        w1 = wave1.get(qid)
        if w1 is None:
            path_counts["ckpt_restored"] += 1
        elif w1.forced:
            path_counts["forced_safety"] += 1
        elif w1.error:
            path_counts["fallback"] += 1
        elif qid in wave2:
            path_counts[f"wave_{w1.route}_sc"] += 1
        else:
            path_counts["wave_direct"] += 1

    total = time.time() - t_start
    infer = time.time() - run_start
    n = max(len(parsed_list), 1)
    print(f"Written {len(final)} predictions to {output_path}", flush=True)
    print(f"Route counts: {dict(route_counts)}", flush=True)
    print(f"Path counts:  {dict(path_counts)}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer:.1f}s, {infer / n:.2f}s/question)",
        flush=True,
    )


# ── Wave 1 ────────────────────────────────────────────────────────────────────

def _run_wave1(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    skip_qids: set[str],
) -> dict[str, _W1Result]:
    results: dict[str, _W1Result] = {}
    pending: list[tuple[ParsedQuestion, str]] = []  # (parsed, route)

    for parsed in parsed_list:
        if parsed.qid in skip_qids:
            continue
        route = route_question(parsed)
        forced = get_forced_answer(parsed, route)
        if forced is not None:
            results[parsed.qid] = _W1Result(
                qid=parsed.qid, route=route, answer=forced, margin=None, forced=True,
            )
        else:
            pending.append((parsed, route))

    if not pending:
        return results

    # Build Pass-1 reasoning prompts.
    reasoning_prompts = [_build_reasoning_prompt(p, r) for p, r in pending]

    # Two sub-batches by thinking mode (vLLM cannot mix modes in one call).
    stem_idx = [i for i, (_, r) in enumerate(pending) if r == "stem"]
    other_idx = [i for i, (_, r) in enumerate(pending) if r != "stem"]

    reasonings = [""] * len(pending)

    if stem_idx:
        outs = _batch_generate(
            agent,
            [reasoning_prompts[i] for i in stem_idx],
            mode="think",
            max_tokens=_TOK["STEM"],
            temperature=0.0,
        )
        for pos, idx in enumerate(stem_idx):
            reasonings[idx] = outs[pos]

    if other_idx:
        outs = _batch_generate(
            agent,
            [reasoning_prompts[i] for i in other_idx],
            mode="no_think",
            max_tokens=_TOK["READING"],   # max of READING/KNOWLEDGE/SAFETY
            temperature=0.0,
        )
        for pos, idx in enumerate(other_idx):
            reasonings[idx] = outs[pos]

    # Pass 2: batch extract choices from reasoning.
    extract_prompts = [
        _build_extraction_from_reasoning(reasoning_prompts[i], reasonings[i])
        for i in range(len(pending))
    ]
    choices = _batch_extract(agent, extract_prompts, [p.options for p, _ in pending])

    for i, (parsed, route) in enumerate(pending):
        try:
            ch = choices[i]
            results[parsed.qid] = _W1Result(
                qid=parsed.qid,
                route=route,
                answer=ch.letter,
                margin=ch.margin,
                reasoning_prompt=reasoning_prompts[i],
                per_letter_logprob=ch.per_letter_logprob,
            )
        except Exception as exc:
            results[parsed.qid] = _W1Result(
                qid=parsed.qid,
                route=route,
                answer=FALLBACK,
                margin=None,
                error=str(exc),
                reasoning_prompt=reasoning_prompts[i],
            )

    return results


# ── SC helpers ───────────────────────────────────────────────────────────────

def _stem_sc_n(margin: float | None, adaptive_sc: bool) -> int:
    """Return the SC sample count for a STEM question.

    adaptive_sc=True : n=SC_N_STEM["low"]=7 when margin is below the STEM
                       threshold, n=SC_N_STEM["high"]=3 otherwise.
    adaptive_sc=False: always n=SC_N_STEM["high"]=3 (faster, non-adaptive).
    """
    if adaptive_sc and margin is not None and margin < _MARGIN_LOW["STEM"]:
        return _SC_N_STEM["low"]
    return _SC_N_STEM["high"]


# ── Wave 2 ────────────────────────────────────────────────────────────────────

def _run_wave2(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, _W1Result],
    *,
    adaptive_sc: bool = True,
) -> dict[str, str]:
    # Determine which questions need SC and with what n.
    escalated: list[tuple[ParsedQuestion, str, _W1Result, int]] = []
    for parsed in parsed_list:
        w1 = wave1.get(parsed.qid)
        if w1 is None or w1.forced or w1.error:
            continue
        route_upper = w1.route.upper()
        margin = w1.margin

        if route_upper == "STEM":
            # Always escalate — never skip STEM SC (v02_delta lesson).
            sc_n = _stem_sc_n(margin, adaptive_sc)
            escalated.append((parsed, w1.route, w1, sc_n))
        elif route_upper == "READING" and _is_reason_purpose_question(parsed.query):
            escalated.append((parsed, w1.route, w1, 3))
        elif (
            route_upper == "KNOWLEDGE"
            and margin is not None
            and margin < _MARGIN_LOW["KNOWLEDGE"]
        ):
            escalated.append((parsed, w1.route, w1, _SC_N))

    if not escalated:
        return {}

    # Flatten SC samples with optional option shuffle.
    # Each entry: (qid, sc_prompt, shuffled_opts, reverse_map, is_stem)
    flat_sc: list[tuple[str, str, dict[str, str], dict[str, str], bool]] = []
    for parsed, route, _w1, sc_n in escalated:
        is_stem = route == "stem"
        for sample_idx in range(sc_n):
            shuffled_opts, reverse_map = _shuffle_options(parsed.options, sample_idx)
            sc_prompt = _build_sc_reasoning_prompt(parsed, route, shuffled_opts)
            flat_sc.append((parsed.qid, sc_prompt, shuffled_opts, reverse_map, is_stem))

    # Two SC sub-batches by mode.
    stem_sc_idx = [i for i, item in enumerate(flat_sc) if item[4]]
    other_sc_idx = [i for i, item in enumerate(flat_sc) if not item[4]]

    sc_reasonings = [""] * len(flat_sc)

    if stem_sc_idx:
        outs = _batch_generate(
            agent,
            [flat_sc[i][1] for i in stem_sc_idx],
            mode="think",
            max_tokens=_TOK["STEM"],
            temperature=_SC_TEMP,
            top_p=_SC_TOP_P,
        )
        for pos, idx in enumerate(stem_sc_idx):
            sc_reasonings[idx] = outs[pos]

    if other_sc_idx:
        outs = _batch_generate(
            agent,
            [flat_sc[i][1] for i in other_sc_idx],
            mode="no_think",
            max_tokens=_TOK["READING"],
            temperature=_SC_TEMP,
            top_p=_SC_TOP_P,
        )
        for pos, idx in enumerate(other_sc_idx):
            sc_reasonings[idx] = outs[pos]

    # Batch extract SC choices using shuffled options.
    sc_extract_prompts = [
        _build_extraction_from_reasoning(flat_sc[i][1], sc_reasonings[i])
        for i in range(len(flat_sc))
    ]
    sc_raw = _batch_extract(
        agent,
        sc_extract_prompts,
        [flat_sc[i][2] for i in range(len(flat_sc))],
    )

    # Remap shuffled labels → original labels and group by qid.
    qid_choices: dict[str, list[ChoiceResult]] = defaultdict(list)
    for i, (qid, _sc_prompt, _shuffled, reverse_map, _is_stem) in enumerate(flat_sc):
        try:
            raw = sc_raw[i]
            orig_letter = reverse_map[raw.letter]
            orig_logprobs = {
                reverse_map.get(sl, sl): lp
                for sl, lp in raw.per_letter_logprob.items()
            }
            qid_choices[qid].append(
                ChoiceResult(
                    letter=orig_letter,
                    margin=raw.margin,
                    per_letter_logprob=orig_logprobs,
                )
            )
        except Exception:
            pass  # one failed sample is skipped; vote continues with the rest

    # Majority vote per qid.
    wave2: dict[str, str] = {}
    for parsed, _route, w1, _sc_n in escalated:
        qid = parsed.qid
        choices = qid_choices.get(qid, [])
        if not choices:
            # All SC samples failed — keep Wave 1 answer.
            wave2[qid] = w1.answer
            continue
        first_choice: ChoiceResult | None = None
        if w1.per_letter_logprob:
            first_choice = ChoiceResult(
                letter=w1.answer,
                margin=w1.margin or 0.0,
                per_letter_logprob=w1.per_letter_logprob,
            )
        vote = _vote(choices, first_choice)
        wave2[qid] = vote.letter

    return wave2


# ── Batch generation ──────────────────────────────────────────────────────────

def _batch_generate(
    agent: ReasoningAgent,
    prompts: list[str],
    *,
    mode: str = "no_think",
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> list[str]:
    if not prompts:
        return []
    return agent.generate_freeform(
        prompts,
        mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


# ── Batch guided-choice extraction ────────────────────────────────────────────

def _batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    """Batch Pass-2 guided-choice extraction.

    For vLLM: uses engine.generate() with per-request allowed_token_ids so
    all questions are scheduled in one vLLM call.  For HF: sequential fallback.
    """
    if not prompts:
        return []
    if agent.is_vllm:
        return _vllm_batch_extract(agent, prompts, options_list)
    return _hf_batch_extract(agent, prompts, options_list)


def _vllm_batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    from vllm import SamplingParams

    tokenizer = agent.tokenizer
    token_maps = [
        build_label_token_map(tokenizer, sorted(opts.keys()))
        for opts in options_list
    ]
    sampling_params_list = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            top_p=1.0,
            logprobs=min(len(tm), 64),
            allowed_token_ids=list(tm.keys()),
        )
        for tm in token_maps
    ]

    # engine.generate() accepts a list of per-request SamplingParams.
    raw_outputs = agent._llm.engine.generate(prompts, sampling_params_list)

    results: list[ChoiceResult] = []
    for i, output in enumerate(raw_outputs):
        try:
            scores: dict[str, float] = {lbl: float("-inf") for lbl in options_list[i]}
            logprobs_seq = getattr(output.outputs[0], "logprobs", None) or []
            if logprobs_seq:
                for token_id, entry in logprobs_seq[0].items():
                    lbl = token_maps[i].get(int(token_id))
                    if lbl is None:
                        continue
                    lp = _get_logprob(entry)
                    if lp is not None:
                        scores[lbl] = float(lp)
            results.append(
                ChoiceResult(
                    letter=best_label(scores),
                    margin=softmax_margin(scores),
                    per_letter_logprob=scores,
                )
            )
        except Exception:
            first_lbl = sorted(options_list[i].keys())[0]
            scores = {lbl: float("-inf") for lbl in options_list[i]}
            scores[first_lbl] = 0.0
            results.append(ChoiceResult(letter=first_lbl, margin=0.0, per_letter_logprob=scores))

    return results


def _hf_batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    results: list[ChoiceResult] = []
    for prompt, opts in zip(prompts, options_list):
        valid_labels = tuple(sorted(opts.keys()))
        try:
            scores = agent.score_valid_labels(prompt, valid_labels)
            results.append(
                ChoiceResult(
                    letter=best_label(scores),
                    margin=softmax_margin(scores),
                    per_letter_logprob=scores,
                )
            )
        except Exception:
            first_lbl = valid_labels[0]
            scores = {lbl: float("-inf") for lbl in valid_labels}
            scores[first_lbl] = 0.0
            results.append(ChoiceResult(letter=first_lbl, margin=0.0, per_letter_logprob=scores))
    return results


def _get_logprob(entry: Any) -> float | None:
    """Extract the logprob float from a vLLM token-logprob entry object."""
    lp = getattr(entry, "logprob", None)
    if lp is not None:
        return float(lp)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (float, int)):
        return float(entry)
    return None


# ── Option shuffle for SC de-bias ─────────────────────────────────────────────

def _shuffle_options(
    options: dict[str, str],
    sample_idx: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Shuffle option values across labels, return (shuffled, reverse_map).

    reverse_map[new_label] = original_label so voted letters can be remapped.
    With SHUFFLE_OPTIONS=False this is a no-op identity.
    """
    if not _SHUFFLE_OPTIONS:
        identity = {k: k for k in options}
        return options, identity

    labels = sorted(options.keys())
    values = [options[lbl] for lbl in labels]
    random.Random(_SC_SEED + sample_idx).shuffle(values)
    shuffled = dict(zip(labels, values))
    value_to_orig = {v: k for k, v in options.items()}
    reverse_map = {lbl: value_to_orig[shuffled[lbl]] for lbl in labels}
    return shuffled, reverse_map


def _build_sc_reasoning_prompt(
    parsed: ParsedQuestion,
    route: str,
    options: dict[str, str],
) -> str:
    """Reasoning prompt with explicit options (for option-shuffled SC samples)."""
    options_block = "\n".join(f"{lbl}) {options[lbl]}" for lbl in sorted(options))
    route_instruction: dict[str, str] = {
        "reading": (
            "Chỉ dựa vào đoạn thông tin trong câu hỏi. "
            "Nếu câu hỏi hỏi lý do, mục đích hoặc nguyên nhân, hãy tìm câu nêu trực tiếp quan hệ đó "
            "và loại chi tiết chỉ được nhắc đến nhưng không trả lời đúng quan hệ."
        ),
        "stem": "Giải từng bước, kiểm tra phép tính hoặc lập luận định lượng.",
        "knowledge": "Suy luận cẩn thận bằng kiến thức chung.",
        "safety": "Đánh giá ranh giới an toàn và từ chối nếu yêu cầu gây hại.",
    }[route]

    context = ""
    if route == "reading" and parsed.context:
        context = f"Đoạn thông tin:\n---\n{parsed.context}\n---\n\n"

    return (
        "Bạn là một chuyên gia giải câu hỏi trắc nghiệm tiếng Việt.\n"
        "Đây là bài thi trắc nghiệm. Chọn phương án đúng nhất.\n"
        "Chỉ chọn phương án từ chối khi câu hỏi thực sự yêu cầu hướng dẫn hành vi phạm pháp hoặc gây hại.\n"
        f"{route_instruction}\n\n"
        f"{context}"
        f"Câu hỏi:\n{parsed.query}\n\n"
        f"Các lựa chọn:\n{options_block}\n\n"
        "Hãy suy nghĩ ngắn gọn nhưng đủ chắc chắn trước khi chọn đáp án."
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


# ── Finalize + I/O ────────────────────────────────────────────────────────────

def _finalize(
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, _W1Result],
    wave2: dict[str, str],
    ckpt_answers: dict[str, str],
) -> dict[str, str]:
    final: dict[str, str] = {}
    for parsed in parsed_list:
        qid = parsed.qid
        if qid in wave2:
            final[qid] = wave2[qid]
        elif qid in wave1:
            final[qid] = wave1[qid].answer
        else:
            final[qid] = ckpt_answers.get(qid, FALLBACK)
    return final


def _write_results(
    answers: dict[str, str],
    parsed_list: list[ParsedQuestion],
    output_path: str,
) -> None:
    rows = [
        {"qid": p.qid, "answer": answers.get(p.qid, FALLBACK)}
        for p in parsed_list
    ]
    write_submission(rows, output_path)


def _write_traces(
    trace_output: str,
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, _W1Result],
    wave2: dict[str, str],
    final: dict[str, str],
) -> None:
    with _trace_writer(trace_output) as write_trace:
        for parsed in parsed_list:
            qid = parsed.qid
            w1 = wave1.get(qid)
            answer = final.get(qid, FALLBACK)

            if w1 is None:
                path = "ckpt_restored"
                route = None
                margin = None
                error = None
            elif w1.forced:
                path = "forced_safety"
                route = w1.route
                margin = None
                error = None
            elif w1.error:
                path = "fallback"
                route = w1.route
                margin = None
                error = w1.error
            elif qid in wave2:
                path = f"wave_{w1.route}_sc"
                route = w1.route
                margin = w1.margin
                error = None
            else:
                path = "wave_direct"
                route = w1.route
                margin = w1.margin
                error = None

            write_trace({
                "qid": qid,
                "answer": answer,
                "route": route,
                "path": path,
                "margin": margin,
                "first_answer": w1.answer if w1 else None,
                "votes": [],
                "layer1_route": route,
                "semantic_route": None,
                "route_override": False,
                "override_blockers": [],
                "rag_used": False,
                "rag_top_score": None,
                "error": error,
            })


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
