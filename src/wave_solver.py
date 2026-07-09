"""Wave-batched solver used by the final v03_gamma runner.

The wave solver keeps vLLM busy by batching all first-pass reasoning/extraction
and then batching all self-consistency escalations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import time

from src.batch_extract import batch_extract
from src.config import FALLBACK, MAX_MODEL_LEN
from src.data_loader import write_submission
from src.extract import ChoiceResult
from src.llm import GenerationOutput
from src.parser import ParsedQuestion
from src.reasoning_agent import ReasoningAgent
from src.router import get_forced_answer, route_question
from src.sc_policy import (
    SC_N_HIGH_CHOICE_KNOWLEDGE,
    SC_N_DEFAULT,
    SC_TEMP,
    SC_TOP_P,
    TOKENS_BY_ROUTE,
    WAVE2_THINK_TOKENS_BY_ROUTE,
    build_sc_reasoning_prompt,
    duplicate_option_label_map,
    knowledge_escalation_reason,
    reading_escalation_reason,
    should_use_think_mode,
    shuffle_options,
    stem_sc_n,
)
from src.solve import (
    _build_extraction_from_reasoning,
    _build_reasoning_prompt,
    _vote,
)
from src.version_runner import _trace_writer

_EXTRACTION_TOKEN_BUFFER = 8


def _canonicalize_scores_for_duplicates(
    options: dict[str, str],
    scores: dict[str, float],
) -> dict[str, float]:
    duplicate_map = duplicate_option_label_map(options)
    if len(set(duplicate_map.values())) == len(duplicate_map):
        return scores

    canonical_scores = {label: float("-inf") for label in options}
    for label, score in scores.items():
        canonical_label = duplicate_map.get(label, label)
        canonical_scores[canonical_label] = max(canonical_scores[canonical_label], score)
    return canonical_scores


def _canonicalize_choice_for_duplicates(
    options: dict[str, str],
    choice: ChoiceResult,
) -> ChoiceResult:
    duplicate_map = duplicate_option_label_map(options)
    canonical_letter = duplicate_map.get(choice.letter, choice.letter)
    canonical_scores = _canonicalize_scores_for_duplicates(options, choice.per_letter_logprob)
    return ChoiceResult(
        letter=canonical_letter,
        margin=choice.margin,
        per_letter_logprob=canonical_scores,
    )


@dataclass
class Wave1Result:
    """Per-question result from Wave 1."""

    qid: str
    route: str
    answer: str
    margin: float | None
    forced: bool = False
    error: str | None = None
    reasoning_prompt: str = ""
    per_letter_logprob: dict[str, float] = field(default_factory=dict)
    gen_tokens_wave1: int = 0
    wave1_time_share: float = 0.0
    wave1_think: bool = False


@dataclass
class Wave2Result:
    """Per-question result from Wave 2 (SC escalation)."""

    answer: str
    votes: list[str] = field(default_factory=list)
    escalation_reason: str = ""
    gen_tokens_wave2: int = 0
    wave2_sample_tokens: list[int] = field(default_factory=list)
    wave2_time_share: float = 0.0
    wave2_think: bool = False
    sc_n: int = 0


def _build_compact_extraction_prompt(
    question: str,
    options: dict[str, str],
    reasoning: str,
    *,
    route: str,
) -> str:
    options_block = "\n".join(f"{label}) {options[label]}" for label in sorted(options))
    route_instruction = {
        "reading": (
            "Chỉ chọn đáp án được lời giải nháp hỗ trợ trực tiếp từ đoạn thông tin đã đọc."
        ),
        "stem": "Ưu tiên đáp án khớp với phép tính hoặc lập luận cuối cùng trong lời giải nháp.",
        "knowledge": "Ưu tiên đáp án khớp trực tiếp nhất với kết luận cuối cùng trong lời giải nháp.",
        "safety": "Nếu lời giải nháp kết luận cần từ chối, chọn đúng phương án từ chối.",
    }[route]
    return (
        "Bạn đang chốt đáp án trắc nghiệm tiếng Việt.\n"
        f"{route_instruction}\n\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Các lựa chọn:\n{options_block}\n\n"
        "Lời giải nháp rút gọn:\n"
        f"{reasoning}\n\n"
        "Chọn đúng một đáp án hợp lệ.\n"
        "Đáp án: "
    )


def _agent_max_input_tokens(agent: ReasoningAgent) -> int:
    llm = getattr(agent, "_llm", None)
    if llm is not None and getattr(llm, "max_model_len", None):
        return int(llm.max_model_len)

    model = getattr(agent, "_model", None)
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "n_positions", "model_max_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    return MAX_MODEL_LEN


def _encode_prompt(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode_prompt(tokenizer, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return ""
    try:
        return decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return decode(token_ids)


def _fit_extraction_prompt(
    agent: ReasoningAgent,
    *,
    route: str,
    question: str,
    options: dict[str, str],
    reasoning_prompt: str,
    reasoning: str,
) -> str:
    tokenizer = agent.tokenizer
    max_input_tokens = _agent_max_input_tokens(agent) - _EXTRACTION_TOKEN_BUFFER

    generic_builder = lambda draft: _build_extraction_from_reasoning(reasoning_prompt, draft)
    compact_builder = lambda draft: _build_compact_extraction_prompt(
        question,
        options,
        draft,
        route=route,
    )

    prompt = generic_builder(reasoning)
    if len(_encode_prompt(tokenizer, prompt, add_special_tokens=True)) <= max_input_tokens:
        return prompt

    builder = compact_builder if route == "reading" else generic_builder
    base_prompt = builder("")
    base_tokens = _encode_prompt(tokenizer, base_prompt, add_special_tokens=True)
    if len(base_tokens) > max_input_tokens:
        return base_prompt

    reasoning_tokens = _encode_prompt(tokenizer, reasoning, add_special_tokens=False)
    budget = max_input_tokens - len(base_tokens)
    if budget <= 0:
        return base_prompt
    if len(reasoning_tokens) <= budget:
        return builder(reasoning)

    trimmed_reasoning = _decode_prompt(tokenizer, reasoning_tokens[-budget:]).strip()
    if trimmed_reasoning:
        return builder(trimmed_reasoning)
    return base_prompt


def _normalize_generation_outputs(outputs: list[object]) -> list[GenerationOutput]:
    normalized: list[GenerationOutput] = []
    for output in outputs:
        if isinstance(output, GenerationOutput):
            normalized.append(output)
        else:
            normalized.append(GenerationOutput(text=str(output)))
    return normalized


def _allocate_time_shares(
    qids: list[str],
    token_counts: list[int],
    total_elapsed: float,
) -> dict[str, float]:
    if not qids:
        return {}

    per_qid_tokens: dict[str, int] = defaultdict(int)
    for qid, token_count in zip(qids, token_counts):
        per_qid_tokens[qid] += max(token_count, 0)

    total_tokens = sum(per_qid_tokens.values())
    if total_tokens > 0:
        return {
            qid: total_elapsed * (count / total_tokens)
            for qid, count in per_qid_tokens.items()
        }

    even_share = total_elapsed / len(per_qid_tokens)
    return {qid: even_share for qid in per_qid_tokens}


def _iter_chunks(items: list[object], chunk_size: int | None) -> list[list[object]]:
    if chunk_size is None or chunk_size <= 0 or len(items) <= chunk_size:
        return [items]
    return [items[start:start + chunk_size] for start in range(0, len(items), chunk_size)]


def _run_wave1_pending(
    agent: ReasoningAgent,
    pending: list[tuple[ParsedQuestion, str]],
) -> dict[str, Wave1Result]:
    chunk_results: dict[str, Wave1Result] = {}
    if not pending:
        return chunk_results

    wave_start = time.perf_counter()
    reasoning_prompts = [_build_reasoning_prompt(parsed, route) for parsed, route in pending]
    think_idx = [
        i for i, (parsed, route) in enumerate(pending)
        if should_use_think_mode(parsed, route, stage="wave1")
    ]
    other_idx = [
        i for i, (parsed, route) in enumerate(pending)
        if not should_use_think_mode(parsed, route, stage="wave1")
    ]

    reasoning_outputs = [GenerationOutput(text="")] * len(pending)

    if think_idx:
        outputs = _normalize_generation_outputs(
            batch_generate(
                agent,
                [reasoning_prompts[i] for i in think_idx],
                mode="think",
                max_tokens=TOKENS_BY_ROUTE["STEM"],
                temperature=0.0,
            )
        )
        for pos, idx in enumerate(think_idx):
            reasoning_outputs[idx] = outputs[pos]

    if other_idx:
        outputs = _normalize_generation_outputs(
            batch_generate(
                agent,
                [reasoning_prompts[i] for i in other_idx],
                mode="no_think",
                max_tokens=TOKENS_BY_ROUTE["READING"],
                temperature=0.0,
            )
        )
        for pos, idx in enumerate(other_idx):
            reasoning_outputs[idx] = outputs[pos]

    reasonings = [output.text for output in reasoning_outputs]
    token_counts = [output.num_generated_tokens or 0 for output in reasoning_outputs]
    time_shares = _allocate_time_shares(
        [parsed.qid for parsed, _route in pending],
        token_counts,
        time.perf_counter() - wave_start,
    )

    extract_prompts = [
        _fit_extraction_prompt(
            agent,
            route=pending[i][1],
            question=pending[i][0].query,
            options=pending[i][0].options,
            reasoning_prompt=reasoning_prompts[i],
            reasoning=reasonings[i],
        )
        for i in range(len(pending))
    ]
    choices = batch_extract(agent, extract_prompts, [parsed.options for parsed, _ in pending])

    for i, (parsed, route) in enumerate(pending):
        try:
            choice = choices[i]
            chunk_results[parsed.qid] = Wave1Result(
                qid=parsed.qid,
                route=route,
                answer=choice.letter,
                margin=choice.margin,
                reasoning_prompt=reasoning_prompts[i],
                per_letter_logprob=choice.per_letter_logprob,
                gen_tokens_wave1=token_counts[i],
                wave1_time_share=time_shares.get(parsed.qid, 0.0),
                wave1_think=i in think_idx,
            )
        except Exception as exc:
            chunk_results[parsed.qid] = Wave1Result(
                qid=parsed.qid,
                route=route,
                answer=FALLBACK,
                margin=None,
                error=str(exc),
                reasoning_prompt=reasoning_prompts[i],
                gen_tokens_wave1=token_counts[i],
                wave1_time_share=time_shares.get(parsed.qid, 0.0),
                wave1_think=i in think_idx,
            )

    return chunk_results


def run_wave1(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    skip_qids: set[str],
    *,
    chunk_size: int | None = None,
) -> dict[str, Wave1Result]:
    """Batch all first-pass route reasoning and guided-choice extraction."""
    results: dict[str, Wave1Result] = {}
    pending: list[tuple[ParsedQuestion, str]] = []

    for parsed in parsed_list:
        if parsed.qid in skip_qids:
            continue
        route = route_question(parsed)
        forced = get_forced_answer(parsed, route)
        if forced is not None:
            results[parsed.qid] = Wave1Result(
                qid=parsed.qid,
                route=route,
                answer=forced,
                margin=None,
                forced=True,
            )
        else:
            pending.append((parsed, route))

    if not pending:
        return results
    for pending_chunk in _iter_chunks(pending, chunk_size):
        results.update(_run_wave1_pending(agent, pending_chunk))

    return results


def run_wave2(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, Wave1Result],
    *,
    adaptive_sc: bool = True,
    chunk_size: int | None = None,
) -> dict[str, Wave2Result]:
    """Batch all self-consistency escalations and return final SC answers."""
    escalated: list[tuple[ParsedQuestion, str, Wave1Result, int, str]] = []
    for parsed in parsed_list:
        w1 = wave1.get(parsed.qid)
        if w1 is None or w1.forced or w1.error:
            continue
        route_upper = w1.route.upper()
        margin = w1.margin

        if route_upper == "STEM":
            sc_n = stem_sc_n(margin, adaptive_sc)
            reason = f"stem_sc_adaptive_n{sc_n}" if adaptive_sc else f"stem_sc_fixed_n{sc_n}"
            escalated.append((parsed, w1.route, w1, sc_n, reason))
        elif route_upper == "READING":
            reason = reading_escalation_reason(parsed.query)
            if reason is not None:
                escalated.append((parsed, w1.route, w1, 3, reason))
        elif route_upper == "KNOWLEDGE":
            reason = knowledge_escalation_reason(parsed, margin)
            if reason is not None:
                sc_n = (
                    SC_N_DEFAULT
                    if reason.startswith("knowledge_low_margin_")
                    else SC_N_HIGH_CHOICE_KNOWLEDGE
                )
                escalated.append((parsed, w1.route, w1, sc_n, reason))

    if not escalated:
        return {}

    wave2: dict[str, Wave2Result] = {}
    for escalated_chunk in _iter_chunks(escalated, chunk_size):
        wave2.update(_run_wave2_escalated(agent, escalated_chunk))
    return wave2


def _run_wave2_escalated(
    agent: ReasoningAgent,
    escalated: list[tuple[ParsedQuestion, str, Wave1Result, int, str]],
) -> dict[str, Wave2Result]:
    wave2: dict[str, Wave2Result] = {}
    if not escalated:
        return wave2

    wave_start = time.perf_counter()
    flat_sc: list[
        tuple[str, str, str, str, dict[str, str], dict[str, str], bool]
    ] = []
    for parsed, route, _w1, sc_n, _reason in escalated:
        use_think = should_use_think_mode(parsed, route, stage="wave2")
        for sample_idx in range(sc_n):
            shuffled_options, reverse_map = shuffle_options(parsed.options, sample_idx)
            sc_prompt = build_sc_reasoning_prompt(parsed, route, shuffled_options)
            flat_sc.append(
                (
                    parsed.qid,
                    route,
                    parsed.query,
                    sc_prompt,
                    shuffled_options,
                    reverse_map,
                    use_think,
                )
            )

    reading_think_sc_idx = [
        i for i, item in enumerate(flat_sc)
        if item[6] and item[1] == "reading"
    ]
    other_think_sc_idx = [
        i for i, item in enumerate(flat_sc)
        if item[6] and item[1] != "reading"
    ]
    other_sc_idx = [i for i, item in enumerate(flat_sc) if not item[6]]

    sc_outputs = [GenerationOutput(text="")] * len(flat_sc)

    if other_think_sc_idx:
        outputs = _normalize_generation_outputs(
            batch_generate(
                agent,
                [flat_sc[i][3] for i in other_think_sc_idx],
                mode="think",
                max_tokens=TOKENS_BY_ROUTE["STEM"],
                temperature=SC_TEMP,
                top_p=SC_TOP_P,
            )
        )
        for pos, idx in enumerate(other_think_sc_idx):
            sc_outputs[idx] = outputs[pos]

    if reading_think_sc_idx:
        outputs = _normalize_generation_outputs(
            batch_generate(
                agent,
                [flat_sc[i][3] for i in reading_think_sc_idx],
                mode="think",
                max_tokens=WAVE2_THINK_TOKENS_BY_ROUTE.get("READING", TOKENS_BY_ROUTE["STEM"]),
                temperature=SC_TEMP,
                top_p=SC_TOP_P,
            )
        )
        for pos, idx in enumerate(reading_think_sc_idx):
            sc_outputs[idx] = outputs[pos]

    if other_sc_idx:
        outputs = _normalize_generation_outputs(
            batch_generate(
                agent,
                [flat_sc[i][3] for i in other_sc_idx],
                mode="no_think",
                max_tokens=TOKENS_BY_ROUTE["READING"],
                temperature=SC_TEMP,
                top_p=SC_TOP_P,
            )
        )
        for pos, idx in enumerate(other_sc_idx):
            sc_outputs[idx] = outputs[pos]

    sc_reasonings = [output.text for output in sc_outputs]
    sample_token_counts = [output.num_generated_tokens or 0 for output in sc_outputs]
    wave2_time_shares = _allocate_time_shares(
        [item[0] for item in flat_sc],
        sample_token_counts,
        time.perf_counter() - wave_start,
    )

    sc_extract_prompts = [
        _fit_extraction_prompt(
            agent,
            route=flat_sc[i][1],
            question=flat_sc[i][2],
            options=flat_sc[i][4],
            reasoning_prompt=flat_sc[i][3],
            reasoning=sc_reasonings[i],
        )
        for i in range(len(flat_sc))
    ]
    sc_raw = batch_extract(
        agent,
        sc_extract_prompts,
        [flat_sc[i][4] for i in range(len(flat_sc))],
    )

    qid_choices: dict[str, list[ChoiceResult]] = defaultdict(list)
    for i, (qid, _route, _query, _prompt, options, reverse_map, _use_think) in enumerate(flat_sc):
        try:
            raw = sc_raw[i]
            original_letter = reverse_map[raw.letter]
            original_options = {
                reverse_map.get(shuffled_label, shuffled_label): text
                for shuffled_label, text in options.items()
            }
            original_logprobs = {
                reverse_map.get(shuffled_label, shuffled_label): logprob
                for shuffled_label, logprob in raw.per_letter_logprob.items()
            }
            qid_choices[qid].append(
                _canonicalize_choice_for_duplicates(
                    original_options,
                    ChoiceResult(
                        letter=original_letter,
                        margin=raw.margin,
                        per_letter_logprob=original_logprobs,
                    ),
                )
            )
        except Exception:
            pass

    qid_sample_tokens: dict[str, list[int]] = defaultdict(list)
    for i, (qid, _route, _query, _prompt, _options, _reverse_map, _use_think) in enumerate(flat_sc):
        qid_sample_tokens[qid].append(sample_token_counts[i])

    for parsed, _route, w1, _sc_n, esc_reason in escalated:
        choices = qid_choices.get(parsed.qid, [])
        use_think = should_use_think_mode(parsed, w1.route, stage="wave2")
        sample_tokens = qid_sample_tokens.get(parsed.qid, [])
        total_tokens = sum(sample_tokens)
        if not choices:
            wave2[parsed.qid] = Wave2Result(
                answer=w1.answer,
                votes=[],
                escalation_reason=esc_reason,
                gen_tokens_wave2=total_tokens,
                wave2_sample_tokens=sample_tokens,
                wave2_time_share=wave2_time_shares.get(parsed.qid, 0.0),
                wave2_think=use_think,
                sc_n=_sc_n,
            )
            continue
        first_choice: ChoiceResult | None = None
        if w1.per_letter_logprob:
            first_choice = ChoiceResult(
                letter=w1.answer,
                margin=w1.margin or 0.0,
                per_letter_logprob=w1.per_letter_logprob,
            )
        vote = _vote(choices, first_choice)
        wave2[parsed.qid] = Wave2Result(
            answer=vote.letter,
            votes=vote.votes,
            escalation_reason=esc_reason,
            gen_tokens_wave2=total_tokens,
            wave2_sample_tokens=sample_tokens,
            wave2_time_share=wave2_time_shares.get(parsed.qid, 0.0),
            wave2_think=use_think,
            sc_n=_sc_n,
        )

    return wave2


def batch_generate(
    agent: ReasoningAgent,
    prompts: list[str],
    *,
    mode: str = "no_think",
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> list[str]:
    """Generate a batch of free-form reasoning completions."""
    if not prompts:
        return []
    outputs = agent.generate_freeform(
        prompts,
        mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        return_outputs=True,
    )
    return _normalize_generation_outputs(outputs)


def finalize_answers(
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, Wave1Result],
    wave2: dict[str, Wave2Result],
    ckpt_answers: dict[str, str],
) -> dict[str, str]:
    """Merge checkpoint, Wave 1, and Wave 2 answers into a complete answer map."""
    final: dict[str, str] = {}
    for parsed in parsed_list:
        qid = parsed.qid
        if qid in wave2:
            final[qid] = wave2[qid].answer
        elif qid in wave1:
            final[qid] = wave1[qid].answer
        else:
            final[qid] = ckpt_answers.get(qid, FALLBACK)
    return final


def write_results(
    answers: dict[str, str],
    parsed_list: list[ParsedQuestion],
    output_path: str,
) -> None:
    rows = [
        {"qid": parsed.qid, "answer": answers.get(parsed.qid, FALLBACK)}
        for parsed in parsed_list
    ]
    write_submission(rows, output_path)


def path_counts(
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, Wave1Result],
    wave2: dict[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for parsed in parsed_list:
        w1 = wave1.get(parsed.qid)
        if w1 is None:
            counts["ckpt_restored"] += 1
        elif w1.forced:
            counts["forced_safety"] += 1
        elif w1.error:
            counts["fallback"] += 1
        elif parsed.qid in wave2:
            counts[f"wave_{w1.route}_sc"] += 1
        else:
            counts["wave_direct"] += 1
    return counts


def write_traces(
    trace_output: str,
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, Wave1Result],
    wave2: dict[str, Wave2Result],
    final: dict[str, str],
    runtime_info: dict[str, object] | None = None,
) -> None:
    with _trace_writer(trace_output) as write_trace:
        for parsed in parsed_list:
            qid = parsed.qid
            w1 = wave1.get(qid)
            w2 = wave2.get(qid)
            answer = final.get(qid, FALLBACK)
            runtime = dict(runtime_info or {})

            if w1 is None:
                path = "ckpt_restored"
                route = None
                margin = None
                error = None
                gen_tokens_wave1 = 0
                wave1_time_share = 0.0
                wave1_think = None
            elif w1.forced:
                path = "forced_safety"
                route = w1.route
                margin = None
                error = None
                gen_tokens_wave1 = 0
                wave1_time_share = 0.0
                wave1_think = False
            elif w1.error:
                path = "fallback"
                route = w1.route
                margin = None
                error = w1.error
                gen_tokens_wave1 = w1.gen_tokens_wave1
                wave1_time_share = w1.wave1_time_share
                wave1_think = w1.wave1_think
            elif w2 is not None:
                path = f"wave_{w1.route}_sc"
                route = w1.route
                margin = w1.margin
                error = None
                gen_tokens_wave1 = w1.gen_tokens_wave1
                wave1_time_share = w1.wave1_time_share
                wave1_think = w1.wave1_think
            else:
                path = "wave_direct"
                route = w1.route
                margin = w1.margin
                error = None
                gen_tokens_wave1 = w1.gen_tokens_wave1
                wave1_time_share = w1.wave1_time_share
                wave1_think = w1.wave1_think

            gen_tokens_wave2 = w2.gen_tokens_wave2 if w2 else 0
            wave2_time_share = w2.wave2_time_share if w2 else 0.0
            wave2_think = w2.wave2_think if w2 else None
            sc_n = w2.sc_n if w2 else 0
            think = wave2_think if w2 is not None else wave1_think
            attributed_time_seconds = wave1_time_share + wave2_time_share

            write_trace(
                {
                    "qid": qid,
                    "answer": answer,
                    "route": route,
                    "path": path,
                    "margin": margin,
                    "first_answer": w1.answer if w1 else None,
                    "votes": w2.votes if w2 else [],
                    "escalation_reason": w2.escalation_reason if w2 else None,
                    "layer1_route": route,
                    "semantic_route": None,
                    "route_override": False,
                    "override_blockers": [],
                    "rag_used": False,
                    "rag_top_score": None,
                    "error": error,
                    "gen_tokens_wave1": gen_tokens_wave1,
                    "gen_tokens_wave2": gen_tokens_wave2,
                    "wave2_sample_tokens": w2.wave2_sample_tokens if w2 else [],
                    "sc_n": sc_n,
                    "think": think,
                    "wave1_think": wave1_think,
                    "wave2_think": wave2_think,
                    "wave1_time_share": round(wave1_time_share, 6),
                    "wave2_time_share": round(wave2_time_share, 6),
                    "attributed_time_seconds": round(attributed_time_seconds, 6),
                    "backend": runtime.get("backend"),
                    "backend_reason": runtime.get("backend_reason"),
                    "runtime_info": runtime,
                }
            )
