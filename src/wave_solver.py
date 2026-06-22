"""Wave-batched solver used by v02_gamma.

The wave solver keeps vLLM busy by batching all first-pass reasoning/extraction
and then batching all self-consistency escalations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.batch_extract import batch_extract
from src.config import FALLBACK
from src.data_loader import write_submission
from src.extract import ChoiceResult
from src.parser import ParsedQuestion
from src.reasoning_agent import ReasoningAgent
from src.router import get_forced_answer, route_question
from src.sc_policy import (
    SC_N_HIGH_CHOICE_KNOWLEDGE,
    SC_N_DEFAULT,
    SC_TEMP,
    SC_TOP_P,
    TOKENS_BY_ROUTE,
    build_sc_reasoning_prompt,
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


@dataclass
class Wave2Result:
    """Per-question result from Wave 2 (SC escalation)."""

    answer: str
    votes: list[str] = field(default_factory=list)
    escalation_reason: str = ""


def run_wave1(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    skip_qids: set[str],
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

    reasoning_prompts = [_build_reasoning_prompt(parsed, route) for parsed, route in pending]
    think_idx = [
        i for i, (parsed, route) in enumerate(pending)
        if should_use_think_mode(parsed, route, stage="wave1")
    ]
    other_idx = [
        i for i, (parsed, route) in enumerate(pending)
        if not should_use_think_mode(parsed, route, stage="wave1")
    ]

    reasonings = [""] * len(pending)

    if think_idx:
        outputs = batch_generate(
            agent,
            [reasoning_prompts[i] for i in think_idx],
            mode="think",
            max_tokens=TOKENS_BY_ROUTE["STEM"],
            temperature=0.0,
        )
        for pos, idx in enumerate(think_idx):
            reasonings[idx] = outputs[pos]

    if other_idx:
        outputs = batch_generate(
            agent,
            [reasoning_prompts[i] for i in other_idx],
            mode="no_think",
            max_tokens=TOKENS_BY_ROUTE["READING"],
            temperature=0.0,
        )
        for pos, idx in enumerate(other_idx):
            reasonings[idx] = outputs[pos]

    extract_prompts = [
        _build_extraction_from_reasoning(reasoning_prompts[i], reasonings[i])
        for i in range(len(pending))
    ]
    choices = batch_extract(agent, extract_prompts, [parsed.options for parsed, _ in pending])

    for i, (parsed, route) in enumerate(pending):
        try:
            choice = choices[i]
            results[parsed.qid] = Wave1Result(
                qid=parsed.qid,
                route=route,
                answer=choice.letter,
                margin=choice.margin,
                reasoning_prompt=reasoning_prompts[i],
                per_letter_logprob=choice.per_letter_logprob,
            )
        except Exception as exc:
            results[parsed.qid] = Wave1Result(
                qid=parsed.qid,
                route=route,
                answer=FALLBACK,
                margin=None,
                error=str(exc),
                reasoning_prompt=reasoning_prompts[i],
            )

    return results


def run_wave2(
    agent: ReasoningAgent,
    parsed_list: list[ParsedQuestion],
    wave1: dict[str, Wave1Result],
    *,
    adaptive_sc: bool = True,
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

    flat_sc: list[tuple[str, str, dict[str, str], dict[str, str], bool]] = []
    for parsed, route, _w1, sc_n, _reason in escalated:
        use_think = should_use_think_mode(parsed, route, stage="wave2")
        for sample_idx in range(sc_n):
            shuffled_options, reverse_map = shuffle_options(parsed.options, sample_idx)
            sc_prompt = build_sc_reasoning_prompt(parsed, route, shuffled_options)
            flat_sc.append((parsed.qid, sc_prompt, shuffled_options, reverse_map, use_think))

    think_sc_idx = [i for i, item in enumerate(flat_sc) if item[4]]
    other_sc_idx = [i for i, item in enumerate(flat_sc) if not item[4]]

    sc_reasonings = [""] * len(flat_sc)

    if think_sc_idx:
        outputs = batch_generate(
            agent,
            [flat_sc[i][1] for i in think_sc_idx],
            mode="think",
            max_tokens=TOKENS_BY_ROUTE["STEM"],
            temperature=SC_TEMP,
            top_p=SC_TOP_P,
        )
        for pos, idx in enumerate(think_sc_idx):
            sc_reasonings[idx] = outputs[pos]

    if other_sc_idx:
        outputs = batch_generate(
            agent,
            [flat_sc[i][1] for i in other_sc_idx],
            mode="no_think",
            max_tokens=TOKENS_BY_ROUTE["READING"],
            temperature=SC_TEMP,
            top_p=SC_TOP_P,
        )
        for pos, idx in enumerate(other_sc_idx):
            sc_reasonings[idx] = outputs[pos]

    sc_extract_prompts = [
        _build_extraction_from_reasoning(flat_sc[i][1], sc_reasonings[i])
        for i in range(len(flat_sc))
    ]
    sc_raw = batch_extract(
        agent,
        sc_extract_prompts,
        [flat_sc[i][2] for i in range(len(flat_sc))],
    )

    qid_choices: dict[str, list[ChoiceResult]] = defaultdict(list)
    for i, (qid, _prompt, _options, reverse_map, _is_stem) in enumerate(flat_sc):
        try:
            raw = sc_raw[i]
            original_letter = reverse_map[raw.letter]
            original_logprobs = {
                reverse_map.get(shuffled_label, shuffled_label): logprob
                for shuffled_label, logprob in raw.per_letter_logprob.items()
            }
            qid_choices[qid].append(
                ChoiceResult(
                    letter=original_letter,
                    margin=raw.margin,
                    per_letter_logprob=original_logprobs,
                )
            )
        except Exception:
            pass

    wave2: dict[str, Wave2Result] = {}
    for parsed, _route, w1, _sc_n, esc_reason in escalated:
        choices = qid_choices.get(parsed.qid, [])
        if not choices:
            wave2[parsed.qid] = Wave2Result(
                answer=w1.answer,
                votes=[],
                escalation_reason=esc_reason,
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
    return agent.generate_freeform(
        prompts,
        mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


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
) -> None:
    with _trace_writer(trace_output) as write_trace:
        for parsed in parsed_list:
            qid = parsed.qid
            w1 = wave1.get(qid)
            w2 = wave2.get(qid)
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
            elif w2 is not None:
                path = f"wave_{w1.route}_sc"
                route = w1.route
                margin = w1.margin
                error = None
            else:
                path = "wave_direct"
                route = w1.route
                margin = w1.margin
                error = None

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
                }
            )
