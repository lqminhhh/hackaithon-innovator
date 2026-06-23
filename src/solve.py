"""S4 route-specific compute budgets and escalation logic."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any

from src.config import FALLBACK, MARGIN_LOW, SC_N, SC_TEMP, TOK
from src.extract import ChoiceResult, best_label, softmax_margin
from src.parser import ParsedQuestion
from src.reasoning_agent import ReasoningAgent
from src.router import Route, get_forced_answer, route_question


@dataclass(frozen=True, slots=True)
class VoteResult:
    letter: str
    confidence: float
    votes: list[str]
    margins: list[float]


@dataclass(frozen=True, slots=True)
class SolveResult:
    qid: str
    answer: str
    route: Route
    margin: float | None
    path: str
    first_answer: str | None = None
    votes: list[str] = field(default_factory=list)
    semantic_route: Route | None = None
    route_override: bool = False
    override_blockers: list[str] = field(default_factory=list)
    error: str | None = None


def solve_question(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    sc_batch_size: int | None = None,
    semantic_router=None,
) -> SolveResult:
    """Solve one parsed question using the S4 policy."""
    route = route_question(parsed)
    route_meta: dict[str, Any] = {
        "semantic_route": None,
        "route_override": False,
        "override_blockers": [],
    }
    if semantic_router is not None and hasattr(semantic_router, "decide_route"):
        decision = semantic_router.decide_route(parsed, route)
        route_meta["semantic_route"] = getattr(decision, "layer2_route", None)
        route_meta["route_override"] = bool(getattr(decision, "should_override", False))
        route_meta["override_blockers"] = list(getattr(decision, "override_blockers", ()))
        if route_meta["route_override"]:
            route = getattr(decision, "final_route", route)
    try:
        forced = get_forced_answer(parsed, route)
        if forced is not None:
            return SolveResult(
                qid=parsed.qid,
                answer=forced,
                route=route,
                margin=None,
                path="forced_safety",
                first_answer=forced,
                **route_meta,
            )

        first = _direct_choice(agent, parsed, route)

        if route == "stem":
            vote = self_consistency(agent, parsed, route, first, batch_size=sc_batch_size)
            return SolveResult(
                qid=parsed.qid,
                answer=vote.letter,
                route=route,
                margin=first.margin,
                path="stem_self_consistency",
                first_answer=first.letter,
                votes=vote.votes,
                **route_meta,
            )

        if route == "reading" and _is_reason_purpose_question(parsed.query):
            vote = self_consistency(
                agent,
                parsed,
                route,
                first,
                n=3,
                batch_size=sc_batch_size,
            )
            return SolveResult(
                qid=parsed.qid,
                answer=vote.letter,
                route=route,
                margin=first.margin,
                path="reading_reason_self_consistency",
                first_answer=first.letter,
                votes=vote.votes,
                **route_meta,
            )

        if route == "knowledge" and first.margin < MARGIN_LOW:
            vote = self_consistency(agent, parsed, route, first, batch_size=sc_batch_size)
            return SolveResult(
                qid=parsed.qid,
                answer=vote.letter,
                route=route,
                margin=first.margin,
                path="low_margin_self_consistency",
                first_answer=first.letter,
                votes=vote.votes,
                **route_meta,
            )

        return SolveResult(
            qid=parsed.qid,
            answer=first.letter,
            route=route,
            margin=first.margin,
            path="direct",
            first_answer=first.letter,
            **route_meta,
        )
    except Exception as exc:
        return SolveResult(
            qid=parsed.qid,
            answer=FALLBACK,
            route=route,
            margin=None,
            path="fallback",
            error=str(exc),
            **route_meta,
        )


def self_consistency(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    route: Route,
    first: ChoiceResult | None = None,
    *,
    n: int = SC_N,
    batch_size: int | None = None,
) -> VoteResult:
    """Run n free-reasoning samples and majority vote over constrained letters."""
    options = parsed.options
    valid_labels = tuple(sorted(options.keys()))
    reasoning_prompt = _build_reasoning_prompt(parsed, route)
    prompts = [reasoning_prompt] * n
    reasonings: list[str] = []
    chunk_size = batch_size if batch_size is not None and batch_size > 0 else len(prompts)
    for start in range(0, len(prompts), chunk_size):
        reasonings.extend(
            agent.generate_freeform(
                prompts[start : start + chunk_size],
                mode="think",
                max_tokens=_route_tokens(route),
                temperature=SC_TEMP,
                top_p=0.95,
            )
        )

    choices: list[ChoiceResult] = []
    for reasoning in reasonings:
        extract_prompt = _build_extraction_from_reasoning(
            reasoning_prompt=reasoning_prompt,
            reasoning=reasoning,
        )
        scores = agent.score_valid_labels(extract_prompt, valid_labels)
        choices.append(
            ChoiceResult(
                letter=best_label(scores),
                margin=softmax_margin(scores),
                per_letter_logprob=scores,
            )
        )

    return _vote(choices, first)


def _direct_choice(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    route: Route,
) -> ChoiceResult:
    return agent.predict_route_choice_result(
        route=route,
        question=parsed.query,
        options=parsed.options,
        context=parsed.context if route == "reading" else None,
    )


def _vote(
    choices: list[ChoiceResult],
    first: ChoiceResult | None = None,
) -> VoteResult:
    if not choices:
        if first is None:
            return VoteResult(FALLBACK, 0.0, [], [])
        return VoteResult(first.letter, 1.0, [first.letter], [first.margin])

    votes = [choice.letter for choice in choices]
    counts = Counter(votes)
    top_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == top_count)

    if len(tied) == 1:
        winner = tied[0]
    elif first is not None and first.letter in tied:
        winner = first.letter
    else:
        winner = _highest_average_logprob(choices, tied)

    confidence = counts[winner] / len(choices)
    return VoteResult(
        letter=winner,
        confidence=confidence,
        votes=votes,
        margins=[choice.margin for choice in choices],
    )


def _highest_average_logprob(choices: list[ChoiceResult], labels: list[str]) -> str:
    totals: dict[str, list[float]] = defaultdict(list)
    for choice in choices:
        for label in labels:
            score = choice.per_letter_logprob.get(label, float("-inf"))
            if math.isfinite(score):
                totals[label].append(score)

    averages = {
        label: sum(values) / len(values) if values else float("-inf")
        for label, values in totals.items()
    }
    return max(labels, key=lambda label: averages.get(label, float("-inf")))


def _is_reason_purpose_question(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "lý do",
            "lí do",
            "mục đích",
            "nguyên nhân",
            "vì sao",
            "tại sao",
            "do đâu",
            "nhằm mục đích",
            "để làm gì",
        )
    )


def _route_tokens(route: Route) -> int:
    return TOK[route.upper()]


def _build_reasoning_prompt(parsed: ParsedQuestion, route: Route) -> str:
    options_block = "\n".join(
        f"{label}) {parsed.options[label]}" for label in sorted(parsed.options)
    )
    route_instruction = {
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


def _build_extraction_from_reasoning(reasoning_prompt: str, reasoning: str) -> str:
    return (
        f"{reasoning_prompt}\n\n"
        "Lời giải nháp:\n"
        f"{reasoning}\n\n"
        "Dựa trên lời giải nháp, chọn đúng một đáp án hợp lệ.\n"
        "Đáp án: "
    )
