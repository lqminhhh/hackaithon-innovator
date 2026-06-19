"""S4 route-specific compute budgets and escalation logic.

S6 RAG is wired in here via the optional ``rag`` parameter of
``solve_question``. Pass a ``RAGEngine`` instance to enable retrieval for
low-margin knowledge questions. Omitting it preserves the pre-S6 behaviour.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from src.config import FALLBACK, MARGIN_LOW, SC_N, SC_TEMP, TOK
from src.extract import ChoiceResult, best_label, softmax_margin
from src.parser import ParsedQuestion
from src.reasoning_agent import ReasoningAgent
from src.router import Route, get_forced_answer, route_question

if TYPE_CHECKING:
    from src.rag import RAGEngine
    from src.semantic_router import SemanticRouter


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
    layer1_route: Route | None = None
    semantic_route: Route | None = None
    route_override: bool = False
    override_blockers: list[str] = field(default_factory=list)
    semantic_error: str | None = None
    first_answer: str | None = None
    votes: list[str] = field(default_factory=list)
    error: str | None = None


def solve_question(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    rag: "RAGEngine | None" = None,
    semantic_router: "SemanticRouter | None" = None,
) -> SolveResult:
    """Solve one parsed question using the S4 policy with optional S6 RAG.

    Parameters
    ----------
    agent:
        Loaded ReasoningAgent (vLLM or HuggingFace backend).
    parsed:
        Pre-parsed question from ``parse_question()``.
    rag:
        Optional RAGEngine. When provided, low-margin knowledge questions
        attempt retrieval before falling back to self-consistency. Pass None
        (the default) to skip RAG entirely and preserve the pre-S6 behaviour.
    semantic_router:
        Optional S5 Layer-2 router. When provided, it may override the Layer-1
        route before route-specific answering begins.
    """
    route, route_meta = _route_with_semantics(parsed, semantic_router)
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
            vote = self_consistency(agent, parsed, route, first)
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
            vote = self_consistency(agent, parsed, route, first, n=3)
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
            # S6: try RAG before self-consistency
            if rag is not None:
                rag_result = _try_rag(agent, parsed, rag, first, route_meta)
                if rag_result is not None:
                    return rag_result

            vote = self_consistency(agent, parsed, route, first)
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


def _route_with_semantics(
    parsed: ParsedQuestion,
    semantic_router: "SemanticRouter | None",
) -> tuple[Route, dict[str, Any]]:
    """Run Layer-1 routing, then optional S5 semantic override."""
    layer1_route = route_question(parsed)
    route_meta: dict[str, Any] = {
        "layer1_route": layer1_route,
    }

    if semantic_router is None:
        return layer1_route, route_meta

    try:
        decision = semantic_router.decide_route(parsed, layer1_route=layer1_route)
    except Exception as exc:
        route_meta["semantic_error"] = str(exc)
        return layer1_route, route_meta

    route_meta.update(
        {
            "semantic_route": decision.layer2_route,
            "route_override": decision.should_override,
            "override_blockers": list(decision.override_blockers),
        }
    )
    return decision.final_route, route_meta


def _try_rag(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    rag: "RAGEngine",
    first: ChoiceResult,
    route_meta: dict[str, Any] | None = None,
) -> SolveResult | None:
    """Attempt RAG-augmented answer for a low-margin knowledge question.

    Returns a SolveResult if retrieval found relevant context and the
    re-answer has margin >= the first-pass margin. Returns None otherwise
    so the caller can fall through to self-consistency.
    """
    try:
        context = rag.retrieve_and_rerank(parsed.query, exclude_id=parsed.qid)
        if context is None:
            return None

        rag_choice = agent.predict_route_choice_result(
            route="knowledge",
            question=parsed.query,
            options=parsed.options,
            context=context,
        )

        if rag_choice.margin >= first.margin:
            return SolveResult(
                qid=parsed.qid,
                answer=rag_choice.letter,
                route="knowledge",
                margin=rag_choice.margin,
                path="knowledge_rag",
                first_answer=first.letter,
                **(route_meta or {}),
            )
        return None
    except Exception as exc:
        print(f"  [RAG] _try_rag failed for {parsed.qid}: {exc}", flush=True)
        return None


def self_consistency(
    agent: ReasoningAgent,
    parsed: ParsedQuestion,
    route: Route,
    first: ChoiceResult | None = None,
    *,
    n: int = SC_N,
) -> VoteResult:
    """Run n free-reasoning samples and majority vote over constrained letters."""
    options = parsed.options
    valid_labels = tuple(sorted(options.keys()))
    reasoning_prompt = _build_reasoning_prompt(parsed, route)
    prompts = [reasoning_prompt] * n
    reasonings = agent.generate_freeform(
        prompts,
        mode="think",
        max_tokens=_route_tokens(route),
        temperature=SC_TEMP,
        top_p=0.95,
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
