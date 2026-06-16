"""Route questions to the cheapest appropriate reasoning path."""

from __future__ import annotations

from typing import Literal

from src.parser import ParsedQuestion

Route = Literal["reading", "stem", "safety", "knowledge"]
Layer1Route = Literal["reading", "stem", "safety"]


def route_l1(parsed: ParsedQuestion) -> Layer1Route | None:
    """Assign the S3 Layer-1 rule route, abstaining on plain knowledge.

    Layer 1 only decides cheap special paths. Returning ``None`` means the
    caller should use the full KNOWLEDGE default or pass the item to Layer 2.
    """
    if parsed.has_refusal_choice and parsed.is_harmful:
        return "safety"
    if parsed.has_context:
        return "reading"
    if parsed.is_quantitative:
        return "stem"
    return None


def route_question(parsed: ParsedQuestion) -> Route:
    """Return the final first-pass route, defaulting L1 abstentions to knowledge."""
    route = route_l1(parsed)
    if route is not None:
        return route
    return "knowledge"


def get_forced_answer(parsed: ParsedQuestion, route: Route) -> str | None:
    """Return a deterministic answer override for special cases.

    For genuinely harmful questions, if one choice is an explicit refusal,
    the refusal option is the correct answer and should be selected directly.
    """
    if route == "safety" and parsed.refusal_labels:
        return parsed.refusal_labels[0]
    return None
