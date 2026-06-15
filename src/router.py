"""Route questions to the cheapest appropriate reasoning path."""

from __future__ import annotations

from typing import Literal

from src.parser import ParsedQuestion

Route = Literal["reading", "stem", "safety", "knowledge"]


def route_question(parsed: ParsedQuestion) -> Route:
    """Assign a first-pass route from parsed metadata.

    The route only chooses the prompt/reasoning path. It does not decide
    correctness and can be overruled by later confidence gating.
    """
    if parsed.has_refusal_choice and parsed.is_harmful:
        return "safety"
    if parsed.has_context:
        return "reading"
    if parsed.is_quantitative:
        return "stem"
    return "knowledge"


def get_forced_answer(parsed: ParsedQuestion, route: Route) -> str | None:
    """Return a deterministic answer override for special cases.

    For genuinely harmful questions, if one choice is an explicit refusal,
    the refusal option is the correct answer and should be selected directly.
    """
    if route == "safety" and parsed.refusal_labels:
        return parsed.refusal_labels[0]
    return None
