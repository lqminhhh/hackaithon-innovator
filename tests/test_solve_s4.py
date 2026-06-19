"""Tests for S4 solve policy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import ChoiceResult
from src.parser import ParsedQuestion
from src.solve import self_consistency, solve_question


def _parsed(
    *,
    qid: str = "q1",
    query: str = "Câu hỏi?",
    options: dict[str, str] | None = None,
    has_context: bool = False,
    is_quantitative: bool = False,
    has_refusal_choice: bool = False,
    is_harmful: bool = False,
    refusal_labels: tuple[str, ...] = (),
) -> ParsedQuestion:
    opts = options or {"A": "Một", "B": "Hai", "C": "Ba"}
    return ParsedQuestion(
        qid=qid,
        original_question=query,
        query=query,
        context="Đoạn thông tin" if has_context else None,
        options=opts,
        refusal_labels=refusal_labels,
        n_choices=len(opts),
        has_context=has_context,
        is_quantitative=is_quantitative,
        is_legal=False,
        has_refusal_choice=has_refusal_choice,
        is_harmful=is_harmful,
    )


class _FakeAgent:
    def __init__(
        self,
        *,
        direct: ChoiceResult,
        sc_scores: list[dict[str, float]] | None = None,
    ):
        self.direct = direct
        self.sc_scores = list(sc_scores or [])
        self.direct_calls = []
        self.generated = []
        self.scored_prompts = []

    def predict_route_choice_result(self, **kwargs):
        self.direct_calls.append(kwargs)
        return self.direct

    def generate_freeform(self, prompts, **kwargs):
        self.generated.append({"prompts": prompts, "kwargs": kwargs})
        return [f"reasoning-{i}" for i, _ in enumerate(prompts)]

    def score_valid_labels(self, prompt, valid_labels):
        self.scored_prompts.append(prompt)
        if self.sc_scores:
            return self.sc_scores.pop(0)
        return {label: -float(i) for i, label in enumerate(valid_labels)}


def _choice(letter: str, margin: float = 1.0) -> ChoiceResult:
    scores = {"A": -3.0, "B": -2.0, "C": -1.0}
    scores[letter] = 0.0
    return ChoiceResult(letter=letter, margin=margin, per_letter_logprob=scores)


class _FakeSemanticDecision:
    def __init__(
        self,
        *,
        layer2_route,
        final_route,
        should_override,
        override_blockers=(),
    ):
        self.layer2_route = layer2_route
        self.final_route = final_route
        self.should_override = should_override
        self.override_blockers = override_blockers


class _FakeSemanticRouter:
    def __init__(
        self,
        *,
        layer2_route,
        final_route,
        should_override=True,
        override_blockers=(),
    ):
        self.layer2_route = layer2_route
        self.final_route = final_route
        self.should_override = should_override
        self.override_blockers = override_blockers
        self.calls = []

    def decide_route(self, parsed, layer1_route):
        self.calls.append({"parsed": parsed, "layer1_route": layer1_route})
        return _FakeSemanticDecision(
            layer2_route=self.layer2_route,
            final_route=self.final_route,
            should_override=self.should_override,
            override_blockers=self.override_blockers,
        )


def test_high_margin_knowledge_accepts_direct_answer():
    agent = _FakeAgent(direct=_choice("B", margin=0.8))
    parsed = _parsed()

    solved = solve_question(agent, parsed)

    assert solved.answer == "B"
    assert solved.path == "direct"
    assert solved.route == "knowledge"
    assert agent.generated == []


def test_stem_always_runs_self_consistency():
    agent = _FakeAgent(
        direct=_choice("A", margin=0.9),
        sc_scores=[
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -0.1, "B": -2.0, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
        ],
    )
    parsed = _parsed(is_quantitative=True)

    solved = solve_question(agent, parsed)

    assert solved.route == "stem"
    assert solved.path == "stem_self_consistency"
    assert solved.first_answer == "A"
    assert solved.answer == "B"
    assert solved.votes == ["B", "B", "A", "B", "B"]
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_reason_purpose_reading_runs_small_self_consistency():
    agent = _FakeAgent(
        direct=_choice("A", margin=1.0),
        sc_scores=[
            {"A": -2.0, "B": -3.0, "C": -0.1},
            {"A": -2.0, "B": -3.0, "C": -0.1},
            {"A": -0.1, "B": -3.0, "C": -2.0},
        ],
    )
    parsed = _parsed(
        query="Theo ngữ cảnh, lý do chính nhân vật xây pháo đài là gì?",
        has_context=True,
    )

    solved = solve_question(agent, parsed)

    assert solved.route == "reading"
    assert solved.path == "reading_reason_self_consistency"
    assert solved.first_answer == "A"
    assert solved.answer == "C"
    assert solved.votes == ["C", "C", "A"]
    assert len(agent.generated[0]["prompts"]) == 3
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_plain_reading_still_accepts_direct_answer():
    agent = _FakeAgent(direct=_choice("B", margin=0.7))
    parsed = _parsed(
        query="Theo đoạn thông tin, nhân vật chính là ai?",
        has_context=True,
    )

    solved = solve_question(agent, parsed)

    assert solved.route == "reading"
    assert solved.path == "direct"
    assert solved.answer == "B"
    assert agent.generated == []


def test_low_margin_knowledge_runs_self_consistency():
    agent = _FakeAgent(
        direct=_choice("A", margin=0.01),
        sc_scores=[
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -0.1, "B": -2.0, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
        ],
    )
    parsed = _parsed()

    solved = solve_question(agent, parsed)

    assert solved.path == "low_margin_self_consistency"
    assert solved.answer == "B"


def test_forced_safety_skips_model():
    agent = _FakeAgent(direct=_choice("A"))
    parsed = _parsed(
        has_refusal_choice=True,
        is_harmful=True,
        refusal_labels=("C",),
    )

    solved = solve_question(agent, parsed)

    assert solved.route == "safety"
    assert solved.path == "forced_safety"
    assert solved.answer == "C"
    assert agent.generated == []


def test_semantic_safety_override_forces_refusal():
    agent = _FakeAgent(direct=_choice("A"))
    semantic_router = _FakeSemanticRouter(
        layer2_route="safety",
        final_route="safety",
        should_override=True,
    )
    parsed = _parsed(
        options={"A": "Làm theo yêu cầu", "B": "Tôi không thể hỗ trợ", "C": "Khác"},
        has_refusal_choice=True,
        is_harmful=False,
        refusal_labels=("B",),
    )

    solved = solve_question(agent, parsed, semantic_router=semantic_router)

    assert semantic_router.calls[0]["layer1_route"] == "knowledge"
    assert solved.route == "safety"
    assert solved.semantic_route == "safety"
    assert solved.route_override is True
    assert solved.path == "forced_safety"
    assert solved.answer == "B"
    assert agent.direct_calls == []


def test_semantic_stem_override_runs_self_consistency():
    agent = _FakeAgent(
        direct=_choice("A", margin=0.9),
        sc_scores=[
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -0.1, "B": -2.0, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
        ],
    )
    semantic_router = _FakeSemanticRouter(
        layer2_route="stem",
        final_route="stem",
        should_override=True,
    )
    parsed = _parsed()

    solved = solve_question(agent, parsed, semantic_router=semantic_router)

    assert semantic_router.calls[0]["layer1_route"] == "knowledge"
    assert solved.route == "stem"
    assert solved.semantic_route == "stem"
    assert solved.route_override is True
    assert solved.path == "stem_self_consistency"
    assert solved.answer == "B"
    assert solved.votes == ["B", "B", "A", "B", "B"]


def test_blocked_semantic_decision_keeps_layer1_route():
    agent = _FakeAgent(direct=_choice("B", margin=0.8))
    semantic_router = _FakeSemanticRouter(
        layer2_route="reading",
        final_route="knowledge",
        should_override=False,
        override_blockers=("reading_without_context",),
    )
    parsed = _parsed()

    solved = solve_question(agent, parsed, semantic_router=semantic_router)

    assert solved.route == "knowledge"
    assert solved.semantic_route == "reading"
    assert solved.route_override is False
    assert solved.override_blockers == ["reading_without_context"]
    assert solved.path == "direct"
    assert solved.answer == "B"


def test_self_consistency_tie_prefers_first_answer():
    agent = _FakeAgent(
        direct=_choice("A", margin=0.1),
        sc_scores=[
            {"A": -0.1, "B": -2.0, "C": -3.0},
            {"A": -2.0, "B": -0.1, "C": -3.0},
        ],
    )
    parsed = _parsed(is_quantitative=True)

    vote = self_consistency(agent, parsed, "stem", _choice("A"), n=2)

    assert vote.letter == "A"
    assert vote.votes == ["A", "B"]
