"""Tests for adaptive self-consistency policy and wave escalation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import ParsedQuestion
from src.sc_policy import SC_N_STEM, stem_sc_n
from src.wave_solver import Wave1Result, run_wave2


def _parsed(*, qid: str = "q1") -> ParsedQuestion:
    options = {"A": "Một", "B": "Hai", "C": "Ba"}
    return ParsedQuestion(
        qid=qid,
        original_question="1 + 1 bằng bao nhiêu?",
        query="1 + 1 bằng bao nhiêu?",
        context=None,
        options=options,
        refusal_labels=(),
        n_choices=len(options),
        has_context=False,
        is_quantitative=True,
        is_legal=False,
        has_refusal_choice=False,
        is_harmful=False,
    )


class _FakeAgent:
    is_vllm = False

    def __init__(self) -> None:
        self.generated = []
        self.scored_prompts = []

    def generate_freeform(self, prompts, **kwargs):
        self.generated.append({"prompts": prompts, "kwargs": kwargs})
        return [f"reasoning-{idx}" for idx, _prompt in enumerate(prompts)]

    def score_valid_labels(self, prompt, valid_labels):
        self.scored_prompts.append({"prompt": prompt, "valid_labels": valid_labels})
        return {label: (0.0 if label == "A" else -2.0) for label in valid_labels}


def test_stem_sc_n_uses_low_depth_only_for_low_margin_adaptive_mode():
    assert stem_sc_n(0.01, adaptive_sc=True) == SC_N_STEM["low"]
    assert stem_sc_n(0.90, adaptive_sc=True) == SC_N_STEM["high"]
    assert stem_sc_n(None, adaptive_sc=True) == SC_N_STEM["high"]
    assert stem_sc_n(0.01, adaptive_sc=False) == SC_N_STEM["high"]


def test_wave2_uses_seven_stem_samples_for_low_margin_when_adaptive():
    agent = _FakeAgent()
    parsed = _parsed()
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="stem",
            answer="B",
            margin=0.01,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].answer in parsed.options
    assert len(wave2[parsed.qid].votes) > 0
    assert wave2[parsed.qid].escalation_reason.startswith("stem_sc_adaptive")
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_STEM["low"]
    assert len(agent.scored_prompts) == SC_N_STEM["low"]
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_wave2_uses_three_stem_samples_when_adaptive_disabled():
    agent = _FakeAgent()
    parsed = _parsed()
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="stem",
            answer="B",
            margin=0.01,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=False)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].answer in parsed.options
    assert len(wave2[parsed.qid].votes) > 0
    assert wave2[parsed.qid].escalation_reason.startswith("stem_sc_fixed")
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_STEM["high"]
    assert len(agent.scored_prompts) == SC_N_STEM["high"]
