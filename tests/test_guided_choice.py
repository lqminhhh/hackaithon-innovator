"""Tests for guided-choice prompt and selection behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reasoning_agent import ReasoningAgent


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


class _GuidedChoiceTestAgent(ReasoningAgent):
    def __init__(self):
        super().__init__(model=object(), tokenizer=_DummyTokenizer())

    def score_valid_labels(self, prompt, valid_labels):
        return {label: {"A": -3.0, "B": -0.2, "C": -1.5}.get(label, -9.0) for label in valid_labels}


class TestGuidedChoicePrompt:
    def test_prompt_lists_only_legal_labels(self):
        agent = _GuidedChoiceTestAgent()
        prompt = agent.build_guided_choice_prompt(
            question="Thủ đô của Việt Nam là gì?",
            options={"A": "Hà Nội", "B": "Huế", "C": "Đà Nẵng"},
        )

        assert "Chỉ trả lời bằng đúng một ký tự" in prompt
        assert "A, B, C" in prompt
        assert "{A/B/C}" in prompt
        assert "D" not in prompt.split("Đáp án:")[0]

    def test_prompt_includes_context_when_provided(self):
        agent = _GuidedChoiceTestAgent()
        prompt = agent.build_guided_choice_prompt(
            question="Theo đoạn thông tin, ai là tác giả?",
            options={"A": "X", "B": "Y"},
            context="Đoạn thông tin: Tác giả là X.",
        )

        assert "Chỉ sử dụng thông tin trong đoạn dưới đây để trả lời." in prompt
        assert "Đoạn thông tin: Tác giả là X." in prompt


class TestGuidedChoicePrediction:
    def test_prediction_returns_best_scored_label(self):
        agent = _GuidedChoiceTestAgent()
        answer, scores = agent.predict_guided_choice(
            question="1 + 1 bằng bao nhiêu?",
            options={"A": "1", "B": "2", "C": "3"},
        )

        assert answer == "B"
        assert scores["B"] > scores["C"] > scores["A"]
