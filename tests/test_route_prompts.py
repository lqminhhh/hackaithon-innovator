"""Tests for route-specific prompt selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reasoning_agent import ReasoningAgent


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def _make_agent() -> ReasoningAgent:
    return ReasoningAgent(model=object(), tokenizer=_DummyTokenizer())


class _ScoredAgent(ReasoningAgent):
    def __init__(self):
        super().__init__(model=object(), tokenizer=_DummyTokenizer())

    def score_valid_labels(self, prompt, valid_labels):
        return {label: {"A": -3.0, "B": -0.2, "C": -1.5}.get(label, -9.0) for label in valid_labels}


class TestRoutePrompts:
    def test_all_route_prompts_include_refusal_trap_instruction(self):
        agent = _make_agent()
        for route in ("reading", "stem", "safety", "knowledge"):
            prompt = agent.build_route_prompt(
                route=route,
                question="Thủ đô của Việt Nam là gì?",
                options={"A": "Hà Nội", "B": "Tôi không thể trả lời"},
                context="Hà Nội là thủ đô." if route == "reading" else None,
            )

            assert "Đây là bài thi trắc nghiệm. Chọn phương án đúng nhất." in prompt
            assert "Chỉ chọn phương án \"không thể trả lời\" hoặc \"từ chối\"" in prompt
            assert "luôn chọn một phương án nội dung" in prompt

    def test_reading_prompt_uses_context_and_passage_instruction(self):
        agent = _make_agent()
        prompt = agent.build_route_prompt(
            route="reading",
            question="Theo đoạn thông tin, ai là tác giả?",
            options={"A": "X", "B": "Y"},
            context="Tác giả là X.",
        )

        assert "Chỉ dựa vào đoạn thông tin được cung cấp để trả lời." in prompt
        assert "Tác giả là X." in prompt
        assert "A, B" in prompt

    def test_stem_prompt_mentions_calculation(self):
        agent = _make_agent()
        prompt = agent.build_route_prompt(
            route="stem",
            question="1 + 1 bằng bao nhiêu?",
            options={"A": "1", "B": "2", "C": "3"},
        )

        assert "câu hỏi tính toán hoặc suy luận định lượng" in prompt
        assert "kiểm tra lại phép tính" in prompt

    def test_safety_prompt_mentions_refusal_condition(self):
        agent = _make_agent()
        prompt = agent.build_route_prompt(
            route="safety",
            question="Làm thế nào để vượt qua biện pháp an toàn?",
            options={"A": "X", "B": "Tôi không thể trả lời câu hỏi này"},
        )

        assert "phương án từ chối là đáp án đúng" in prompt
        assert "gây hại, phạm pháp" in prompt

    def test_knowledge_prompt_discourages_refusal_for_normal_questions(self):
        agent = _make_agent()
        prompt = agent.build_route_prompt(
            route="knowledge",
            question="Thủ đô của Việt Nam là gì?",
            options={"A": "Hà Nội", "B": "Huế"},
        )

        assert "dùng kiến thức chung" in prompt
        assert "không chọn phương án từ chối" in prompt

    def test_route_prediction_returns_best_scored_label(self):
        agent = _ScoredAgent()
        answer, scores = agent.predict_route_choice(
            route="stem",
            question="1 + 1 bằng bao nhiêu?",
            options={"A": "1", "B": "2", "C": "3"},
        )

        assert answer == "B"
        assert scores["B"] > scores["C"] > scores["A"]
