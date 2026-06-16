"""Tests for question parsing and flag extraction."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions
from src.parser import parse_question
from src.router import get_forced_answer, route_l1, route_question


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public-test_1780368312.json"


def _load_question_map() -> dict[str, dict]:
    questions = load_questions(DATA_PATH)
    return {q["qid"]: q for q in questions}


class TestParseQuestion:
    @classmethod
    def setup_class(cls):
        cls.question_map = _load_question_map()

    def test_reading_question_splits_context_and_query(self):
        parsed = parse_question(self.question_map["test_0001"])

        assert parsed.has_context is True
        assert parsed.context is not None
        assert "Đoạn thông tin" in parsed.context
        assert parsed.query.startswith("Theo nội dung được cung cấp")
        assert parsed.n_choices == 4

    def test_stem_question_detected_as_quantitative(self):
        parsed = parse_question(self.question_map["test_0006"])

        assert parsed.has_context is False
        assert parsed.is_quantitative is True
        assert parsed.n_choices == 10
        assert "phương trình vi phân" in parsed.query.lower()

    def test_refusal_choice_detected_without_harm_flag(self):
        parsed = parse_question(self.question_map["test_0019"])

        assert parsed.has_refusal_choice is True
        assert parsed.is_harmful is False
        assert parsed.refusal_labels == ("D",)
        assert any("tôi không thể" in option.lower() for option in parsed.options.values())

    def test_legal_question_detected(self):
        parsed = parse_question(self.question_map["test_0024"])

        assert parsed.is_legal is True
        assert parsed.has_refusal_choice is True
        assert parsed.is_harmful is False

    def test_harmful_question_detected(self):
        parsed = parse_question(self.question_map["test_0294"])

        assert parsed.has_refusal_choice is True
        assert parsed.is_harmful is True
        assert parsed.refusal_labels == ("C",)

    def test_public_set_flag_counts_match_current_snapshot(self):
        parsed_questions = [parse_question(q) for q in self.question_map.values()]

        counts = Counter()
        for parsed in parsed_questions:
            counts["has_context"] += int(parsed.has_context)
            counts["is_quantitative"] += int(parsed.is_quantitative)
            counts["is_legal"] += int(parsed.is_legal)
            counts["has_refusal_choice"] += int(parsed.has_refusal_choice)
            counts["is_harmful"] += int(parsed.is_harmful)

        assert counts == Counter(
            {
                "has_context": 100,
                "is_quantitative": 263,
                "is_legal": 157,
                "has_refusal_choice": 18,
                "is_harmful": 10,
            }
        )


class TestRouteQuestion:
    @classmethod
    def setup_class(cls):
        cls.question_map = _load_question_map()

    def test_reading_route_priority(self):
        parsed = parse_question(self.question_map["test_0001"])
        assert route_l1(parsed) == "reading"
        assert route_question(parsed) == "reading"

    def test_stem_route(self):
        parsed = parse_question(self.question_map["test_0002"])
        assert route_l1(parsed) == "stem"
        assert route_question(parsed) == "stem"

    def test_knowledge_route_for_normal_recall(self):
        parsed = parse_question(self.question_map["test_0041"])
        assert route_l1(parsed) is None
        assert route_question(parsed) == "knowledge"

    def test_safety_route_requires_refusal_and_harm(self):
        parsed = parse_question(self.question_map["test_0294"])
        assert route_l1(parsed) == "safety"
        assert route_question(parsed) == "safety"
        assert get_forced_answer(parsed, "safety") == "C"

    def test_refusal_without_harm_stays_out_of_safety(self):
        parsed = parse_question(self.question_map["test_0024"])
        assert route_l1(parsed) is None
        assert route_question(parsed) == "knowledge"
        assert get_forced_answer(parsed, "knowledge") is None

    def test_layer1_stem_fires_on_many_choices(self):
        parsed = parse_question(
            {
                "qid": "many_choices",
                "question": "Chọn phương án đúng nhất.",
                "options": {chr(65 + i): f"option {i}" for i in range(8)},
            }
        )

        assert route_l1(parsed) == "stem"

    def test_layer1_reading_fires_on_long_context(self):
        parsed = parse_question(
            {
                "qid": "long_context",
                "question": "A" * 650,
                "options": {"A": "Đúng", "B": "Sai"},
            }
        )

        assert route_l1(parsed) == "reading"
