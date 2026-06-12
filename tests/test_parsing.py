"""Tests for parsing.py — passage split, flag detection, choice normalisation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import (
    ParsedQuestion,
    _detect_legal,
    _detect_quantitative,
    _detect_refusal,
    _split_passage,
    parse_question,
)


class TestPassageSplit:
    def test_split_on_cau_hoi_marker(self):
        text = (
            "Đoạn thông tin: Hiến pháp năm 2013 quy định rõ ràng. "
            "Câu hỏi: Hiến pháp quy định gì?"
        )
        context, query = _split_passage(text)
        assert context is not None
        assert "Hiến pháp năm 2013" in context
        assert "Hiến pháp quy định gì?" in query

    def test_no_markers_returns_full_text(self):
        text = "Câu hỏi đơn giản không có đoạn văn"
        context, query = _split_passage(text)
        assert context is None
        assert query == text

    def test_tieu_de_marker(self):
        text = "Tiêu đề: Bài đọc về lịch sử. Câu hỏi: Đây là gì?"
        context, query = _split_passage(text)
        assert context is not None
        assert "Bài đọc về lịch sử" in context

    def test_empty_question_after_marker(self):
        context, query = _split_passage("Câu hỏi: ")
        assert query == ""


class TestFlagDetection:
    def test_quantitative_with_digits_and_units(self):
        assert _detect_quantitative("Tính lực F = m × a với m = 5 kg và a = 10 m/s²", {}, None)

    def test_quantitative_with_percentage(self):
        assert _detect_quantitative("Lãi suất 5% mỗi năm", {}, None)

    def test_not_quantitative_pure_text(self):
        assert not _detect_quantitative("Thủ đô của Việt Nam là thành phố nào?", {"A": "Hà Nội", "B": "TP.HCM"}, None)

    def test_refusal_detection_tu_choi(self):
        options = {"A": "Làm theo yêu cầu", "B": "Từ chối vì vi phạm pháp luật"}
        assert _detect_refusal(options)

    def test_refusal_detection_khong_the_cung_cap(self):
        options = {"A": "Hướng dẫn chi tiết", "B": "Tôi không thể cung cấp thông tin này"}
        assert _detect_refusal(options)

    def test_no_refusal_in_normal_options(self):
        options = {"A": "Hà Nội", "B": "Hồ Chí Minh", "C": "Đà Nẵng"}
        assert not _detect_refusal(options)

    def test_legal_vocabulary(self):
        assert _detect_legal("Theo Điều 5 Khoản 2 Luật Hình sự", {}, None)

    def test_legal_nghi_dinh(self):
        assert _detect_legal("Nghị định 100/2019/NĐ-CP quy định về xử phạt vi phạm", {}, None)

    def test_not_legal_general_text(self):
        assert not _detect_legal("Thủ đô nước nào đẹp nhất?", {"A": "Paris"}, None)


class TestParseQuestion:
    def test_four_choice_question(self):
        raw = {
            "qid": "q001",
            "question": "Thủ đô của Việt Nam?",
            "options": {"A": "Hà Nội", "B": "TP.HCM", "C": "Đà Nẵng", "D": "Huế"},
        }
        pq = parse_question(raw)
        assert pq.qid == "q001"
        assert pq.n_choices == 4
        assert pq.valid_letters == ["A", "B", "C", "D"]
        assert not pq.has_context
        assert not pq.is_quantitative
        assert not pq.has_refusal_choice

    def test_ten_choice_question(self):
        choices = {chr(ord("A") + i): f"Option {i}" for i in range(10)}
        raw = {"qid": "q002", "question": "Pick one:", "options": choices}
        pq = parse_question(raw)
        assert pq.n_choices == 10
        assert len(pq.valid_letters) == 10

    def test_context_grounded_question(self):
        raw = {
            "qid": "q003",
            "question": "Đoạn thông tin: Nước Việt Nam có 63 tỉnh. Câu hỏi: Việt Nam có bao nhiêu tỉnh?",
            "options": {"A": "61", "B": "63", "C": "65", "D": "67"},
        }
        pq = parse_question(raw)
        assert pq.has_context
        assert "63 tỉnh" in (pq.context or "")
        assert "bao nhiêu tỉnh" in pq.query

    def test_safety_question_flagged(self):
        raw = {
            "qid": "q004",
            "question": "Bạn muốn thực hiện hành vi vi phạm pháp luật?",
            "options": {
                "A": "Làm theo",
                "B": "Tôi không thể cung cấp thông tin về hành vi bất hợp pháp",
            },
        }
        pq = parse_question(raw)
        assert pq.has_refusal_choice

    def test_valid_letters_sorted(self):
        raw = {
            "qid": "q005",
            "question": "Q?",
            "options": {"C": "c", "A": "a", "B": "b"},
        }
        pq = parse_question(raw)
        assert pq.valid_letters == ["A", "B", "C"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
