"""Tests for the answer normaliser and confidence parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normaliser import normalise_answer, parse_confidence


class TestNormaliseAnswer:
    def test_explicit_dap_an_format(self):
        assert normalise_answer("Sau khi phân tích, ĐÁP ÁN: B") == "B"

    def test_dap_an_with_dash(self):
        assert normalise_answer("ĐÁP ÁN - C") == "C"

    def test_vietnamese_phrasing_chon(self):
        assert normalise_answer("Tôi chọn D vì lý do...") == "D"

    def test_vietnamese_phrasing_cau_tra_loi(self):
        assert normalise_answer("Câu trả lời: A") == "A"

    def test_parenthesised_letter(self):
        assert normalise_answer("The answer is (B)") == "B"

    def test_bracket_letter(self):
        assert normalise_answer("I think [C] is correct") == "C"

    def test_last_standalone_letter(self):
        assert normalise_answer("Considering A and B, but ultimately D") == "D"

    def test_fallback_most_common(self):
        assert normalise_answer("xxxxxx") in "ABCD"

    def test_mixed_format(self):
        assert normalise_answer("Bước 1: A sai. Bước 2: B đúng. ĐÁP ÁN: B") == "B"

    def test_lowercase_dap_an(self):
        assert normalise_answer("đáp án: c") == "C"

    def test_empty_string(self):
        result = normalise_answer("")
        assert result in "ABCD"


class TestParseConfidence:
    def test_standard_format(self):
        assert parse_confidence("ĐỘ TỰ TIN: 0.85") == 0.85

    def test_high_confidence(self):
        assert parse_confidence("ĐỘ TỰ TIN: 0.95") == 0.95

    def test_clamp_above_one(self):
        assert parse_confidence("ĐỘ TỰ TIN: 1.5") == 1.0

    def test_clamp_below_zero(self):
        assert parse_confidence("ĐỘ TỰ TIN: -0.3") == 0.0

    def test_missing_confidence(self):
        assert parse_confidence("No confidence here") == 0.5

    def test_empty_string(self):
        assert parse_confidence("") == 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
