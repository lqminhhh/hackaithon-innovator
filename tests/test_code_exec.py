"""Tests for tools/code_exec.py — subprocess sandbox + numeric fuzzy matcher."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.code_exec import (
    execute_code,
    match_to_choice,
    normalise_number,
    _check_imports,
)


class TestNormaliseNumber:
    def test_simple_integer(self):
        assert normalise_number("100") == 100.0

    def test_decimal_point(self):
        assert normalise_number("3.14") == pytest.approx(3.14)

    def test_decimal_comma(self):
        assert normalise_number("3,14") == pytest.approx(3.14)

    def test_thousand_dot(self):
        assert normalise_number("1.000") == 1000.0

    def test_thousand_dot_with_decimal_comma(self):
        assert normalise_number("1.000,5") == pytest.approx(1000.5)

    def test_percentage(self):
        assert normalise_number("12,5%") == pytest.approx(12.5)

    def test_trieu_scale(self):
        assert normalise_number("2 triệu") == pytest.approx(2e6)

    def test_ty_scale(self):
        assert normalise_number("1,5 tỷ") == pytest.approx(1.5e9)

    def test_unit_stripped(self):
        val = normalise_number("220 V")
        assert val == pytest.approx(220.0)

    def test_not_a_number(self):
        assert normalise_number("không có số") is None


class TestCheckImports:
    def test_allowed_import(self):
        assert _check_imports("import math\nprint(math.pi)") is None

    def test_disallowed_import(self):
        result = _check_imports("import requests\nrequests.get('http://x')")
        assert result is not None
        assert "requests" in result

    def test_from_import_allowed(self):
        assert _check_imports("from fractions import Fraction\nprint(Fraction(1,2))") is None

    def test_from_import_disallowed(self):
        result = _check_imports("from os import system\nsystem('ls')")
        assert result is not None


class TestExecuteCode:
    def test_simple_print(self):
        code = "print(42)"
        stdout, err = execute_code(code)
        assert err is None
        assert stdout == "42"

    def test_math_import(self):
        code = "import math\nprint(round(math.sqrt(144)))"
        stdout, err = execute_code(code)
        assert err is None
        assert stdout == "12"

    def test_disallowed_import_blocked(self):
        code = "import os\nprint(os.getcwd())"
        stdout, err = execute_code(code)
        assert stdout is None
        assert err is not None

    def test_syntax_error(self):
        code = "def broken("
        stdout, err = execute_code(code)
        assert stdout is None
        assert err is not None

    def test_runtime_error(self):
        code = "print(1 / 0)"
        stdout, err = execute_code(code)
        assert stdout is None
        assert err is not None


class TestMatchToChoice:
    def test_exact_match(self):
        options = {"A": "100", "B": "200", "C": "300", "D": "400"}
        assert match_to_choice("100", options) == "A"
        assert match_to_choice("300", options) == "C"

    def test_close_match_within_tolerance(self):
        options = {"A": "100", "B": "200,0", "C": "300", "D": "400"}
        assert match_to_choice("200.0", options) == "B"

    def test_vietnamese_format_matches(self):
        options = {"A": "1.000", "B": "2.000", "C": "5.000"}
        assert match_to_choice("2000", options) == "B"

    def test_no_match(self):
        options = {"A": "100", "B": "200"}
        assert match_to_choice("999", options) is None

    def test_letter_output_match(self):
        options = {"A": "100", "B": "200"}
        assert match_to_choice("A", options) == "A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
