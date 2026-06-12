"""Tests for gate.py — logprob margin, escalation decisions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gate import compute_margin, escalation_rate, should_escalate
from src.inference import InferenceResult


def _make_result(top1: str, dist: dict[str, float]) -> InferenceResult:
    return InferenceResult(letter=top1, logprob_dist=dist)


class TestComputeMargin:
    def test_clear_winner(self):
        r = _make_result("A", {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0})
        assert abs(compute_margin(r) - 1.9) < 1e-6

    def test_tie(self):
        r = _make_result("A", {"A": -1.0, "B": -1.0})
        assert compute_margin(r) == 0.0

    def test_single_letter(self):
        r = _make_result("A", {"A": -0.5})
        assert compute_margin(r) == 10.0  # only one option → no second


class TestShouldEscalate:
    def test_high_margin_accepted(self):
        r = _make_result("B", {"A": -3.0, "B": -0.1, "C": -2.0, "D": -4.0})
        escalate, reason = should_escalate(r, n_choices=4, tau=1.5)
        assert not escalate

    def test_low_margin_escalates(self):
        r = _make_result("A", {"A": -1.0, "B": -1.1, "C": -2.0, "D": -3.0})
        escalate, reason = should_escalate(r, n_choices=4, tau=1.5)
        assert escalate
        assert "margin" in reason

    def test_disc_check_triggers_for_10_plus_choices(self):
        gen = _make_result("A", {"A": -0.1, "B": -2.0})
        disc = _make_result("B", {"A": -2.0, "B": -0.1})  # different winner
        escalate, reason = should_escalate(gen, n_choices=10, disc_result=disc, tau=1.5)
        assert escalate
        assert "disc_check" in reason

    def test_disc_agrees_no_escalation(self):
        gen = _make_result("A", {"A": -0.1, "B": -3.0})
        disc = _make_result("A", {"A": -0.1, "B": -3.0})  # same winner
        escalate, _ = should_escalate(gen, n_choices=10, disc_result=disc, tau=1.5)
        assert not escalate


class TestEscalationRate:
    def test_all_accepted(self):
        results = [_make_result("A", {"A": -0.1, "B": -3.0}) for _ in range(5)]
        n_choices = [4] * 5
        flags, rate = escalation_rate(results, n_choices, tau=1.5)
        assert not any(flags)
        assert rate == 0.0

    def test_all_escalated(self):
        results = [_make_result("A", {"A": -1.0, "B": -1.05}) for _ in range(5)]
        n_choices = [4] * 5
        flags, rate = escalation_rate(results, n_choices, tau=1.5)
        assert all(flags)
        assert rate == 1.0

    def test_mixed(self):
        clear = _make_result("A", {"A": -0.1, "B": -3.0})
        close = _make_result("A", {"A": -1.0, "B": -1.05})
        results = [clear, close, clear, close]
        flags, rate = escalation_rate(results, [4] * 4, tau=1.5)
        assert sum(flags) == 2
        assert abs(rate - 0.5) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
