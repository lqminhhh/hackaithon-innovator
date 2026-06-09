"""Tests for the confidence gate routing logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence_gate import route


class TestConfidenceGate:
    def test_high_confidence_fast_exit(self):
        assert route(0.90) == "fast_exit"
        assert route(0.85) == "fast_exit"
        assert route(1.0) == "fast_exit"

    def test_medium_confidence_consistency(self):
        assert route(0.70) == "consistency"
        assert route(0.55) == "consistency"
        assert route(0.84) == "consistency"

    def test_low_confidence_ensemble(self):
        assert route(0.54) == "ensemble"
        assert route(0.30) == "ensemble"
        assert route(0.0) == "ensemble"

    def test_boundary_fast_exit(self):
        assert route(0.85) == "fast_exit"
        assert route(0.849) == "consistency"

    def test_boundary_ensemble(self):
        assert route(0.55) == "consistency"
        assert route(0.549) == "ensemble"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
