"""Smoke tests for the Entropy-Gated Jury — no GPU / model required.

Tests the data-layer components that work without any ML models:
  - data_loader round-trip
  - parsing flags + choice normalisation
  - gate margin logic
  - assemble output invariants
  - code_exec fuzzy matcher
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions, write_submission
from src.parsing import parse_questions
from src.gate import should_escalate
from src.inference import InferenceResult
from src.assemble import assemble, AssemblyError
from src.tools.code_exec import match_to_choice, normalise_number


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_json(tmp_path):
    data = [
        {
            "qid": "test_0001",
            "question": "Thủ đô của Việt Nam là gì?",
            "choices": ["Hà Nội", "TP.HCM", "Đà Nẵng", "Huế"],
        },
        {
            "qid": "test_0002",
            "question": (
                "Đoạn thông tin: Hiến pháp 2013. "
                "Câu hỏi: Hiến pháp ra đời năm nào?"
            ),
            "choices": ["2011", "2013", "2015", "2016"],
        },
        {
            "qid": "test_0003",
            "question": "Tính 2 + 2 × 3 = ?",
            "choices": ["6", "8", "10", "12"],
        },
        {
            "qid": "test_0004",
            "question": "Bạn có muốn thực hiện hành vi trái pháp luật không?",
            "choices": [
                "Có, hướng dẫn tôi",
                "Tôi không thể cung cấp thông tin vi phạm pháp luật",
                "Không rõ",
            ],
        },
    ]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ── data loader ───────────────────────────────────────────────────────

class TestDataLoader:
    def test_json_loads(self, sample_json):
        qs = load_questions(sample_json)
        assert len(qs) == 4
        assert qs[0]["qid"] == "test_0001"
        assert set(qs[0]["options"].keys()) == {"A", "B", "C", "D"}

    def test_choice_labels_a_to_d(self, sample_json):
        qs = load_questions(sample_json)
        for q in qs:
            for label in q["options"]:
                assert label in "ABCDEFGHIJK"

    def test_3_choice_question(self, sample_json):
        qs = load_questions(sample_json)
        three_choice = [q for q in qs if len(q["options"]) == 3]
        assert len(three_choice) == 1
        assert set(three_choice[0]["options"].keys()) == {"A", "B", "C"}


# ── parsing ───────────────────────────────────────────────────────────

class TestParsing:
    def test_flags_correct(self, sample_json):
        qs = load_questions(sample_json)
        pqs = parse_questions(qs)

        # test_0001: plain knowledge
        assert not pqs[0].has_context
        assert not pqs[0].has_refusal_choice

        # test_0002: has context
        assert pqs[1].has_context
        assert pqs[1].context is not None

        # test_0003: quantitative (has digits + formula tokens)
        assert pqs[2].is_quantitative

        # test_0004: refusal choice
        assert pqs[3].has_refusal_choice

    def test_n_choices(self, sample_json):
        qs = load_questions(sample_json)
        pqs = parse_questions(qs)
        assert pqs[0].n_choices == 4
        assert pqs[3].n_choices == 3


# ── gate ─────────────────────────────────────────────────────────────

class TestGate:
    def test_clear_win_accepted(self):
        r = InferenceResult("A", {"A": -0.05, "B": -4.0, "C": -5.0, "D": -6.0})
        esc, _ = should_escalate(r, 4, tau=1.5)
        assert not esc

    def test_close_call_escalated(self):
        r = InferenceResult("A", {"A": -1.0, "B": -1.1, "C": -2.0, "D": -3.0})
        esc, _ = should_escalate(r, 4, tau=1.5)
        assert esc


# ── assemble ──────────────────────────────────────────────────────────

class TestAssemble:
    def test_round_trip(self, sample_json, tmp_path):
        qs = load_questions(sample_json)
        pqs = parse_questions(qs)

        tier1_results = [
            InferenceResult(letter=pq.valid_letters[0], logprob_dist={l: -1.0 for l in pq.valid_letters})
            for pq in pqs
        ]
        tier1_accepted = [True] * len(pqs)

        out_csv = tmp_path / "submission.csv"
        out_audit = tmp_path / "audit.json"

        df = assemble(
            questions=pqs,
            tier1_results=tier1_results,
            tier1_accepted=tier1_accepted,
            jury_verdicts={},
            output_csv=out_csv,
            output_audit=out_audit,
            strict=True,
        )

        assert len(df) == len(pqs)
        assert list(df.columns) == ["id", "answer"]
        loaded = pd.read_csv(out_csv)
        assert set(loaded["id"].astype(str)) == {pq.qid for pq in pqs}

    def test_strict_raises_on_invalid_answer(self, sample_json, tmp_path):
        qs = load_questions(sample_json)
        pqs = parse_questions(qs)

        bad_results = [
            InferenceResult(letter="Z", logprob_dist={"Z": -1.0})
            for _ in pqs
        ]

        with pytest.raises(AssemblyError):
            assemble(
                questions=pqs,
                tier1_results=bad_results,
                tier1_accepted=[True] * len(pqs),
                jury_verdicts={},
                output_csv=tmp_path / "s.csv",
                output_audit=tmp_path / "a.json",
                strict=True,
            )


# ── code_exec fuzzy matcher ───────────────────────────────────────────

class TestCodeExecMatcher:
    def test_basic_match(self):
        assert match_to_choice("100", {"A": "100", "B": "200"}) == "A"

    def test_vietnamese_decimal(self):
        assert match_to_choice("2,2", {"A": "1,5 A", "B": "2,2 A", "C": "3,0 A"}) == "B"

    def test_thousand_separator(self):
        assert match_to_choice("1000", {"A": "500", "B": "1.000", "C": "2.000"}) == "B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
