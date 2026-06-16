"""S0 contract tests for config, I/O helpers, and fallback runner."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run as run_s0
from src.config import FALLBACK, LLM_MODEL
from src.data_loader import letters, load_questions, write_submission


def test_config_exposes_planned_defaults():
    assert LLM_MODEL == "Qwen/Qwen3-8B-AWQ"
    assert FALLBACK == "A"


def test_letters_supports_a_through_j():
    assert letters(1) == ["A"]
    assert letters(10) == list("ABCDEFGHIJ")


def test_letters_rejects_out_of_contract_counts():
    with pytest.raises(ValueError):
        letters(0)
    with pytest.raises(ValueError):
        letters(27)


def test_load_questions_preserves_vietnamese_and_normalises_choices(tmp_path):
    input_path = tmp_path / "input.json"
    payload = [
        {
            "qid": "test_0001",
            "question": "Thủ đô của Việt Nam là gì?",
            "choices": ["A. Hà Nội", "B. Huế", "C. Đà Nẵng"],
        }
    ]
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    questions = load_questions(input_path)

    assert questions == [
        {
            "qid": "test_0001",
            "question": "Thủ đô của Việt Nam là gì?",
            "options": {"A": "Hà Nội", "B": "Huế", "C": "Đà Nẵng"},
        }
    ]


def test_write_submission_outputs_qid_answer_columns(tmp_path):
    output_path = tmp_path / "submission.csv"

    write_submission(
        [
            {"qid": "test_0001", "answer": "A"},
            {"id": "test_0002", "answer": "B"},
        ],
        output_path,
    )

    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows == [
        ["qid", "answer"],
        ["test_0001", "A"],
        ["test_0002", "B"],
    ]


def test_write_submission_allows_empty_rows(tmp_path):
    output_path = tmp_path / "empty.csv"

    write_submission([], output_path)

    with output_path.open(encoding="utf-8", newline="") as f:
        assert list(csv.reader(f)) == [["qid", "answer"]]


def test_s0_runner_writes_one_fallback_per_input_qid(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    payload = [
        {"qid": "q1", "question": "Một?", "choices": ["Đúng", "Sai"]},
        {"qid": "q2", "question": "Hai?", "choices": ["Có", "Không"]},
    ]
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    run_s0(str(input_path), str(output_path))

    df = pd.read_csv(output_path)
    assert list(df.columns) == ["qid", "answer"]
    assert df.to_dict("records") == [
        {"qid": "q1", "answer": FALLBACK},
        {"qid": "q2", "answer": FALLBACK},
    ]
