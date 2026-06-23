from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.v02_gamma as v02_gamma
import src.v03_gamma as v03_gamma


def test_v02_gamma_shim_forwards_to_v03_gamma(monkeypatch):
    captured = {}

    def fake_run_v03_gamma(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(v02_gamma, "run_v03_gamma", fake_run_v03_gamma)

    v02_gamma.run_v02_gamma(
        input_path="input.json",
        output_path="submission.csv",
        trace_output="trace.jsonl",
        safe_mode=True,
    )

    assert captured["input_path"] == "input.json"
    assert captured["output_path"] == "submission.csv"
    assert captured["trace_output"] == "trace.jsonl"
    assert captured["safe_mode"] is True


def test_run_sh_targets_v03_gamma_safe_mode():
    run_sh = (Path(__file__).resolve().parent.parent / "run.sh").read_text(encoding="utf-8")

    assert "src.v03_gamma" in run_sh
    assert "--safe-mode" in run_sh
    assert "/output/pred.csv" in run_sh
    assert "/data/private_test.csv" in run_sh
    assert "/data/public_test.csv" in run_sh


def test_v03_gamma_exports_main_runner():
    assert callable(v03_gamma.main)
    assert callable(v03_gamma.run_v03_gamma)


def test_v03_gamma_writes_complete_fallback_submission_on_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "pred.csv"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {"qid": "q1", "question": "2 + 2 = ?", "choices": ["3", "4"]},
                {"qid": "q2", "question": "Thủ đô Việt Nam là?", "choices": ["Hà Nội", "Huế"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(v03_gamma, "_load_agent", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    v03_gamma.run_v03_gamma(
        input_path=str(input_path),
        output_path=str(output_path),
        trace_output=str(trace_path),
        install_handlers=False,
    )

    assert output_path.exists()
    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"qid": "q1", "answer": "A"},
        {"qid": "q2", "answer": "A"},
    ]


def test_v03_gamma_preserves_checkpoint_answers_on_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "pred.csv"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {"qid": "q1", "question": "2 + 2 = ?", "choices": ["3", "4"]},
                {"qid": "q2", "question": "Thủ đô Việt Nam là?", "choices": ["Hà Nội", "Huế"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path.with_suffix(".ckpt").write_text(json.dumps({"q1": "B"}), encoding="utf-8")

    monkeypatch.setattr(v03_gamma, "_load_agent", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    v03_gamma.run_v03_gamma(
        input_path=str(input_path),
        output_path=str(output_path),
        trace_output=str(trace_path),
        install_handlers=False,
    )

    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"qid": "q1", "answer": "B"},
        {"qid": "q2", "answer": "A"},
    ]
