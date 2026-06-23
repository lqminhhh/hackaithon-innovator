from __future__ import annotations

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
