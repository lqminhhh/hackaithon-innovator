from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.v02_gamma as v02_gamma
import src.v03_gamma as v03_gamma
import predict


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

    assert "predict.py" in run_sh
    assert "/code/private_test.json" in run_sh
    assert "/code/submission.csv" in run_sh
    assert "/code/submission_time.csv" in run_sh
    assert "/tmp/trace_v03_gamma.jsonl" in run_sh
    assert "/data/private_test.csv" in run_sh
    assert "/data/public_test.csv" in run_sh


def test_inference_sh_targets_predict_py():
    inference_sh = (Path(__file__).resolve().parent.parent / "inference.sh").read_text(encoding="utf-8")

    assert "predict.py" in inference_sh


def test_predict_writes_submission_time(tmp_path):
    submission = tmp_path / "submission.csv"
    timing = tmp_path / "submission_time.csv"
    submission.write_text("qid,answer\nq1,A\nq2,B\n", encoding="utf-8")

    predict._write_submission_time(str(submission), str(timing), elapsed=4.0)

    with timing.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"qid": "q1", "answer": "A", "time": "2.000000"},
        {"qid": "q2", "answer": "B", "time": "2.000000"},
    ]


def test_predict_writes_submission_time_from_trace(tmp_path):
    submission = tmp_path / "submission.csv"
    timing = tmp_path / "submission_time.csv"
    trace = tmp_path / "trace.jsonl"
    submission.write_text("qid,answer\nq1,A\nq2,B\n", encoding="utf-8")
    trace.write_text(
        "\n".join(
            [
                json.dumps({"qid": "q1", "attributed_time_seconds": 1.25}),
                json.dumps({"qid": "q2", "attributed_time_seconds": 2.75}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    predict._write_submission_time(
        str(submission),
        str(timing),
        elapsed=10.0,
        trace_path=str(trace),
    )

    with timing.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"qid": "q1", "answer": "A", "time": "1.250000"},
        {"qid": "q2", "answer": "B", "time": "2.750000"},
    ]


def test_v03_gamma_exports_main_runner():
    assert callable(v03_gamma.main)
    assert callable(v03_gamma.run_v03_gamma)


def test_dynamic_vllm_attempt_clamps_to_available_free_memory(monkeypatch):
    gib = 1024 ** 3

    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return int(8 * gib), int(16 * gib)

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    attempt = v03_gamma._dynamic_vllm_attempt(
        headroom_gb=1.0,
        headroom_index=0,
        clamp_min=0.50,
        clamp_max=0.92,
    )

    assert attempt["gpu_memory_free_gb"] == 8.0
    assert attempt["gpu_memory_total_gb"] == 16.0
    assert attempt["gpu_memory_utilization"] == 0.5
    assert attempt["gpu_memory_utilization_unclamped"] == 0.4375


def test_load_agent_retries_vllm_headroom_before_hf(monkeypatch):
    monkeypatch.setattr(v03_gamma, "SAFE_HEADROOM_LADDER_GB", (1.0, 2.0, 3.0))
    monkeypatch.setattr(v03_gamma, "SAFE_UTILIZATION_CLAMP_MIN", 0.50)
    monkeypatch.setattr(v03_gamma, "SAFE_UTILIZATION_CLAMP_MAX", 0.92)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            gib = 1024 ** 3
            return int(14 * gib), int(16 * gib)

        @staticmethod
        def empty_cache():
            return None

    class _FakeTorch:
        cuda = _FakeCuda()
        OutOfMemoryError = RuntimeError

    attempts: list[float] = []

    class _FakeLLM:
        def __init__(self, util):
            self.model = "fake-model"
            self.enable_prefix_caching = True
            self.engine = None
            self.gpu_memory_utilization = util

    def fake_load_vllm_primary(**kwargs):
        attempts.append(kwargs["gpu_memory_utilization"])
        if len(attempts) < 3:
            raise RuntimeError("CUDA out of memory")
        return _FakeLLM(kwargs["gpu_memory_utilization"])

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr("src.models.load_vllm_primary", fake_load_vllm_primary)
    monkeypatch.setattr("src.models.load_primary_model", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("HF fallback should not run")))

    agent = v03_gamma._load_agent(
        model_id=None,
        gpu_memory_utilization=None,
        max_model_len=4096,
        max_num_seqs=32,
        t_start=0.0,
        safe_mode=True,
    )

    assert len(attempts) == 3
    assert agent._backend_info["backend"] == "vllm"
    assert agent._backend_info["vllm_headroom_index"] == 2
    assert agent._backend_info["gpu_memory_headroom_gb"] == 3.0
    assert agent._backend_info["dynamic_vram_sizing"] is True
    assert len(agent._backend_info["vllm_attempts"]) == 3


def test_run_v03_gamma_retries_wave_on_oom_like_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "pred.csv"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        json.dumps(
            [{"qid": "q1", "question": "2 + 2 = ?", "choices": ["3", "4"]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    load_calls: list[int] = []

    class _FakeAgent:
        is_vllm = True

        def __init__(self, headroom_index: int):
            self._llm = None
            self._backend_info = {
                "backend": "vllm",
                "backend_reason": "loaded_vllm",
                "gpu_memory_utilization_requested": None,
                "gpu_memory_headroom_gb": float(headroom_index + 1),
                "vllm_headroom_index": headroom_index,
                "vllm_headroom_ladder_gb": [1.0, 2.0, 3.0],
            }

    def fake_load_agent(**kwargs):
        idx = int(kwargs.get("min_headroom_index", 0))
        load_calls.append(idx)
        return _FakeAgent(idx)

    wave1_calls = {"count": 0}

    def fake_run_wave1(_agent, parsed_list, _skip_qids, *, chunk_size=None):
        wave1_calls["count"] += 1
        if wave1_calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        qid = parsed_list[0].qid
        return {
            qid: type(
                "W1",
                (),
                {
                    "qid": qid,
                    "answer": "B",
                    "route": "stem",
                    "margin": 1.0,
                    "forced": False,
                    "error": None,
                    "reasoning_prompt": "",
                    "per_letter_logprob": {"A": -1.0, "B": 0.0},
                },
            )()
        }

    monkeypatch.setattr(v03_gamma, "_load_agent", fake_load_agent)
    monkeypatch.setattr(v03_gamma, "_warmup_agent", lambda _agent: None)
    monkeypatch.setattr("src.wave_solver.run_wave1", fake_run_wave1)
    monkeypatch.setattr("src.wave_solver.run_wave2", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.wave_solver.finalize_answers", lambda parsed_list, _wave1, _wave2, _answers: {parsed.qid: "B" for parsed in parsed_list})
    monkeypatch.setattr("src.wave_solver.write_traces", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.wave_solver.path_counts", lambda *_args, **_kwargs: {"wave_direct": 1})

    v03_gamma.run_v03_gamma(
        input_path=str(input_path),
        output_path=str(output_path),
        trace_output=str(trace_path),
        install_handlers=False,
    )

    assert load_calls == [0, 1]
    assert wave1_calls["count"] == 2
    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"qid": "q1", "answer": "B"}]


def test_run_v03_gamma_retries_wave_with_chunked_fallback(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "pred.csv"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        json.dumps(
            [{"qid": "q1", "question": "2 + 2 = ?", "choices": ["3", "4"]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(v03_gamma, "SAFE_WAVE_RETRY_CHUNK_SIZES", (32,))

    load_calls: list[int] = []

    class _FakeAgent:
        is_vllm = True

        def __init__(self, headroom_index: int):
            self._llm = None
            self._backend_info = {
                "backend": "vllm",
                "backend_reason": "loaded_vllm",
                "gpu_memory_utilization_requested": None,
                "gpu_memory_headroom_gb": float(headroom_index + 1),
                "vllm_headroom_index": headroom_index,
                "vllm_headroom_ladder_gb": [1.0, 2.0, 3.0],
            }

    def fake_load_agent(**kwargs):
        idx = int(kwargs.get("min_headroom_index", 0))
        load_calls.append(idx)
        return _FakeAgent(idx)

    wave1_calls: list[int | None] = []

    def fake_run_wave1(_agent, parsed_list, _skip_qids, *, chunk_size=None):
        wave1_calls.append(chunk_size)
        if chunk_size is None:
            raise RuntimeError("CUDA out of memory")
        qid = parsed_list[0].qid
        return {
            qid: type(
                "W1",
                (),
                {
                    "qid": qid,
                    "answer": "B",
                    "route": "stem",
                    "margin": 1.0,
                    "forced": False,
                    "error": None,
                    "reasoning_prompt": "",
                    "per_letter_logprob": {"A": -1.0, "B": 0.0},
                },
            )()
        }

    monkeypatch.setattr(v03_gamma, "_load_agent", fake_load_agent)
    monkeypatch.setattr(v03_gamma, "_warmup_agent", lambda _agent: None)
    monkeypatch.setattr("src.wave_solver.run_wave1", fake_run_wave1)
    monkeypatch.setattr("src.wave_solver.run_wave2", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.wave_solver.finalize_answers", lambda parsed_list, _wave1, _wave2, _answers: {parsed.qid: "B" for parsed in parsed_list})
    monkeypatch.setattr("src.wave_solver.write_traces", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.wave_solver.path_counts", lambda *_args, **_kwargs: {"wave_direct": 1})

    v03_gamma.run_v03_gamma(
        input_path=str(input_path),
        output_path=str(output_path),
        trace_output=str(trace_path),
        install_handlers=False,
    )

    assert load_calls == [0, 1, 1]
    assert wave1_calls == [None, None, 32]
    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"qid": "q1", "answer": "B"}]


def test_v03_gamma_runs_additive_warmup_before_pipeline(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "pred.csv"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        json.dumps(
            [{"qid": "q1", "question": "2 + 2 = ?", "choices": ["3", "4"]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    class _FakeAgent:
        is_vllm = True

    monkeypatch.setattr(v03_gamma, "_load_agent", lambda **_kwargs: _FakeAgent())
    monkeypatch.setattr(v03_gamma, "_warmup_agent", lambda _agent: calls.append("warmup"))

    def fake_run_wave1(_agent, parsed_list, _skip_qids, *, chunk_size=None):
        calls.append("wave1")
        qid = parsed_list[0].qid
        return {
            qid: type(
                "W1",
                (),
                {
                    "qid": qid,
                    "answer": "B",
                    "route": "stem",
                    "margin": 1.0,
                    "forced": False,
                    "error": None,
                    "reasoning_prompt": "",
                    "per_letter_logprob": {"A": -1.0, "B": 0.0},
                },
            )()
        }

    def fake_run_wave2(_agent, _parsed_list, _wave1, adaptive_sc=True, chunk_size=None):
        calls.append("wave2")
        return {}

    monkeypatch.setattr("src.wave_solver.run_wave1", fake_run_wave1)
    monkeypatch.setattr("src.wave_solver.run_wave2", fake_run_wave2)
    monkeypatch.setattr("src.wave_solver.finalize_answers", lambda parsed_list, _wave1, _wave2, _answers: {parsed.qid: "B" for parsed in parsed_list})
    monkeypatch.setattr("src.wave_solver.write_traces", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.wave_solver.path_counts", lambda *_args, **_kwargs: {"wave_direct": 1})

    v03_gamma.run_v03_gamma(
        input_path=str(input_path),
        output_path=str(output_path),
        trace_output=str(trace_path),
        install_handlers=False,
    )

    assert calls[:3] == ["warmup", "wave1", "wave2"]


def test_v03_gamma_warmup_is_noop_for_non_vllm():
    class _FakeAgent:
        is_vllm = False

    v03_gamma._warmup_agent(_FakeAgent())


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
