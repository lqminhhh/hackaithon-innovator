"""Accept tests for the S7 never-crash runner (src/run.py).

  Test 2  — G2 Fault isolation
  Test 4  — qid integrity (shuffle invariance)
  Test 6  — G4 signal (SIGTERM via subprocess)
  Test 7  — empty input → header-only CSV
  Test 8  — smoke: 5 questions, 5 valid rows, UTF-8

Tests 1, 3, 5 require killing a real process mid-run (checkpoint resume,
atomic-write kill, completeness-on-kill). They are included as subprocess
tests at the bottom — they are skipped if the `slow` marker is not requested
to keep the default suite fast.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FALLBACK
from src.run import run


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_questions(n: int, *, prefix: str = "q") -> list[dict]:
    return [
        {
            "qid": f"{prefix}{i:03d}",
            "question": f"Câu hỏi {i}?",
            "choices": ["A. Đúng", "B. Sai", "C. Không biết"],
        }
        for i in range(1, n + 1)
    ]


def _write_questions(path: Path, questions: list[dict]) -> None:
    path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _make_answer_fn(answers: dict[str, str]):
    """Return a solve_fn that returns pre-defined answers (no model needed)."""
    def solve_fn(q: dict):
        return answers[q["qid"]]
    return solve_fn


def _make_failing_fn(fail_on: str, answers: dict[str, str]):
    """Return a solve_fn that raises for ``fail_on`` and succeeds otherwise."""
    def solve_fn(q: dict):
        if q["qid"] == fail_on:
            raise RuntimeError(f"injected failure for {fail_on}")
        return answers[q["qid"]]
    return solve_fn


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Empty input → header-only CSV (§12 #7)
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_input_writes_header_only_csv(tmp_path):
    """§12 test 7: empty input must produce a header-only CSV without crashing."""
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, [])

    run(
        str(input_path),
        str(output_path),
        solve_fn=lambda q: "A",
        install_handlers=False,
    )

    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["qid", "answer"]], f"expected header only, got {rows}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Smoke: 5 questions → 5 valid rows, UTF-8 (§12 #8)
# ─────────────────────────────────────────────────────────────────────────────

def test_smoke_5_questions_5_valid_rows(tmp_path):
    """§12 test 8: smoke run with 5 questions produces 5 valid rows, UTF-8."""
    questions = _make_questions(5)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, questions)

    answers = {q["qid"]: "B" for q in questions}
    run(
        str(input_path),
        str(output_path),
        solve_fn=_make_answer_fn(answers),
        install_handlers=False,
    )

    rows = _read_csv(output_path)
    assert len(rows) == 5, f"expected 5 rows, got {len(rows)}"
    for row in rows:
        assert row["answer"] in "ABCDEFGHIJ", f"invalid answer: {row['answer']}"

    # UTF-8 check: file must be decodable as UTF-8
    output_path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — G2 Fault isolation (§12 #2)
# ─────────────────────────────────────────────────────────────────────────────

def test_fault_isolation_one_bad_question(tmp_path):
    """§12 test 2: injected failure on one qid → FALLBACK for that qid,
    correct answers for all others, no crash, complete CSV."""
    questions = _make_questions(5)
    bad_qid = questions[2]["qid"]  # q003
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, questions)

    answers = {q["qid"]: "C" for q in questions}
    run(
        str(input_path),
        str(output_path),
        solve_fn=_make_failing_fn(bad_qid, answers),
        install_handlers=False,
    )

    rows = {r["qid"]: r["answer"] for r in _read_csv(output_path)}
    assert len(rows) == 5, "must have exactly one row per qid"
    assert rows[bad_qid] == FALLBACK, f"failed qid must get FALLBACK, got {rows[bad_qid]}"
    for q in questions:
        if q["qid"] != bad_qid:
            assert rows[q["qid"]] == "C", f"good qid {q['qid']} got wrong answer"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — qid integrity / shuffle invariance (§12 #4)
# ─────────────────────────────────────────────────────────────────────────────

def test_qid_integrity_shuffled_input_same_answers(tmp_path):
    """§12 test 4: running on a shuffled copy of the input must produce the
    same answer for every qid. Catches position-based mapping bugs."""
    import random

    questions = _make_questions(6)
    answers = {q["qid"]: chr(ord("A") + i % 4) for i, q in enumerate(questions)}

    # Normal order
    input_a = tmp_path / "input_a.json"
    output_a = tmp_path / "out_a.csv"
    _write_questions(input_a, questions)
    run(
        str(input_a),
        str(output_a),
        solve_fn=_make_answer_fn(answers),
        install_handlers=False,
    )

    # Shuffled order
    shuffled = questions.copy()
    random.Random(42).shuffle(shuffled)
    input_b = tmp_path / "input_b.json"
    output_b = tmp_path / "out_b.csv"
    _write_questions(input_b, shuffled)
    run(
        str(input_b),
        str(output_b),
        solve_fn=_make_answer_fn(answers),
        install_handlers=False,
    )

    rows_a = {r["qid"]: r["answer"] for r in _read_csv(output_a)}
    rows_b = {r["qid"]: r["answer"] for r in _read_csv(output_b)}
    assert rows_a == rows_b, f"qid→answer mapping changed on shuffle:\n{rows_a}\n{rows_b}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2b — G1 completeness: FALLBACK pre-fill visible immediately
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_prefill_all_qids_have_row_even_on_all_failures(tmp_path):
    """All questions fail → every qid still gets a FALLBACK row (G1 + G2)."""
    questions = _make_questions(4)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, questions)

    def always_fail(q):
        raise RuntimeError("always fail")

    run(
        str(input_path),
        str(output_path),
        solve_fn=always_fail,
        install_handlers=False,
    )

    rows = {r["qid"]: r["answer"] for r in _read_csv(output_path)}
    assert set(rows.keys()) == {q["qid"] for q in questions}
    assert all(v == FALLBACK for v in rows.values())


# ─────────────────────────────────────────────────────────────────────────────
# Test 3b — Checkpoint written and loadable
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_is_written_and_loadable(tmp_path):
    """Checkpoint file is created after a run and contains the correct answers."""
    questions = _make_questions(3)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    ckpt_path = tmp_path / "ckpt.json"
    _write_questions(input_path, questions)

    answers = {q["qid"]: "D" for q in questions}
    run(
        str(input_path),
        str(output_path),
        solve_fn=_make_answer_fn(answers),
        checkpoint_path=str(ckpt_path),
        install_handlers=False,
    )

    assert ckpt_path.exists(), "checkpoint file must exist after run"
    with ckpt_path.open(encoding="utf-8") as f:
        ckpt = json.load(f)
    assert set(ckpt["answers"].keys()) == {q["qid"] for q in questions}
    assert all(v == "D" for v in ckpt["answers"].values())
    assert all(v == "done" for v in ckpt["status"].values())


# ─────────────────────────────────────────────────────────────────────────────
# Test G3 resume — already-done qids are not re-solved
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_skips_done_qids(tmp_path):
    """Resume: qids marked done in an existing checkpoint are not re-solved."""
    questions = _make_questions(4)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    ckpt_path = tmp_path / "ckpt.json"
    _write_questions(input_path, questions)

    # Pre-seed a checkpoint where q001 and q002 are already done
    pre_answers = {"q001": "B", "q002": "C"}
    pre_status = {"q001": "done", "q002": "done", "q003": "todo", "q004": "todo"}
    ckpt_path.write_text(
        json.dumps({"answers": pre_answers, "status": pre_status}, ensure_ascii=False),
        encoding="utf-8",
    )

    calls: list[str] = []

    def solve_fn(q):
        calls.append(q["qid"])
        return "D"

    run(
        str(input_path),
        str(output_path),
        solve_fn=solve_fn,
        checkpoint_path=str(ckpt_path),
        resume=True,
        install_handlers=False,
    )

    assert "q001" not in calls, "q001 was already done and must not be re-solved"
    assert "q002" not in calls, "q002 was already done and must not be re-solved"
    assert "q003" in calls
    assert "q004" in calls

    rows = {r["qid"]: r["answer"] for r in _read_csv(output_path)}
    assert rows["q001"] == "B"
    assert rows["q002"] == "C"
    assert rows["q003"] == "D"
    assert rows["q004"] == "D"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — G4 signal: SIGTERM → complete CSV written (§12 #6)
# ─────────────────────────────────────────────────────────────────────────────

def test_sigterm_writes_complete_csv(tmp_path):
    """§12 test 6: SIGTERM causes the runner subprocess to write a complete CSV
    before exiting. Every input qid must appear in the output."""
    questions = _make_questions(10)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, questions)

    # A tiny driver script that sleeps 0.2 s per question (gives us time to SIGTERM)
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(f"""\
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from src.run import run

            def slow_fn(q):
                time.sleep(0.2)
                return "B"

            run(
                {str(input_path)!r},
                {str(output_path)!r},
                solve_fn=slow_fn,
                checkpoint_path={str(tmp_path / "ckpt.json")!r},
            )
        """),
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(driver)])
    time.sleep(0.5)  # let it solve a couple of questions
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)

    assert output_path.exists(), "CSV must exist after SIGTERM"
    rows = _read_csv(output_path)
    qids_in_output = {r["qid"] for r in rows}
    qids_expected = {q["qid"] for q in questions}
    assert qids_in_output == qids_expected, (
        f"missing qids after SIGTERM: {qids_expected - qids_in_output}"
    )
    for r in rows:
        assert r["answer"] in "ABCDEFGHIJ", f"invalid answer {r['answer']!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests 1 / 3 / 5 — require real process kill (marked slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_resume_after_kill_mid_run(tmp_path):
    """§12 test 1 (slow): kill mid-run → restart → resumes from checkpoint,
    previously-solved answers intact."""
    questions = _make_questions(20)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    ckpt_path = tmp_path / "ckpt.json"
    _write_questions(input_path, questions)

    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(f"""\
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from src.run import run

            def slow_fn(q):
                time.sleep(0.15)
                return "C"

            run(
                {str(input_path)!r},
                {str(output_path)!r},
                solve_fn=slow_fn,
                checkpoint_path={str(ckpt_path)!r},
                checkpoint_every=3,
            )
        """),
        encoding="utf-8",
    )

    # First run — kill after ~4 questions
    proc = subprocess.Popen([sys.executable, str(driver)])
    time.sleep(0.8)
    proc.kill()
    proc.wait(timeout=5)

    assert ckpt_path.exists(), "checkpoint must exist after partial run"
    with ckpt_path.open(encoding="utf-8") as f:
        ckpt1 = json.load(f)
    done_after_kill = [qid for qid, s in ckpt1["status"].items() if s == "done"]
    assert len(done_after_kill) > 0, "at least one question should have been solved before kill"

    # Second run — should resume
    proc2 = subprocess.Popen([sys.executable, str(driver)])
    proc2.wait(timeout=30)

    rows = {r["qid"]: r["answer"] for r in _read_csv(output_path)}
    assert set(rows.keys()) == {q["qid"] for q in questions}
    for qid in done_after_kill:
        assert rows[qid] == "C", f"resumed qid {qid} should still have answer C"


@pytest.mark.slow
def test_completeness_on_kill_before_any_solve(tmp_path):
    """§12 test 5 (slow): kill before any question finishes → CSV still has
    one FALLBACK row per input qid (G1 via atexit/signal pre-fill)."""
    questions = _make_questions(10)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.csv"
    _write_questions(input_path, questions)

    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(f"""\
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from src.run import run

            def very_slow_fn(q):
                time.sleep(5)
                return "B"

            run(
                {str(input_path)!r},
                {str(output_path)!r},
                solve_fn=very_slow_fn,
            )
        """),
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(driver)])
    time.sleep(0.5)  # process started but no question finished yet
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)

    assert output_path.exists(), "CSV must exist even when killed before any solve"
    rows = _read_csv(output_path)
    qids = {r["qid"] for r in rows}
    assert qids == {q["qid"] for q in questions}
    assert all(r["answer"] == FALLBACK for r in rows), "all rows must be FALLBACK"
