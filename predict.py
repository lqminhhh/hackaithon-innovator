"""BTC-compatible submission entrypoint.

This file is intentionally a thin wrapper around ``src.v03_gamma``. It adapts
file names and emits ``submission_time.csv`` without changing model, routing,
prompting, or accuracy logic.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.v03_gamma import run_v03_gamma


DEFAULT_INPUT_CANDIDATES = (
    "/code/private_test.json",
    "/code/private_test.csv",
    "/data/private_test.csv",
    "/data/public_test.csv",
    "/data/private_test.json",
    "/data/public_test.json",
)


def _find_input(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    joined = ", ".join(DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No input file found. Expected one of: {joined}")


def _write_submission_time(submission_path: str, time_path: str, elapsed: float) -> None:
    submission = Path(submission_path)
    target = Path(time_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not submission.exists():
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "answer", "time"])
            writer.writeheader()
        return

    with submission.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    per_sample_time = elapsed / len(rows) if rows else 0.0
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "time"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "qid": row.get("qid", row.get("id", "")),
                "answer": row.get("answer", ""),
                "time": f"{per_sample_time:.6f}",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VietMind MCQ final pipeline.")
    parser.add_argument("--input", default=None, help="Input JSON/CSV path.")
    parser.add_argument("--output", default="/code/submission.csv", help="Output submission CSV.")
    parser.add_argument(
        "--time-output",
        default="/code/submission_time.csv",
        help="Output timing CSV with qid,answer,time.",
    )
    parser.add_argument(
        "--trace-output",
        default="/tmp/trace_v03_gamma.jsonl",
        help="Optional trace JSONL path.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-safe-mode", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    input_path = _find_input(args.input)
    start = time.perf_counter()
    run_v03_gamma(
        input_path=input_path,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        limit=args.limit,
        safe_mode=not args.no_safe_mode,
        warmup=not args.no_warmup,
    )
    elapsed = time.perf_counter() - start
    _write_submission_time(args.output, args.time_output, elapsed)
    print(f"Written timing file to {args.time_output}", flush=True)


if __name__ == "__main__":
    main()
