"""S0 fallback runner.

Loads a competition input file and writes a complete fallback submission.
This is intentionally model-free: it exists to prove the I/O contract and to
provide a safe baseline that never emits missing answers.
"""

from __future__ import annotations

import argparse

from src.config import FALLBACK
from src.data_loader import load_questions, write_submission


def run(input_path: str, output_path: str) -> None:
    questions = load_questions(input_path)
    rows = [{"qid": q["qid"], "answer": FALLBACK} for q in questions]
    write_submission(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="S0 fallback submission runner")
    parser.add_argument("--input", required=True, help="Path to input JSON or CSV")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
