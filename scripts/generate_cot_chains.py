#!/usr/bin/env python3
"""Generate CoT reasoning chains from REFERENCE questions.

Accepts either a single file or a directory of reference files.
When given a directory, all .json and .csv files inside it are
loaded and merged.  Chains are appended to the output file so you
can re-run as you add new reference sets without losing old chains.

IMPORTANT: Only use reference/sample questions here, NEVER the
active test file you will run inference on.

Usage:
    # Single file
    python scripts/generate_cot_chains.py --input data/reference/public_test.csv

    # Entire directory (all .json + .csv inside)
    python scripts/generate_cot_chains.py --input data/reference/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions
from src.models import load_primary_model
from src.reasoning_agent import ReasoningAgent
from src.normaliser import normalise_answer, parse_confidence


def _collect_questions(input_path: Path) -> list[dict]:
    """Load questions from a file or every .json/.csv in a directory."""
    if input_path.is_file():
        return load_questions(input_path)

    if input_path.is_dir():
        all_qs: list[dict] = []
        files = sorted(input_path.glob("*.json")) + sorted(input_path.glob("*.csv"))
        for f in files:
            qs = load_questions(f)
            print(f"  {f.name}: {len(qs)} questions")
            all_qs.extend(qs)
        return all_qs

    raise FileNotFoundError(f"Not a file or directory: {input_path}")


def _load_existing_qids(output_path: Path) -> set[str]:
    """Return qids already present in the output file (for skip-if-done)."""
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["qid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def main():
    parser = argparse.ArgumentParser(description="Generate CoT chains from reference questions")
    parser.add_argument(
        "--input",
        default="data/reference",
        help="Path to a reference file or directory of reference files",
    )
    parser.add_argument(
        "--output",
        default="data/cot_chains.jsonl",
        help="Path to output JSONL file (appends to existing)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("Loading model...")
    model, tokenizer = load_primary_model()
    agent = ReasoningAgent(model, tokenizer)

    print(f"Scanning {input_path} ...")
    questions = _collect_questions(input_path)
    print(f"Found {len(questions)} total questions")

    existing = _load_existing_qids(output_path)
    new_qs = [q for q in questions if q["qid"] not in existing]
    if not new_qs:
        print("All questions already have CoT chains. Nothing to do.")
        return
    if existing:
        print(f"Skipping {len(existing)} already-processed questions, {len(new_qs)} new")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for q in tqdm(new_qs):
            raw_output = agent.infer_no_context(q["question"], q["options"])
            valid_labels = tuple(sorted(q["options"].keys()))
            answer = normalise_answer(raw_output, valid_labels)
            confidence = parse_confidence(raw_output)

            record = {
                "qid": q["qid"],
                "question": q["question"],
                "options": q["options"],
                "correct_answer": answer,
                "confidence": confidence,
                "cot_chain": raw_output,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Total CoT chains in {output_path}: {len(existing) + len(new_qs)}")


if __name__ == "__main__":
    main()
