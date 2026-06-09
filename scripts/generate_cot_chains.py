#!/usr/bin/env python3
"""Generate CoT reasoning chains for public test questions.

For every question in the test file, runs the primary model with CoT
prompting and saves the full reasoning trace.  These chains are later
embedded into the FAISS index so the model can retrieve worked examples
for similar private-test questions.

Usage:
    python scripts/generate_cot_chains.py \
        --input data/public-test_1780368312.json \
        --output data/cot_chains.jsonl
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


def main():
    parser = argparse.ArgumentParser(description="Generate CoT chains for public test")
    parser.add_argument(
        "--input",
        default="data/public-test_1780368312.json",
        help="Path to test file (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        default="data/cot_chains.jsonl",
        help="Path to output JSONL file",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_primary_model()
    agent = ReasoningAgent(model, tokenizer)

    questions = load_questions(args.input)
    print(f"Processing {len(questions)} questions...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in tqdm(questions):
            raw_output = agent.infer_no_context(q["question"], q["options"])
            answer = normalise_answer(raw_output)
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

    print(f"Saved {len(questions)} CoT chains to {output_path}")


if __name__ == "__main__":
    main()
