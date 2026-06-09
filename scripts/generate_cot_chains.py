#!/usr/bin/env python3
"""Generate CoT reasoning chains from SAMPLE/REFERENCE questions.

Runs the primary model over each sample question and saves the full
reasoning trace.  These chains are embedded into the FAISS index so
the model can retrieve worked examples at inference time.

IMPORTANT: Only run this on sample/reference questions (public_test.csv),
NEVER on the actual test file you will run inference on.  The test file
is for inference only.

Usage:
    python scripts/generate_cot_chains.py \
        --input data/public_test.csv \
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
        default="data/public_test.csv",
        help="Path to sample/reference questions (NOT the test file)",
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
