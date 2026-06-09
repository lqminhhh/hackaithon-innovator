#!/usr/bin/env python3
"""Generate CoT reasoning chains for public test questions.

For every question in public_test.csv, runs the primary model with CoT
prompting and saves the full reasoning trace.  These chains are later
embedded into the FAISS index so the model can retrieve worked examples
for similar private-test questions.

Usage:
    python scripts/generate_cot_chains.py \
        --input data/public_test.csv \
        --output data/cot_chains.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_primary_model
from src.reasoning_agent import ReasoningAgent
from src.normaliser import normalise_answer, parse_confidence
from src.classifier import classify


def main():
    parser = argparse.ArgumentParser(description="Generate CoT chains for public test")
    parser.add_argument(
        "--input",
        default="data/public_test.csv",
        help="Path to public_test.csv",
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

    df = pd.read_csv(args.input)
    print(f"Processing {len(df)} questions...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            question = str(row["question"])
            options = {
                "A": str(row["A"]),
                "B": str(row["B"]),
                "C": str(row["C"]),
                "D": str(row["D"]),
            }

            q_type = classify(question, options)
            raw_output = agent.infer_no_context(question, options)
            answer = normalise_answer(raw_output)
            confidence = parse_confidence(raw_output)

            record = {
                "id": int(row["id"]) if "id" in row else int(row.name),
                "question": question,
                "options": options,
                "correct_answer": answer,
                "confidence": confidence,
                "cot_chain": raw_output,
                "q_type": q_type,
                "topic_tags": [],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} CoT chains to {output_path}")


if __name__ == "__main__":
    main()
