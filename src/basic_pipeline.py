"""Minimal LLM-only baseline pipeline.

Architecture:
    questions -> primary LLM -> answer normaliser -> CSV output

This intentionally skips retrieval, confidence routing, consistency
sampling, and ensemble logic so it can serve as a clean baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions, write_submission
from src.models import load_primary_model
from src.normaliser import normalise_answer
from src.reasoning_agent import ReasoningAgent

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "basic_pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def run_basic_pipeline(
    input_path: str,
    output_path: str,
    model_id: str | None = None,
    limit: int | None = None,
):
    """Run the LLM-only baseline end to end."""
    cfg = _load_config()
    t_start = time.time()

    chosen_model = model_id or cfg["model"]["primary"]
    print(f"Loading primary model: {chosen_model}", flush=True)
    model, tokenizer = load_primary_model(model_id=chosen_model)
    agent = ReasoningAgent(model=model, tokenizer=tokenizer)
    print(f"Primary model loaded in {time.time() - t_start:.1f}s", flush=True)

    questions = load_questions(input_path)
    if limit is None:
        limit = cfg.get("inference", {}).get("max_questions")
    if limit is not None:
        questions = questions[:limit]

    print(f"Processing {len(questions)} questions (LLM-only)...", flush=True)

    results = []
    run_start = time.time()
    for i, q in enumerate(questions):
        q_start = time.time()
        raw = agent.infer_no_context(q["question"], q["options"])
        valid_labels = tuple(sorted(q["options"].keys()))
        answer = normalise_answer(raw, valid_labels)
        results.append({"qid": q["qid"], "answer": answer})

        q_elapsed = time.time() - q_start
        avg = (time.time() - run_start) / (i + 1)
        eta = avg * (len(questions) - i - 1)
        print(
            f"  [{i + 1}/{len(questions)}] {q['qid']} -> {answer} "
            f"({q_elapsed:.1f}s, avg {avg:.1f}s/q, ETA {eta / 60:.0f}min)",
            flush=True,
        )

    write_submission(results, output_path)
    total = time.time() - t_start
    infer_only = time.time() - run_start
    print(f"Written {len(results)} predictions to {output_path}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer_only:.1f}s, {infer_only / max(len(questions), 1):.2f}s/question)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Basic LLM-only baseline pipeline")
    parser.add_argument("--input", required=True, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model override. Defaults to configs/basic_pipeline_config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions to process",
    )
    args = parser.parse_args()

    run_basic_pipeline(args.input, args.output, args.model_id, args.limit)


if __name__ == "__main__":
    main()
