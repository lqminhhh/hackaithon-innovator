"""Minimal v02 alpha pipeline.

Architecture:
    questions -> parser -> router -> forced safety override OR
    route-specific prompt -> guided-choice scoring -> CSV output

This is the first runnable version of the route-aware design. It keeps the
system intentionally small: no retrieval, no confidence gating, no voting.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions, write_submission
from src.models import load_primary_model, load_vllm_primary
from src.parser import parse_question
from src.reasoning_agent import ReasoningAgent
from src.router import get_forced_answer, route_question

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "v02_alpha_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def run_v02_alpha(
    input_path: str,
    output_path: str,
    model_id: str | None = None,
    limit: int | None = None,
):
    cfg = _load_config()
    t_start = time.time()

    chosen_model = model_id or cfg["model"]["primary"]
    if torch.cuda.is_available():
        try:
            print(f"Loading primary model with vLLM: {chosen_model}", flush=True)
            llm = load_vllm_primary(model_id=chosen_model)
            agent = ReasoningAgent(llm=llm)
            print(f"Primary model loaded with vLLM in {time.time() - t_start:.1f}s", flush=True)
        except Exception as e:
            print(f"vLLM unavailable ({e}), falling back to HuggingFace", flush=True)
            model, tokenizer = load_primary_model(model_id=chosen_model)
            agent = ReasoningAgent(model=model, tokenizer=tokenizer)
            print(f"Primary model loaded in {time.time() - t_start:.1f}s", flush=True)
    else:
        print(f"Loading primary model: {chosen_model}", flush=True)
        model, tokenizer = load_primary_model(model_id=chosen_model)
        agent = ReasoningAgent(model=model, tokenizer=tokenizer)
        print(f"Primary model loaded in {time.time() - t_start:.1f}s", flush=True)

    questions = load_questions(input_path)
    if limit is None:
        limit = cfg.get("inference", {}).get("max_questions")
    if limit is not None:
        questions = questions[:limit]

    print(f"Processing {len(questions)} questions (v02_alpha)...", flush=True)

    results = []
    route_counts: Counter[str] = Counter()
    forced_count = 0
    run_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        parsed = parse_question(q)
        route = route_question(parsed)
        route_counts[route] += 1

        forced = get_forced_answer(parsed, route)
        if forced is not None:
            answer = forced
            forced_count += 1
        else:
            answer, _ = agent.predict_route_choice(
                route=route,
                question=parsed.query,
                options=parsed.options,
                context=parsed.context if route == "reading" else None,
            )

        results.append({"qid": parsed.qid, "answer": answer})

        q_elapsed = time.time() - q_start
        avg = (time.time() - run_start) / (i + 1)
        eta = avg * (len(questions) - i - 1)
        print(
            f"  [{i + 1}/{len(questions)}] {parsed.qid} "
            f"route={route} answer={answer} "
            f"({q_elapsed:.1f}s, avg {avg:.1f}s/q, ETA {eta / 60:.0f}min)",
            flush=True,
        )

    write_submission(results, output_path)
    total = time.time() - t_start
    infer_only = time.time() - run_start
    print(f"Written {len(results)} predictions to {output_path}", flush=True)
    print(f"Route counts: {dict(route_counts)}", flush=True)
    print(f"Forced safety answers: {forced_count}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer_only:.1f}s, {infer_only / max(len(questions), 1):.2f}s/question)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Minimal route-aware v02 alpha pipeline")
    parser.add_argument("--input", required=True, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model override. Defaults to configs/v02_alpha_config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions to process",
    )
    args = parser.parse_args()

    run_v02_alpha(args.input, args.output, args.model_id, args.limit)


if __name__ == "__main__":
    main()
