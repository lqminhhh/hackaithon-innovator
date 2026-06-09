"""Main pipeline orchestrator.

Wires all modules together: reads input (JSON or CSV), runs retrieval +
reasoning in parallel, gates by confidence, and writes submission.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import yaml

from src.data_loader import load_questions, write_submission
from src.models import load_primary_model, load_secondary_model, load_embedder
from src.retrieval_agent import RetrievalAgent
from src.reasoning_agent import ReasoningAgent
from src.confidence_gate import route
from src.consistency_sampler import adaptive_consistency
from src.ensemble_agent import ensemble_answer
from src.normaliser import normalise_answer, parse_confidence

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


async def process_question(
    q: dict,
    retriever: RetrievalAgent,
    primary_agent: ReasoningAgent,
    secondary_agent: ReasoningAgent | None,
    cfg: dict,
) -> dict:
    """Process a single question through the full pipeline."""
    question = q["question"]
    options = q["options"]
    loop = asyncio.get_running_loop()

    # Step 1: run retrieval and first CoT pass in parallel
    qid = q["qid"]
    retrieve_fut = loop.run_in_executor(None, retriever.retrieve, question, qid)
    cot_fut = loop.run_in_executor(
        None, primary_agent.infer_no_context, question, options
    )
    retrieved_chunks, raw_cot = await asyncio.gather(retrieve_fut, cot_fut)

    # Build context string from retrieved chunks (may be empty if gate filtered all)
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else None

    # If we have context, do a second pass with context injected
    if context:
        raw_cot = await loop.run_in_executor(
            None, primary_agent.infer_with_context, question, options, context
        )

    answer = normalise_answer(raw_cot)
    confidence = parse_confidence(raw_cot)

    # Step 2: confidence gate
    path = route(confidence)

    if path == "fast_exit":
        return {"qid": q["qid"], "answer": answer}

    if path == "consistency":
        answer, _ = await loop.run_in_executor(
            None, adaptive_consistency, primary_agent, question, options, context
        )
        return {"qid": q["qid"], "answer": answer}

    # path == "ensemble"
    if secondary_agent is not None:
        answer, _ = await loop.run_in_executor(
            None,
            ensemble_answer,
            primary_agent,
            secondary_agent,
            question,
            options,
            context,
        )
    else:
        answer, _ = await loop.run_in_executor(
            None, adaptive_consistency, primary_agent, question, options, context
        )
    return {"qid": q["qid"], "answer": answer}


async def run_pipeline(input_path: str, output_path: str):
    """Run the full pipeline end-to-end."""
    t_start = time.time()
    cfg = _load_config()

    print("Loading models...")
    primary_model, primary_tok = load_primary_model()
    primary_agent = ReasoningAgent(primary_model, primary_tok)

    secondary_agent: ReasoningAgent | None = None
    try:
        secondary_model, secondary_tok = load_secondary_model()
        secondary_agent = ReasoningAgent(secondary_model, secondary_tok)
        print("Secondary model (ensemble) loaded.")
    except Exception as e:
        print(f"Secondary model not available, skipping ensemble: {e}")

    embedder = load_embedder()
    retriever = RetrievalAgent(
        embedder=embedder,
        top_k=cfg["retrieval"]["top_k"],
        relevance_threshold=cfg["retrieval"]["relevance_threshold"],
    )
    print(f"Models loaded in {time.time() - t_start:.1f}s")

    questions = load_questions(input_path)
    print(f"Processing {len(questions)} questions...")

    results = []
    for i, q in enumerate(questions):
        result = await process_question(
            q, retriever, primary_agent, secondary_agent, cfg
        )
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(questions)}] done")

    write_submission(results, output_path)
    elapsed = time.time() - t_start
    print(f"Written {len(results)} predictions to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / len(questions):.2f}s/question)")


def main():
    parser = argparse.ArgumentParser(description="HackAIthon Bảng C pipeline")
    parser.add_argument("--input", required=True, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.input, args.output))


if __name__ == "__main__":
    main()
