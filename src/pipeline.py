"""Main pipeline orchestrator.

Two execution paths:
  - **Batched (vLLM):** all questions processed in phases — retrieval,
    first CoT, context-augmented CoT, confidence gate, consistency.
    Each phase sends one big batch to vLLM for maximum GPU utilisation.
  - **Sequential (HuggingFace fallback):** one question at a time with
    async parallel retrieval + CoT, used when vLLM is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ── vLLM batched pipeline ────────────────────────────────────────────


def _run_batched(input_path: str, output_path: str):
    """Process all questions in bulk phases using vLLM."""
    t_start = time.time()
    cfg = _load_config()

    # Phase 0: retrieval (embedder on GPU for speed, freed before vLLM starts)
    print("Loading embedder for retrieval...", flush=True)
    embedder = load_embedder()
    print("  Loading retrieval index...", flush=True)
    retriever = RetrievalAgent(
        embedder=embedder,
        top_k=cfg["retrieval"]["top_k"],
        relevance_threshold=cfg["retrieval"]["relevance_threshold"],
    )

    questions = load_questions(input_path)
    print(f"Phase 1: Batch retrieval ({len(questions)} questions)...", flush=True)
    t1 = time.time()

    query_texts = [q["question"] for q in questions]
    exclude_qids = [q["qid"] for q in questions]
    all_chunk_lists = retriever.batch_retrieve(query_texts, exclude_qids)

    all_contexts: list[str | None] = [
        "\n\n".join(chunks) if chunks else None for chunks in all_chunk_lists
    ]

    n_ctx = sum(1 for c in all_contexts if c is not None)
    print(f"  {n_ctx}/{len(questions)} got context ({time.time() - t1:.1f}s)", flush=True)

    del retriever, embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 1: load vLLM engine
    print("Loading vLLM engine...", flush=True)
    from src.models import load_vllm_primary

    llm = load_vllm_primary()
    agent = ReasoningAgent(llm=llm)
    print(f"  Engine ready ({time.time() - t_start:.1f}s total)", flush=True)

    # Phase 2: first CoT pass (no context) — all questions at once
    print(f"Phase 2: First CoT pass ({len(questions)} prompts)...", flush=True)
    t2 = time.time()
    prompts_p2 = [
        agent.build_prompt(q["question"], q["options"]) for q in questions
    ]
    raw_outputs = agent.generate_batch(prompts_p2)
    print(f"  done ({time.time() - t2:.1f}s)", flush=True)

    # Phase 3: context-augmented pass for questions that got retrieval hits
    ctx_indices = [i for i, c in enumerate(all_contexts) if c is not None]
    if ctx_indices:
        print(
            f"Phase 3: Context-augmented pass ({len(ctx_indices)} prompts)...",
            flush=True,
        )
        t3 = time.time()
        prompts_p3 = [
            agent.build_prompt(
                questions[i]["question"], questions[i]["options"], all_contexts[i]
            )
            for i in ctx_indices
        ]
        ctx_outputs = agent.generate_batch(prompts_p3)
        for idx, output in zip(ctx_indices, ctx_outputs):
            raw_outputs[idx] = output
        print(f"  done ({time.time() - t3:.1f}s)", flush=True)

    # Phase 4: confidence gate
    results: list[dict | None] = [None] * len(questions)
    needs_consistency: list[int] = []
    n_fast = 0

    for i, (q, raw) in enumerate(zip(questions, raw_outputs)):
        valid_labels = tuple(sorted(q["options"].keys()))
        answer = normalise_answer(raw, valid_labels)
        confidence = parse_confidence(raw)
        path = route(confidence)

        if path == "fast_exit":
            results[i] = {"qid": q["qid"], "answer": answer}
            n_fast += 1
        else:
            needs_consistency.append(i)

    print(
        f"Phase 4: Gate — {n_fast} fast-exit, "
        f"{len(needs_consistency)} need consistency",
        flush=True,
    )

    # Phase 5: consistency sampling — batch all samples at once
    if needs_consistency:
        n_samples = cfg["consistency_sampler"]["n_max"]
        total = len(needs_consistency) * n_samples
        print(
            f"Phase 5: Consistency ({len(needs_consistency)} × {n_samples} = "
            f"{total} prompts)...",
            flush=True,
        )
        t5 = time.time()

        consistency_prompts: list[str] = []
        prompt_to_qidx: list[int] = []
        for idx in needs_consistency:
            q = questions[idx]
            ctx = all_contexts[idx]
            prompt = agent.build_prompt(q["question"], q["options"], ctx)
            for _ in range(n_samples):
                consistency_prompts.append(prompt)
                prompt_to_qidx.append(idx)

        temp = cfg["inference"]["temperature_sampling"]
        consistency_outputs = agent.generate_batch(
            consistency_prompts, temperature=temp
        )

        groups: dict[int, list[str]] = defaultdict(list)
        for idx, output in zip(prompt_to_qidx, consistency_outputs):
            q = questions[idx]
            valid_labels = tuple(sorted(q["options"].keys()))
            groups[idx].append(normalise_answer(output, valid_labels))

        for idx, answers in groups.items():
            best = Counter(answers).most_common(1)[0][0]
            results[idx] = {"qid": questions[idx]["qid"], "answer": best}

        print(f"  done ({time.time() - t5:.1f}s)", flush=True)

    # Phase 6: write
    write_submission(results, output_path)
    elapsed = time.time() - t_start
    print(f"Written {len(results)} predictions to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / len(questions):.2f}s/question)")


# ── HuggingFace sequential pipeline (fallback) ───────────────────────


async def _process_question_sequential(
    q: dict,
    retriever: RetrievalAgent,
    primary_agent: ReasoningAgent,
    secondary_agent: ReasoningAgent | None,
    cfg: dict,
) -> dict:
    """Process a single question through the full pipeline (HF path)."""
    question = q["question"]
    options = q["options"]
    valid_labels = tuple(sorted(options.keys()))
    loop = asyncio.get_running_loop()

    qid = q["qid"]
    retrieve_fut = loop.run_in_executor(None, retriever.retrieve, question, qid)
    cot_fut = loop.run_in_executor(
        None, primary_agent.infer_no_context, question, options
    )
    retrieved_chunks, raw_cot = await asyncio.gather(retrieve_fut, cot_fut)

    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else None
    if context:
        raw_cot = await loop.run_in_executor(
            None, primary_agent.infer_with_context, question, options, context
        )

    answer = normalise_answer(raw_cot, valid_labels)
    confidence = parse_confidence(raw_cot)
    path = route(confidence)

    if path == "fast_exit":
        return {"qid": q["qid"], "answer": answer}

    if path == "consistency":
        answer, _ = await loop.run_in_executor(
            None, adaptive_consistency, primary_agent, question, options, context
        )
        return {"qid": q["qid"], "answer": answer}

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


async def _run_sequential(input_path: str, output_path: str):
    """One-at-a-time pipeline using HuggingFace Transformers."""
    t_start = time.time()
    cfg = _load_config()

    print("Loading models (HuggingFace)...", flush=True)
    primary_model, primary_tok = load_primary_model()
    primary_agent = ReasoningAgent(model=primary_model, tokenizer=primary_tok)
    print("  Primary model loaded.", flush=True)

    secondary_agent: ReasoningAgent | None = None
    try:
        secondary_model, secondary_tok = load_secondary_model()
        secondary_agent = ReasoningAgent(model=secondary_model, tokenizer=secondary_tok)
        print("  Secondary model (ensemble) loaded.", flush=True)
    except Exception as e:
        print(f"  Secondary model not available, skipping ensemble: {e}", flush=True)

    embedder = load_embedder()
    retriever = RetrievalAgent(
        embedder=embedder,
        top_k=cfg["retrieval"]["top_k"],
        relevance_threshold=cfg["retrieval"]["relevance_threshold"],
    )
    print(f"Models loaded in {time.time() - t_start:.1f}s", flush=True)

    questions = load_questions(input_path)
    print(f"Processing {len(questions)} questions (sequential)...", flush=True)

    results = []
    for i, q in enumerate(questions):
        q_start = time.time()
        result = await _process_question_sequential(
            q, retriever, primary_agent, secondary_agent, cfg
        )
        results.append(result)
        q_elapsed = time.time() - q_start
        avg = (time.time() - t_start) / (i + 1)
        eta = avg * (len(questions) - i - 1)
        print(
            f"  [{i + 1}/{len(questions)}] {q['qid']} → {result['answer']} "
            f"({q_elapsed:.1f}s, avg {avg:.1f}s/q, ETA {eta / 60:.0f}min)",
            flush=True,
        )

    write_submission(results, output_path)
    elapsed = time.time() - t_start
    print(f"Written {len(results)} predictions to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / len(questions):.2f}s/question)")


# ── entry point ───────────────────────────────────────────────────────


def run_pipeline(input_path: str, output_path: str):
    """Auto-detect vLLM availability and run the best pipeline."""
    try:
        import vllm  # noqa: F401

        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires CUDA")
        print("vLLM detected — using batched pipeline", flush=True)
        _run_batched(input_path, output_path)
    except (ImportError, RuntimeError) as e:
        print(f"vLLM not available ({e}), falling back to sequential pipeline", flush=True)
        asyncio.run(_run_sequential(input_path, output_path))


def main():
    parser = argparse.ArgumentParser(description="HackAIthon Bảng C pipeline")
    parser.add_argument("--input", required=True, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
