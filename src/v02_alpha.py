"""v02 alpha pipeline with optional S6 RAG.

Architecture:
    questions -> parser -> router -> forced safety override OR
    route-specific prompt -> guided-choice scoring ->
    (knowledge, low-margin) -> RAG retrieval + re-answer OR self-consistency
    -> CSV output

RAG is enabled with ``--use-rag``. It requires a pre-built FAISS index
(run ``python scripts/build_vmlu_index.py`` first). Pass ``--no-reranker``
to skip the cross-encoder and use cosine-only scoring (saves ~1-2 GB VRAM).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions, write_submission
from src.models import load_primary_model, load_vllm_primary
from src.parser import parse_question
from src.reasoning_agent import ReasoningAgent
from src.solve import solve_question

if TYPE_CHECKING:
    from src.rag import RAGEngine
    from src.semantic_router import SemanticRouter

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "v02_alpha_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_rag(use_reranker: bool) -> "RAGEngine | None":
    """Load RAGEngine, returning None on any error so the run is never blocked."""
    try:
        from src.rag import RAGEngine
        from src.config import VMLU_INDEX_PATH, VMLU_CHUNKS_PATH

        if not Path(VMLU_INDEX_PATH).exists():
            print(
                f"  [RAG] Index not found at {VMLU_INDEX_PATH}. "
                "Run: python scripts/build_vmlu_index.py",
                flush=True,
            )
            return None
        if not Path(VMLU_CHUNKS_PATH).exists():
            print(
                f"  [RAG] Chunks file not found at {VMLU_CHUNKS_PATH}. "
                "Run: python scripts/build_vmlu_index.py",
                flush=True,
            )
            return None

        return RAGEngine(use_reranker=use_reranker)
    except Exception as exc:
        print(f"  [RAG] Failed to load RAGEngine ({exc}), running without RAG", flush=True)
        return None


def _load_semantic_router() -> "SemanticRouter | None":
    """Load S5 semantic router, returning None on error so inference can continue."""
    try:
        from src.semantic_router import SemanticRouter

        router = SemanticRouter()
        router.warmup()
        return router
    except Exception as exc:
        print(
            f"  [S5] Failed to load semantic router ({exc}), running without S5",
            flush=True,
        )
        return None


def run_v02_alpha(
    input_path: str,
    output_path: str,
    model_id: str | None = None,
    limit: int | None = None,
    use_rag: bool = False,
    use_reranker: bool = True,
    use_semantic_router: bool = False,
) -> None:
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

    rag: RAGEngine | None = None
    if use_rag:
        print(
            f"Loading RAG engine (reranker={'enabled' if use_reranker else 'disabled'})...",
            flush=True,
        )
        rag = _load_rag(use_reranker=use_reranker)
        if rag is not None:
            print("RAG engine ready.", flush=True)
        else:
            print("RAG unavailable; continuing without retrieval.", flush=True)

    semantic_router: SemanticRouter | None = None
    if use_semantic_router:
        print("Loading S5 semantic router...", flush=True)
        semantic_router = _load_semantic_router()
        if semantic_router is not None:
            print("S5 semantic router ready.", flush=True)
        else:
            print("S5 unavailable; continuing with Layer-1 routes only.", flush=True)

    questions = load_questions(input_path)
    if limit is None:
        limit = cfg.get("inference", {}).get("max_questions")
    if limit is not None:
        questions = questions[:limit]

    print(f"Processing {len(questions)} questions (v02_alpha)...", flush=True)

    results = []
    route_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    run_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        parsed = parse_question(q)
        solved = solve_question(agent, parsed, rag=rag, semantic_router=semantic_router)
        route_counts[solved.route] += 1
        path_counts[solved.path] += 1

        results.append({"qid": solved.qid, "answer": solved.answer})

        q_elapsed = time.time() - q_start
        avg = (time.time() - run_start) / (i + 1)
        eta = avg * (len(questions) - i - 1)
        margin_text = f"{solved.margin:.3f}" if solved.margin is not None else "n/a"
        votes_text = f" votes={''.join(solved.votes)}" if solved.votes else ""
        s5_text = _format_s5_log(solved)
        error_text = f" error={solved.error}" if solved.error else ""
        print(
            f"  [{i + 1}/{len(questions)}] {parsed.qid} "
            f"route={solved.route} path={solved.path} answer={solved.answer} "
            f"margin={margin_text}{votes_text}{s5_text}{error_text} "
            f"({q_elapsed:.1f}s, avg {avg:.1f}s/q, ETA {eta / 60:.0f}min)",
            flush=True,
        )

    write_submission(results, output_path)
    total = time.time() - t_start
    infer_only = time.time() - run_start
    print(f"Written {len(results)} predictions to {output_path}", flush=True)
    print(f"Route counts: {dict(route_counts)}", flush=True)
    print(f"Path counts: {dict(path_counts)}", flush=True)
    print(
        f"Total time: {total:.1f}s "
        f"(inference loop: {infer_only:.1f}s, {infer_only / max(len(questions), 1):.2f}s/question)",
        flush=True,
    )


def _format_s5_log(solved) -> str:
    if solved.semantic_error:
        return f" s5_error={solved.semantic_error}"
    if solved.semantic_route is None:
        return ""

    marker = "->" if solved.route_override else "~"
    source = solved.layer1_route or "?"
    return f" s5={source}{marker}{solved.semantic_route}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Route-aware v02 alpha pipeline with optional S6 RAG")
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
    parser.add_argument(
        "--use-rag",
        action="store_true",
        default=False,
        help=(
            "Enable S6 RAG for low-margin knowledge questions. "
            "Requires a pre-built index (run scripts/build_vmlu_index.py first)."
        ),
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        default=False,
        help=(
            "When --use-rag is set, skip the cross-encoder reranker and use "
            "cosine similarity scores only. Saves ~1-2 GB VRAM."
        ),
    )
    parser.add_argument(
        "--use-semantic-router",
        action="store_true",
        default=False,
        help=(
            "Enable S5 Layer-2 semantic route overrides. This is opt-in so "
            "baseline v02_alpha runs remain reproducible."
        ),
    )
    args = parser.parse_args()

    run_v02_alpha(
        args.input,
        args.output,
        args.model_id,
        args.limit,
        use_rag=args.use_rag,
        use_reranker=not args.no_reranker,
        use_semantic_router=args.use_semantic_router,
    )


if __name__ == "__main__":
    main()
