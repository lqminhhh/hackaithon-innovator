"""Entropy-Gated Jury — main orchestrator.

Flow (planning doc §3):
  1. Parse & flag     parsing.py
  2. Tier 1           batched greedy pass, masked answer token, logprobs recorded
  3. Confidence gate  top1−top2 logprob margin vs τ
  4. Tier 2 (jury)    escalated questions only
       a. Self-consistency n=6
       b. Gemma verdict
       c. Tool: code_exec (quantitative) / gated RAG (knowledge/legal)
       d. Resolution rules
  5. Assemble         invariant checks → submission.csv + audit_log.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assemble import assemble
from src.data_loader import load_questions
from src.gate import escalation_rate, should_escalate
from src.inference import (
    InferenceRequest,
    InferenceResult,
    build_backend,
    build_chat_messages,
    get_model_family,
)
from src.jury import JuryVerdict, run_jury
from src.parsing import ParsedQuestion, parse_questions
from src.prompts import (
    build_user_message,
    get_system_prompt,
    select_template,
    thinking_budget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


# ── prompt builder ────────────────────────────────────────────────────


def _make_request(
    q: ParsedQuestion,
    cfg: dict,
    model_family: str | None = None,
    temperature: float = 0.0,
    n_samples: int = 1,
    top_p: float = 1.0,
) -> InferenceRequest:
    """Build an InferenceRequest for the given question and backend."""
    if model_family is None:
        model_family = get_model_family(cfg, "primary")

    template = select_template(
        has_context=q.has_context,
        is_quantitative=q.is_quantitative,
    )
    system = get_system_prompt(template)
    user = build_user_message(
        query=q.query,
        options=q.options,
        context=q.context,
        template=template,
    )
    budget = thinking_budget(template, cfg)
    backend_type = cfg.get("backend", "llamacpp")

    if backend_type == "llamacpp":
        from src.inference import build_chat_prompt_llamacpp
        prompt = build_chat_prompt_llamacpp(system, user, budget, model_family)
        return InferenceRequest(
            prompt=prompt,
            allowed_letters=q.valid_letters,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
        )

    # vLLM — pass structured messages; VllmBackend.chat() applies the template
    messages = build_chat_messages(system, user)
    return InferenceRequest(
        prompt="",  # unused for vLLM
        allowed_letters=q.valid_letters,
        thinking_budget=budget,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        messages=messages,
    )


def _make_prompt(q: ParsedQuestion, cfg: dict, model_family: str | None = None) -> str:
    """Legacy helper — returns a raw prompt string (llamacpp only)."""
    if model_family is None:
        model_family = get_model_family(cfg, "primary")
    template = select_template(
        has_context=q.has_context,
        is_quantitative=q.is_quantitative,
    )
    system = get_system_prompt(template)
    user = build_user_message(
        query=q.query,
        options=q.options,
        context=q.context,
        template=template,
    )
    budget = thinking_budget(template, cfg)
    from src.inference import build_chat_prompt_llamacpp
    return build_chat_prompt_llamacpp(system, user, budget, model_family)


# ── tool runner ───────────────────────────────────────────────────────


def _run_tool(
    q: ParsedQuestion,
    qwen_backend,
    retriever,
    cfg: dict,
) -> str | None:
    """Run code_exec (quantitative) or gated RAG (knowledge/legal).

    Returns the answer letter or None on failure.
    """
    backend_type = cfg.get("backend", "llamacpp")

    if q.is_quantitative:
        from src.tools.code_exec import build_code_prompt, execute_code, match_to_choice

        code_system = get_system_prompt("code_exec")
        code_user = build_code_prompt(q.query, q.options)
        budget = thinking_budget("code_exec", cfg)

        code_text = _generate_full_text(
            qwen_backend,
            system=code_system,
            user=code_user,
            cfg=cfg,
        )
        if code_text is None:
            return None

        stdout, err = execute_code(code_text)
        if err or stdout is None:
            logger.debug("code_exec failed for %s: %s", q.qid, err)
            return None

        return match_to_choice(stdout, q.options)

    elif (q.is_legal or not q.has_context) and retriever is not None:
        context = retriever.retrieve_for_question(q.query, exclude_qid=q.qid)
        if context is None:
            return None  # gate rejected — flat profile (Lesson A)

        template = "evidence_extract" if q.has_context else "general_knowledge"
        system = get_system_prompt(template)
        user = build_user_message(q.query, q.options, context, template)
        budget = thinking_budget(template, cfg)

        if backend_type == "llamacpp":
            from src.inference import build_chat_prompt_llamacpp
            prompt = build_chat_prompt_llamacpp(system, user, budget, "qwen3")
            req = InferenceRequest(
                prompt=prompt,
                allowed_letters=q.valid_letters,
                thinking_budget=budget,
                temperature=0.0,
            )
        else:
            messages = build_chat_messages(system, user)
            req = InferenceRequest(
                prompt="",
                allowed_letters=q.valid_letters,
                thinking_budget=budget,
                temperature=0.0,
                messages=messages,
            )

        results = qwen_backend.generate_batch([req])
        return results[0][0].letter if results and results[0] else None

    return None


def _generate_full_text(
    backend,
    system: str,
    user: str,
    cfg: dict,
    max_tokens: int = 1024,
) -> str | None:
    """Free-form text generation (code_exec). Works for both backends."""
    backend_type = cfg.get("backend", "llamacpp")
    try:
        if backend_type == "vllm" and hasattr(backend, "generate_text"):
            messages = build_chat_messages(system, user)
            return backend.generate_text(messages, max_tokens=max_tokens)

        # LlamaCppBackend — build a prompt and call the underlying llm()
        from src.inference import build_chat_prompt_llamacpp
        budget = thinking_budget("code_exec", cfg)
        prompt = build_chat_prompt_llamacpp(system, user, budget, "qwen3")
        llm = backend._llm
        out = llm(prompt, max_tokens=max_tokens, temperature=0.0, logprobs=None)
        return out["choices"][0].get("text", "")
    except Exception as exc:
        logger.debug("Full-text generation failed: %s", exc)
    return None


# ── main pipeline ─────────────────────────────────────────────────────


def run_pipeline(
    input_path: str,
    output_path: str,
    audit_path: str | None = None,
    strict: bool = False,
):
    t_start = time.time()
    cfg = _load_config()

    if audit_path is None:
        audit_path = str(Path(output_path).with_suffix("")) + "_audit.json"

    # ── Load data ─────────────────────────────────────────────────────
    raw_questions = load_questions(input_path)
    questions: list[ParsedQuestion] = parse_questions(raw_questions)
    n = len(questions)
    logger.info("Loaded %d questions", n)

    flag_counts = {
        "has_context": sum(q.has_context for q in questions),
        "is_quantitative": sum(q.is_quantitative for q in questions),
        "has_refusal_choice": sum(q.has_refusal_choice for q in questions),
        "is_legal": sum(q.is_legal for q in questions),
    }
    logger.info("Flags: %s", flag_counts)

    # ── Load backends ─────────────────────────────────────────────────
    logger.info("Loading Qwen backend (primary)...")
    qwen_backend = build_backend(cfg, model_key="primary")

    gemma_backend = None
    if cfg.get("models", {}).get("secondary"):
        try:
            logger.info("Loading Gemma backend (secondary)...")
            gemma_backend = build_backend(cfg, model_key="secondary")
        except Exception as e:
            logger.warning("Secondary model unavailable: %s — proceeding without Gemma", e)

    # Pre-warm
    logger.info("Pre-warming backends...")
    qwen_backend.warmup()
    if gemma_backend:
        gemma_backend.warmup()

    # ── Load retriever (lazy) ─────────────────────────────────────────
    retriever = None
    if cfg.get("rag", {}).get("enabled", True):
        try:
            from sentence_transformers import SentenceTransformer
            from src.tools.retrieval import load_retriever

            logger.info("Loading BGE-M3 embedder for retrieval...")
            embedder_id = cfg["models"].get("embedder", "BAAI/bge-m3")
            embedder = SentenceTransformer(embedder_id)
            retriever = load_retriever(cfg, embedder)
            logger.info("Retriever ready.")
        except Exception as e:
            logger.warning("Retriever unavailable: %s — RAG disabled", e)

    # ── Tier 1: batched greedy pass over all questions ────────────────
    logger.info("Tier 1: batched greedy pass (%d prompts)...", n)
    t1_start = time.time()

    tier1_requests = [_make_request(q, cfg, temperature=0.0) for q in questions]

    tier1_nested = qwen_backend.generate_batch(tier1_requests)
    tier1_results: list[InferenceResult] = [samples[0] for samples in tier1_nested]

    logger.info("Tier 1 done in %.1fs", time.time() - t1_start)

    # ── Confidence gate ───────────────────────────────────────────────
    tau = cfg.get("gate", {}).get("tau", 1.5)
    escalate_flags: list[bool] = []
    escalate_reasons: list[str] = []
    for q, t1 in zip(questions, tier1_results):
        esc, reason = should_escalate(t1, q.n_choices, tau=tau)
        escalate_flags.append(esc)
        escalate_reasons.append(reason)

    tier1_accepted = [not e for e in escalate_flags]
    n_fast = sum(tier1_accepted)
    n_escalated = n - n_fast
    logger.info(
        "Gate: %d fast-exit (%.0f%%), %d escalated",
        n_fast, 100 * n_fast / n, n_escalated,
    )

    # ── Tier 2: jury on escalated questions ───────────────────────────
    jury_verdicts: dict[str, JuryVerdict] = {}

    if n_escalated > 0:
        logger.info("Tier 2: jury on %d questions...", n_escalated)
        t2_start = time.time()

        escalated = [
            (q, t1)
            for q, t1, esc in zip(questions, tier1_results, escalate_flags)
            if esc
        ]

        for q, t1 in escalated:
            # Run tool first (code_exec or RAG)
            tool_answer = None
            try:
                tool_answer = _run_tool(q, qwen_backend, retriever, cfg)
                if tool_answer:
                    logger.debug("qid=%s: tool → %s", q.qid, tool_answer)
            except Exception as e:
                logger.warning("Tool error for %s: %s", q.qid, e)

            try:
                verdict = run_jury(
                    question=q,
                    qwen_backend=qwen_backend,
                    gemma_backend=gemma_backend,
                    tier1_result=t1,
                    tool_answer=tool_answer,
                    cfg=cfg,
                )
                jury_verdicts[q.qid] = verdict
            except Exception as e:
                logger.error("Jury error for %s: %s — using tier-1 fallback", q.qid, e)
                jury_verdicts[q.qid] = JuryVerdict(
                    qid=q.qid,
                    answer=t1.letter,
                    tier=2,
                    resolution_rule=f"jury_error:{e}",
                )

        logger.info("Tier 2 done in %.1fs", time.time() - t2_start)

    # ── Assemble ──────────────────────────────────────────────────────
    logger.info("Assembling %d results...", n)
    df = assemble(
        questions=questions,
        tier1_results=tier1_results,
        tier1_accepted=tier1_accepted,
        jury_verdicts=jury_verdicts,
        output_csv=output_path,
        output_audit=audit_path,
        strict=strict,
    )

    elapsed = time.time() - t_start
    logger.info(
        "Done. %d predictions written to %s (%.1fs, %.2fs/q)",
        len(df), output_path, elapsed, elapsed / n,
    )
    return df


# ── entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Entropy-Gated Jury pipeline")
    parser.add_argument("--input", required=True, help="Input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Output submission CSV")
    parser.add_argument("--audit", default=None, help="Audit log JSON (default: <output>_audit.json)")
    parser.add_argument("--strict", action="store_true", help="Fail on invariant violations")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.audit, args.strict)


if __name__ == "__main__":
    main()
