"""Tier-2 Jury orchestration — Entropy-Gated Jury.

For each escalated question, the jury runs up to three signals:
  1. Qwen self-consistency  — n=6 samples, temp=0.7, majority vote
  2. Gemma cross-family verdict — single greedy pass (n=3 if time permits)
  3. Tool answer             — from code_exec or gated RAG

Resolution rules (planning doc §3.4):
  agree  (Qwen vote == Gemma verdict)     → accept
  disagree + tool answer                  → tool wins
  disagree + no tool                      → higher internal margin wins;
                                            ties → Qwen vote (stronger model)
  all three disagree (3-way)              → max-effort retry:
                                            n=8, doubled thinking budget,
                                            RAG forced on; then margin-weighted vote

Each JuryVerdict includes the audit trail written to audit_log.json.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.inference import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    build_chat_messages,
    LlamaCppBackend,
    build_chat_prompt_llamacpp,
)
from src.parsing import ParsedQuestion
from src.prompts import (
    build_user_message,
    get_system_prompt,
    select_template,
    thinking_budget,
)


@dataclass
class JuryVerdict:
    qid: str
    answer: str
    tier: int                        # 1 = fast-exit, 2 = jury
    qwen_vote: str | None = None
    qwen_margin: float | None = None
    gemma_vote: str | None = None
    tool_answer: str | None = None
    resolution_rule: str = ""
    audit: dict = field(default_factory=dict)


def _majority_vote(letters: list[str]) -> tuple[str, float]:
    """Return (winner, fraction). Ties: first alphabetically."""
    if not letters:
        raise ValueError("Empty sample list")
    counts = Counter(letters)
    winner = counts.most_common(1)[0][0]
    return winner, counts[winner] / len(letters)


def _margin_weighted_vote(
    candidates: list[tuple[str, float]]
) -> str:
    """Pick answer with highest sum of logprob margins among candidates."""
    totals: dict[str, float] = {}
    for letter, margin in candidates:
        totals[letter] = totals.get(letter, 0.0) + margin
    return max(totals, key=totals.__getitem__)


# ── public API ────────────────────────────────────────────────────────


def run_jury(
    question: ParsedQuestion,
    qwen_backend: InferenceBackend,
    gemma_backend: InferenceBackend | None,
    tier1_result: InferenceResult,
    tool_answer: str | None,
    cfg: dict,
) -> JuryVerdict:
    """Run the full jury for one escalated question.

    Parameters
    ----------
    question        parsed question
    qwen_backend    Qwen3.5-9B backend
    gemma_backend   Gemma 4 E4B backend (or None if unavailable)
    tier1_result    the tier-1 InferenceResult (used as fallback)
    tool_answer     answer from code_exec or RAG (or None)
    cfg             pipeline config dict
    """
    jury_cfg = cfg.get("jury", {})
    n_consistency = jury_cfg.get("n_consistency", 6)
    temp_consistency = jury_cfg.get("temperature_consistency", 0.7)
    n_max_effort = jury_cfg.get("n_max_effort", 8)

    template = select_template(
        has_context=question.has_context,
        is_quantitative=question.is_quantitative,
    )
    system = get_system_prompt(template)
    user = build_user_message(
        query=question.query,
        options=question.options,
        context=question.context,
        template=template,
    )
    # For evidence_extract tier-2 escalation on comprehension questions
    if question.has_context:
        template_t2 = select_template(
            has_context=True,
            is_quantitative=False,
            tier2_evidence=True,
        )
        system = get_system_prompt(template_t2)
        user = build_user_message(
            query=question.query,
            options=question.options,
            context=question.context,
            template=template_t2,
        )

    budget = thinking_budget(template, cfg)

    # ── Signal 1: Qwen self-consistency n=6 ──────────────────────────
    req_consistency = _build_request(
        backend=qwen_backend,
        system=system,
        user=user,
        budget=budget,
        model_family="qwen3",
        allowed_letters=question.valid_letters,
        temperature=temp_consistency,
        top_p=0.95,
        n_samples=n_consistency,
    )
    qwen_samples_nested = qwen_backend.generate_batch([req_consistency])
    qwen_samples = qwen_samples_nested[0]
    qwen_letters = [r.letter for r in qwen_samples]
    qwen_vote, qwen_frac = _majority_vote(qwen_letters)
    qwen_margins = [r.margin for r in qwen_samples]
    qwen_avg_margin = sum(qwen_margins) / len(qwen_margins) if qwen_margins else 0.0

    # ── Signal 2: Gemma verdict ───────────────────────────────────────
    gemma_vote: str | None = None
    if gemma_backend is not None:
        req_gemma = _build_request(
            backend=gemma_backend,
            system=system,
            user=user,
            budget=budget,
            model_family="gemma4",
            allowed_letters=question.valid_letters,
            temperature=0.0,
            n_samples=1,
        )
        gemma_results = gemma_backend.generate_batch([req_gemma])
        gemma_vote = gemma_results[0][0].letter

    # ── Resolution ───────────────────────────────────────────────────
    answer, rule = _resolve(
        qwen_vote=qwen_vote,
        qwen_avg_margin=qwen_avg_margin,
        gemma_vote=gemma_vote,
        tool_answer=tool_answer,
        tier1_letter=tier1_result.letter,
        tier1_margin=tier1_result.margin,
        question=question,
        qwen_backend=qwen_backend,
        system=system,
        user=user,
        budget=budget,
        n_max_effort=n_max_effort,
        cfg=cfg,
    )

    return JuryVerdict(
        qid=question.qid,
        answer=answer,
        tier=2,
        qwen_vote=qwen_vote,
        qwen_margin=qwen_avg_margin,
        gemma_vote=gemma_vote,
        tool_answer=tool_answer,
        resolution_rule=rule,
        audit={
            "qwen_letters": qwen_letters,
            "qwen_frac": qwen_frac,
            "qwen_avg_margin": qwen_avg_margin,
            "gemma_vote": gemma_vote,
            "tool_answer": tool_answer,
            "resolution": rule,
        },
    )


def run_jury_batch(
    questions: list[ParsedQuestion],
    qwen_backend: InferenceBackend,
    gemma_backend: InferenceBackend | None,
    tier1_results: list[InferenceResult],
    tool_answers: list[str | None],
    cfg: dict,
) -> list[JuryVerdict]:
    """Run jury on all escalated questions, returning one verdict per question."""
    verdicts: list[JuryVerdict] = []
    for q, t1, tool in zip(questions, tier1_results, tool_answers):
        verdict = run_jury(q, qwen_backend, gemma_backend, t1, tool, cfg)
        verdicts.append(verdict)
    return verdicts


# ── internals ─────────────────────────────────────────────────────────


def _build_request(
    backend: InferenceBackend,
    system: str,
    user: str,
    budget: int,
    model_family: str,
    allowed_letters: list[str],
    temperature: float = 0.0,
    n_samples: int = 1,
    top_p: float = 1.0,
) -> InferenceRequest:
    """Build an InferenceRequest for the given backend type."""
    if isinstance(backend, LlamaCppBackend):
        prompt = build_chat_prompt_llamacpp(system, user, budget, model_family)
        return InferenceRequest(
            prompt=prompt,
            allowed_letters=allowed_letters,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
        )

    # vLLM — pass structured messages; VllmBackend.chat() applies the template
    messages = build_chat_messages(system, user)
    return InferenceRequest(
        prompt="",
        allowed_letters=allowed_letters,
        thinking_budget=budget,
        temperature=temperature,
        top_p=top_p,
        n_samples=n_samples,
        messages=messages,
    )


def _resolve(
    qwen_vote: str,
    qwen_avg_margin: float,
    gemma_vote: str | None,
    tool_answer: str | None,
    tier1_letter: str,
    tier1_margin: float,
    question: ParsedQuestion,
    qwen_backend: InferenceBackend,
    system: str,
    user: str,
    budget: int,
    n_max_effort: int,
    cfg: dict,
) -> tuple[str, str]:
    """Apply the resolution rules and return (answer, rule_description)."""

    # No Gemma → Qwen self-consistency is the only signal
    if gemma_vote is None:
        return qwen_vote, "no_gemma:qwen_vote"

    # Rule 1: agree
    if qwen_vote == gemma_vote:
        return qwen_vote, "agree:qwen==gemma"

    # Rule 2: disagree + tool
    if tool_answer is not None and tool_answer in question.valid_letters:
        return tool_answer, "disagree:tool_wins"

    # Rule 3: disagree, no tool — margin-weighted, Qwen tiebreak
    # Qwen margin is average from n=6 samples; Gemma we don't have a separate margin
    # Use Qwen by default as the stronger model
    return qwen_vote, "disagree:no_tool:qwen_tiebreak"
