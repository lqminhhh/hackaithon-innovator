"""Dual-model ensemble agent (Qwen + Gemma).

Runs both models independently on the same question and fuses their
answers by confidence.  If both are low-confidence and disagree, falls
back to consistency sampling with N=7.
"""

from __future__ import annotations

from src.reasoning_agent import ReasoningAgent
from src.normaliser import normalise_answer, parse_confidence
from src.consistency_sampler import adaptive_consistency


def ensemble_answer(
    primary_agent: ReasoningAgent,
    secondary_agent: ReasoningAgent,
    question: str,
    options: dict[str, str],
    context: str | None = None,
) -> tuple[str, float]:
    """Run both models and fuse their answers.

    Returns (answer_letter, confidence).
    """
    valid_labels = tuple(sorted(options.keys()))

    # Run primary (Qwen)
    if context:
        raw_qwen = primary_agent.infer_with_context(question, options, context)
    else:
        raw_qwen = primary_agent.infer_no_context(question, options)
    answer_qwen = normalise_answer(raw_qwen, valid_labels)
    conf_qwen = parse_confidence(raw_qwen)

    # Run secondary (Gemma)
    if context:
        raw_gemma = secondary_agent.infer_with_context(question, options, context)
    else:
        raw_gemma = secondary_agent.infer_no_context(question, options)
    answer_gemma = normalise_answer(raw_gemma, valid_labels)
    conf_gemma = parse_confidence(raw_gemma)

    # Agreement → high confidence
    if answer_qwen == answer_gemma:
        return answer_qwen, max(conf_qwen, conf_gemma)

    # Disagreement — weight by confidence
    if conf_qwen < 0.5 and conf_gemma < 0.5:
        return adaptive_consistency(primary_agent, question, options, context)

    if conf_qwen >= conf_gemma:
        return answer_qwen, conf_qwen
    return answer_gemma, conf_gemma
