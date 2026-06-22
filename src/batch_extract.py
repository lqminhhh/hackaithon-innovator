"""Batched guided-choice extraction utilities.

These helpers are shared by wave-style runners that need to turn a batch of
reasoning prompts into one valid multiple-choice label per question.
"""

from __future__ import annotations

from src.extract import (
    ChoiceResult,
    batch_score_continuations,
    best_label,
    build_label_token_map,
    softmax_margin,
)
from src.reasoning_agent import ReasoningAgent


def batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    """Batch guided-choice extraction.

    For vLLM, this uses per-request ``allowed_token_ids`` so heterogeneous
    choice sets can be scheduled in one engine call. For HuggingFace fallback,
    it scores labels sequentially through ``ReasoningAgent``.
    """
    if not prompts:
        return []
    if agent.is_vllm:
        return _vllm_batch_extract(agent, prompts, options_list)
    return _hf_batch_extract(agent, prompts, options_list)


def _vllm_batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    tokenizer = agent.tokenizer
    token_maps = [
        build_label_token_map(tokenizer, sorted(opts.keys()))
        for opts in options_list
    ]

    # Score each legal label explicitly via prompt-logprob continuation rather
    # than reading a single greedy decode's truncated top-k logprobs. The latter
    # only ever returned the sampled token as finite, collapsing every margin to
    # a degenerate constant and making every question escalate.
    scores_per_prompt = batch_score_continuations(
        agent._llm, tokenizer, list(zip(prompts, token_maps))
    )

    results: list[ChoiceResult] = []
    for i, scores in enumerate(scores_per_prompt):
        try:
            results.append(
                ChoiceResult(
                    letter=best_label(scores),
                    margin=softmax_margin(scores),
                    per_letter_logprob=scores,
                )
            )
        except Exception:
            results.append(_fallback_choice(options_list[i]))

    return results


def _hf_batch_extract(
    agent: ReasoningAgent,
    prompts: list[str],
    options_list: list[dict[str, str]],
) -> list[ChoiceResult]:
    results: list[ChoiceResult] = []
    for prompt, options in zip(prompts, options_list):
        valid_labels = tuple(sorted(options.keys()))
        try:
            scores = agent.score_valid_labels(prompt, valid_labels)
            results.append(
                ChoiceResult(
                    letter=best_label(scores),
                    margin=softmax_margin(scores),
                    per_letter_logprob=scores,
                )
            )
        except Exception:
            results.append(_fallback_choice(options))
    return results


def _fallback_choice(options: dict[str, str]) -> ChoiceResult:
    first_label = sorted(options.keys())[0]
    scores = {label: float("-inf") for label in options}
    scores[first_label] = 0.0
    return ChoiceResult(letter=first_label, margin=0.0, per_letter_logprob=scores)
