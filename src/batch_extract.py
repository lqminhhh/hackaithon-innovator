"""Batched guided-choice extraction utilities.

These helpers are shared by wave-style runners that need to turn a batch of
reasoning prompts into one valid multiple-choice label per question.
"""

from __future__ import annotations

from typing import Any

from src.extract import ChoiceResult, best_label, build_label_token_map, softmax_margin
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
    from vllm import SamplingParams

    tokenizer = agent.tokenizer
    token_maps = [
        build_label_token_map(tokenizer, sorted(opts.keys()))
        for opts in options_list
    ]
    sampling_params_list = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            top_p=1.0,
            logprobs=min(len(token_map), 64),
            allowed_token_ids=list(token_map.keys()),
        )
        for token_map in token_maps
    ]

    raw_outputs = agent._llm.engine.generate(prompts, sampling_params_list)

    results: list[ChoiceResult] = []
    for i, output in enumerate(raw_outputs):
        try:
            scores: dict[str, float] = {label: float("-inf") for label in options_list[i]}
            logprobs_seq = getattr(output.outputs[0], "logprobs", None) or []
            if logprobs_seq:
                for token_id, entry in logprobs_seq[0].items():
                    label = token_maps[i].get(int(token_id))
                    if label is None:
                        continue
                    logprob = _get_logprob(entry)
                    if logprob is not None:
                        scores[label] = float(logprob)
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


def _get_logprob(entry: Any) -> float | None:
    """Extract a logprob float from a vLLM token-logprob entry object."""
    logprob = getattr(entry, "logprob", None)
    if logprob is not None:
        return float(logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (float, int)):
        return float(entry)
    return None
