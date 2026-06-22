"""S2 guided-choice extraction with logprob margin.

This module is the v2 answer-extraction boundary: callers get a valid option
letter plus confidence evidence, not free-form model text that must be parsed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    letter: str
    margin: float
    per_letter_logprob: dict[str, float]


def build_choice_prompt(
    question: str,
    options: dict[str, str],
    context: str | None = None,
) -> str:
    """Build a concise prompt that ends exactly at the answer slot."""
    labels = sorted(options.keys())
    options_block = "\n".join(f"{label}) {options[label]}" for label in labels)
    label_list = ", ".join(labels)

    context_block = ""
    if context is not None:
        context_block = (
            "Thông tin tham khảo:\n"
            "---\n"
            f"{context}\n"
            "---\n\n"
        )

    return (
        "Bạn là một chuyên gia giải câu hỏi trắc nghiệm tiếng Việt.\n\n"
        f"{context_block}"
        "Câu hỏi:\n"
        f"{question}\n\n"
        "Các lựa chọn:\n"
        f"{options_block}\n\n"
        f"Chọn đúng một đáp án hợp lệ trong các lựa chọn sau: {label_list}.\n"
        "Chỉ trả lời bằng một ký tự đáp án.\n"
        "Đáp án: "
    )


def softmax_margin(logprobs: dict[str, float]) -> float:
    """Return prob(top1) - prob(top2) after softmax over finite logprobs."""
    finite = [value for value in logprobs.values() if math.isfinite(value)]
    if not finite:
        raise ValueError("cannot compute margin without finite logprobs")
    if len(finite) == 1:
        return 1.0

    max_logprob = max(finite)
    probs = [math.exp(value - max_logprob) for value in finite]
    total = sum(probs)
    normalized = sorted((p / total for p in probs), reverse=True)
    return float(normalized[0] - normalized[1])


def safe_margin(logprobs: dict[str, float], expected_labels: int) -> float:
    """Return a conservative margin when too few label scores are available."""
    finite_count = sum(math.isfinite(value) for value in logprobs.values())
    if expected_labels >= 2 and finite_count < 2:
        return 0.0
    return softmax_margin(logprobs)


def best_label(logprobs: dict[str, float]) -> str:
    """Return the highest-logprob label, requiring at least one finite value."""
    finite_items = {
        label: value for label, value in logprobs.items() if math.isfinite(value)
    }
    if not finite_items:
        raise ValueError("cannot choose a label without finite logprobs")
    return max(finite_items, key=finite_items.get)


def batch_score_continuations(
    llm,
    tokenizer,
    items: list[tuple[str, dict[int, str]]],
    *,
    sampling_params: Any | None = None,
) -> list[dict[str, float]]:
    """Score every legal label by its prompt-logprob continuation."""
    scores: list[dict[str, float]] = [
        {label: float("-inf") for label in token_map.values()}
        for _prompt, token_map in items
    ]

    requests: list[dict[str, list[int]]] = []
    owners: list[tuple[int, str, int]] = []
    for idx, (prompt, token_map) in enumerate(items):
        base_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
        for token_id, label in token_map.items():
            requests.append({"prompt_token_ids": base_ids + [int(token_id)]})
            owners.append((idx, label, int(token_id)))

    if not requests:
        return scores

    params = (
        sampling_params
        if sampling_params is not None
        else _continuation_sampling_params(llm)
    )
    raw_outputs = _raw_generate(llm, requests, params)

    for output, (idx, label, token_id) in zip(raw_outputs, owners):
        logprob = _continuation_logprob(output, token_id)
        if logprob is not None:
            scores[idx][label] = logprob

    return scores


def build_label_token_map(
    tokenizer,
    valid_labels: Iterable[str],
) -> dict[int, str]:
    """Map one token id to each valid answer label.

    The extraction prompt ends with a trailing space after ``Đáp án:``, so the
    bare label token is preferred. Some tokenizers only expose a whitespace
    prefixed single-token form, so we keep that as a fallback.
    """
    token_map: dict[int, str] = {}
    for label in valid_labels:
        for variant in (label, f" {label}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                token_map[int(token_ids[0])] = label
                break
    return token_map


class GuidedChoiceExtractor:
    """Constrain answer generation to legal labels and return logprob margin."""

    def __init__(
        self,
        llm,
        tokenizer=None,
        *,
        max_logprobs: int = 64,
    ):
        self.llm = llm
        self.tokenizer = tokenizer if tokenizer is not None else llm.get_tokenizer()
        self.max_logprobs = max_logprobs

    def predict(
        self,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> ChoiceResult:
        prompt = build_choice_prompt(question, options, context)
        return self.extract(prompt, sorted(options.keys()))

    def extract(self, prompt: str, valid_labels: Iterable[str]) -> ChoiceResult:
        labels = tuple(valid_labels)
        token_map = build_label_token_map(self.tokenizer, labels)
        missing = sorted(set(labels) - set(token_map.values()))
        if missing:
            raise ValueError(f"could not derive single-token ids for labels: {missing}")

        per_letter_logprob = batch_score_continuations(
            self.llm,
            self.tokenizer,
            [(prompt, token_map)],
            sampling_params=self._sampling_params(token_map),
        )[0]
        letter = best_label(per_letter_logprob)
        margin = safe_margin(per_letter_logprob, len(labels))
        return ChoiceResult(
            letter=letter,
            margin=margin,
            per_letter_logprob=per_letter_logprob,
        )

    def _sampling_params(self, token_map: dict[int, str]):
        kwargs = {
            "temperature": 0.0,
            "max_tokens": 1,
            "top_p": 1.0,
            "prompt_logprobs": 1,
        }
        if hasattr(self.llm, "sampling_params"):
            return self.llm.sampling_params(**kwargs)

        from vllm import SamplingParams

        return SamplingParams(**kwargs)


def _continuation_sampling_params(llm):
    kwargs = {
        "temperature": 0.0,
        "max_tokens": 1,
        "top_p": 1.0,
        "prompt_logprobs": 1,
    }
    if hasattr(llm, "sampling_params"):
        return llm.sampling_params(**kwargs)

    from vllm import SamplingParams

    return SamplingParams(**kwargs)


def _raw_generate(llm, prompts, params):
    if hasattr(llm, "generate"):
        return llm.generate(prompts, params)
    if hasattr(llm, "engine") and hasattr(llm.engine, "generate"):
        return llm.engine.generate(prompts, params)
    raise ValueError("LLM does not expose a compatible generate method")


def _continuation_logprob(output: Any, token_id: int) -> float | None:
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs:
        last = prompt_logprobs[-1]
        if last:
            entry = last.get(token_id)
            if entry is None:
                entry = last.get(int(token_id))
            if entry is not None:
                return _entry_logprob(entry)

    outputs = getattr(output, "outputs", None)
    if outputs:
        candidate_logprobs = getattr(outputs[0], "logprobs", None) or []
        if candidate_logprobs:
            entry = candidate_logprobs[0].get(token_id)
            if entry is None:
                entry = candidate_logprobs[0].get(int(token_id))
            if entry is not None:
                return _entry_logprob(entry)
    return None


def _entry_logprob(entry: Any) -> float | None:
    logprob = getattr(entry, "logprob", None)
    if logprob is not None:
        return float(logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (float, int)):
        return float(entry)
    return None
