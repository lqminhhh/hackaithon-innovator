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
    """Score every legal label by its logprob as the next token after a prompt.

    This is the robust replacement for reading a single greedy decode's top-k
    logprobs. For each ``(prompt, token_map)`` we append each label's token to
    the prompt and read that exact token's ``prompt_logprobs`` value, so every
    legal label always receives a finite, directly comparable logprob — even
    for many-choice questions where vLLM's top-k logprob list would truncate
    most legal tokens and leave them at ``-inf`` (the bug that collapsed every
    margin to a degenerate constant).

    All ``(prompt, label)`` continuations are submitted in a single batched
    engine call; the shared prompt prefix is reused across a question's labels.

    Returns one ``{label -> logprob}`` dict per item. A label whose continuation
    logprob could not be recovered is left at ``-inf``.
    """
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
    raw_outputs = llm.raw_generate(requests, params)

    for output, (idx, label, token_id) in zip(raw_outputs, owners):
        logprob = _continuation_logprob(output, token_id)
        if logprob is not None:
            scores[idx][label] = logprob

    return scores


def _continuation_sampling_params(llm) -> Any:
    """Sampling params that expose the appended token's prompt logprob.

    ``max_tokens=1`` is the minimum vLLM allows; the generated token is ignored.
    ``prompt_logprobs`` is what surfaces the logprob of the appended label token.
    """
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


def _continuation_logprob(output: Any, token_id: int) -> float | None:
    """Read the logprob of the final (appended) prompt token from an output."""
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if not prompt_logprobs:
        return None
    last = prompt_logprobs[-1]
    if not last:
        return None
    entry = last.get(token_id)
    if entry is None:
        entry = last.get(int(token_id))
    if entry is None:
        return None
    return _entry_logprob(entry)


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
            self.llm, self.tokenizer, [(prompt, token_map)]
        )[0]
        if not any(math.isfinite(value) for value in per_letter_logprob.values()):
            raise ValueError(
                "guided-choice continuation scoring did not recover any legal "
                "label logprobs"
            )
        return ChoiceResult(
            letter=best_label(per_letter_logprob),
            margin=softmax_margin(per_letter_logprob),
            per_letter_logprob=per_letter_logprob,
        )


def _entry_logprob(entry: Any) -> float | None:
    logprob = getattr(entry, "logprob", None)
    if logprob is not None:
        return float(logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (float, int)):
        return float(entry)
    return None
