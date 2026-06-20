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
    """Margin that treats a degenerate extraction as low confidence.

    A healthy guided-choice extraction over ``expected_labels`` legal options
    returns a finite logprob for each. If the engine returns fewer than two
    finite logprobs for a multi-option question, the result is malformed —
    ``softmax_margin`` would early-exit at ``1.0`` (maximum confidence) and
    silently suppress every downstream escalation. We invert that to ``0.0``
    (minimum confidence) so the item escalates instead of being skipped.
    """
    finite = sum(1 for value in logprobs.values() if math.isfinite(value))
    if expected_labels >= 2 and finite < 2:
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

        params = self._sampling_params(token_map)
        raw = self.llm.raw_generate([prompt], params)
        output = raw[0].outputs[0]
        per_letter_logprob = self._scores_from_output(output, labels, token_map)
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
            "logprobs": min(len(token_map), self.max_logprobs),
            "allowed_token_ids": list(token_map.keys()),
        }
        if hasattr(self.llm, "sampling_params"):
            return self.llm.sampling_params(**kwargs)

        from vllm import SamplingParams

        return SamplingParams(**kwargs)

    def _scores_from_output(
        self,
        output,
        valid_labels: tuple[str, ...],
        token_map: dict[int, str],
    ) -> dict[str, float]:
        scores = {label: float("-inf") for label in valid_labels}
        candidate_logprobs = getattr(output, "logprobs", None) or []
        if not candidate_logprobs:
            raise ValueError("guided-choice output did not include token logprobs")

        for token_id, entry in candidate_logprobs[0].items():
            label = token_map.get(int(token_id))
            if label is None:
                continue
            logprob = _entry_logprob(entry)
            if logprob is not None:
                scores[label] = float(logprob)

        if not any(math.isfinite(value) for value in scores.values()):
            raise ValueError("guided-choice logprobs did not contain legal labels")
        return scores


def _entry_logprob(entry: Any) -> float | None:
    logprob = getattr(entry, "logprob", None)
    if logprob is not None:
        return float(logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (float, int)):
        return float(entry)
    return None
