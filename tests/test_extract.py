"""Tests for S2 guided-choice extraction."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import (
    GuidedChoiceExtractor,
    best_label,
    build_choice_prompt,
    build_label_token_map,
    softmax_margin,
)
from src.reasoning_agent import ReasoningAgent


class _FakeParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text.isalpha():
            return [ord(text)]
        if text.startswith(" ") and len(text.strip()) == 1:
            return [1000 + ord(text.strip())]
        return [999, 1000]


class _FakeLLM:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls = []
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def sampling_params(self, **kwargs):
        return _FakeParams(**kwargs)

    def raw_generate(self, requests, params):
        self.calls.append({"requests": requests, "params": params})
        outputs = []
        for request in requests:
            token_id = request["prompt_token_ids"][-1]
            label = chr(token_id)
            score = self.scores.get(label, -99.0)
            outputs.append(
                SimpleNamespace(
                    prompt_logprobs=[
                        {token_id: SimpleNamespace(logprob=score)}
                    ]
                )
            )
        return outputs


def test_build_choice_prompt_ends_at_answer_slot():
    prompt = build_choice_prompt(
        "Thủ đô của Việt Nam là gì?",
        {"A": "Hà Nội", "B": "Huế"},
    )

    assert "A) Hà Nội" in prompt
    assert "B) Huế" in prompt
    assert prompt.endswith("Đáp án: ")


def test_build_label_token_map_prefers_bare_label_tokens():
    token_map = build_label_token_map(_FakeTokenizer(), list("ABC"))

    assert token_map == {
        ord("A"): "A",
        ord("B"): "B",
        ord("C"): "C",
    }


def test_softmax_margin_matches_expected_probability_gap():
    logprobs = {"A": -3.0, "B": -0.2, "C": -1.5}

    margin = softmax_margin(logprobs)

    probs = [math.exp(v - max(logprobs.values())) for v in logprobs.values()]
    total = sum(probs)
    expected = sorted((p / total for p in probs), reverse=True)
    assert margin == pytest.approx(expected[0] - expected[1])
    assert best_label(logprobs) == "B"


def test_guided_choice_extractor_returns_letter_margin_and_logprobs():
    llm = _FakeLLM({"A": -3.0, "B": -0.2, "C": -1.5})
    extractor = GuidedChoiceExtractor(llm)

    result = extractor.predict(
        "1 + 1 bằng bao nhiêu?",
        {"A": "1", "B": "2", "C": "3"},
    )

    assert result.letter == "B"
    assert 0.0 <= result.margin <= 1.0
    assert result.per_letter_logprob == {"A": -3.0, "B": -0.2, "C": -1.5}
    call = llm.calls[0]
    assert len(call["requests"]) == 3
    assert [request["prompt_token_ids"][-1] for request in call["requests"]] == [
        ord("A"),
        ord("B"),
        ord("C"),
    ]
    assert call["params"].kwargs["max_tokens"] == 1
    assert call["params"].kwargs["prompt_logprobs"] == 1


def test_guided_choice_extractor_supports_eleven_choices():
    labels = list("ABCDEFGHIJK")
    scores = {label: -float(i) for i, label in enumerate(labels)}
    llm = _FakeLLM(scores)

    result = GuidedChoiceExtractor(llm).extract("Đáp án: ", labels)

    assert result.letter == "A"
    assert set(result.per_letter_logprob) == set(labels)
    assert len(llm.calls[0]["requests"]) == 11


def test_guided_choice_extractor_requires_logprobs_not_output_text_parsing():
    llm = _FakeLLM({"A": -0.1, "B": -2.0})

    def _generate_without_logprobs(requests, params):
        return [SimpleNamespace(prompt_logprobs=None) for _request in requests]

    llm.raw_generate = _generate_without_logprobs

    with pytest.raises(ValueError, match="did not recover"):
        GuidedChoiceExtractor(llm).extract("Đáp án: ", ["A", "B"])


class _ScoredAgent(ReasoningAgent):
    def __init__(self):
        super().__init__(model=object(), tokenizer=_FakeTokenizer())

    def score_valid_labels(self, prompt, valid_labels):
        return {
            label: {"A": -3.0, "B": -0.2, "C": -1.5}.get(label, -9.0)
            for label in valid_labels
        }


def test_reasoning_agent_exposes_choice_result_with_margin():
    agent = _ScoredAgent()

    result = agent.predict_guided_choice_result(
        question="1 + 1 bằng bao nhiêu?",
        options={"A": "1", "B": "2", "C": "3"},
    )

    assert result.letter == "B"
    assert result.margin > 0
    assert result.per_letter_logprob["B"] > result.per_letter_logprob["C"]
