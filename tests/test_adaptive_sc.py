"""Tests for adaptive self-consistency policy and wave escalation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import ParsedQuestion
from src.sc_policy import (
    SC_N_HIGH_CHOICE_KNOWLEDGE,
    SC_N_STEM,
    build_sc_reasoning_prompt,
    knowledge_escalation_reason,
    option_disambiguation_instruction,
    reading_escalation_reason,
    shuffle_options,
    should_use_think_mode,
    stem_sc_n,
)
from src.wave_solver import Wave1Result, _fit_extraction_prompt, run_wave2


def _parsed(
    *,
    qid: str = "q1",
    options: dict[str, str] | None = None,
    is_quantitative: bool = True,
) -> ParsedQuestion:
    options = options or {"A": "Một", "B": "Hai", "C": "Ba"}
    return ParsedQuestion(
        qid=qid,
        original_question="1 + 1 bằng bao nhiêu?",
        query="1 + 1 bằng bao nhiêu?",
        context=None,
        options=options,
        refusal_labels=(),
        n_choices=len(options),
        has_context=False,
        is_quantitative=is_quantitative,
        is_legal=False,
        has_refusal_choice=False,
        is_harmful=False,
    )


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        tokens = list(range(len(text.split())))
        if add_special_tokens:
            return [999] + tokens
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i in token_ids)


class _FakeAgent:
    is_vllm = False

    def __init__(self) -> None:
        self.generated = []
        self.scored_prompts = []
        self._tokenizer = _FakeTokenizer()
        self._model = type(
            "_FakeModel",
            (),
            {"config": type("_FakeConfig", (), {"max_position_embeddings": 64})()},
        )()

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate_freeform(self, prompts, **kwargs):
        self.generated.append({"prompts": prompts, "kwargs": kwargs})
        return [f"reasoning-{idx}" for idx, _prompt in enumerate(prompts)]

    def score_valid_labels(self, prompt, valid_labels):
        self.scored_prompts.append({"prompt": prompt, "valid_labels": valid_labels})
        return {label: (0.0 if label == "A" else -2.0) for label in valid_labels}


def test_stem_sc_n_uses_low_depth_only_for_low_margin_adaptive_mode():
    assert stem_sc_n(0.01, adaptive_sc=True) == SC_N_STEM["low"]
    assert stem_sc_n(0.90, adaptive_sc=True) == SC_N_STEM["high"]
    assert stem_sc_n(None, adaptive_sc=True) == SC_N_STEM["high"]
    assert stem_sc_n(0.01, adaptive_sc=False) == SC_N_STEM["high"]


def test_should_use_think_mode_for_stem_and_high_choice_knowledge_only():
    stem = _parsed()
    high_choice_knowledge = _parsed(
        options={chr(ord("A") + i): f"Lựa chọn {i}" for i in range(10)},
        is_quantitative=False,
    )
    low_choice_knowledge = _parsed(is_quantitative=False)

    assert should_use_think_mode(stem, "stem", stage="wave1") is True
    assert should_use_think_mode(high_choice_knowledge, "knowledge", stage="wave1") is True
    assert should_use_think_mode(low_choice_knowledge, "knowledge", stage="wave1") is False
    assert should_use_think_mode(low_choice_knowledge, "reading", stage="wave1") is False


def test_should_use_think_mode_for_reading_detail_lookup_questions():
    parsed = _parsed(
        qid="reading-detail",
        is_quantitative=False,
    )
    parsed = ParsedQuestion(
        qid=parsed.qid,
        original_question="Theo ngữ cảnh, lần đầu tiên sự kiện này xảy ra vào năm nào?",
        query="Theo ngữ cảnh, lần đầu tiên sự kiện này xảy ra vào năm nào?",
        context="Đoạn thông tin dài",
        options=parsed.options,
        refusal_labels=(),
        n_choices=len(parsed.options),
        has_context=True,
        is_quantitative=False,
        is_legal=False,
        has_refusal_choice=False,
        is_harmful=False,
    )

    assert reading_escalation_reason(parsed.query) == "reading_detail_lookup_sc"
    assert should_use_think_mode(parsed, "reading", stage="wave1") is False
    assert should_use_think_mode(parsed, "reading", stage="wave2") is True


def test_knowledge_escalation_reason_uses_ambiguous_option_signal():
    parsed = _parsed(
        is_quantitative=False,
        options={
            "A": "Độ thỏa dụng biên trên mỗi đồng là lớn nhất",
            "B": "Tổng độ thỏa dụng trên mỗi đồng là lớn nhất",
            "C": "Độ thỏa dụng trung bình trên mỗi đồng là lớn nhất",
            "D": "Tổng độ thỏa dụng trên mỗi đồng là nhỏ nhất",
        },
    )

    assert knowledge_escalation_reason(parsed, margin=1.0) == "knowledge_ambiguous_options_sc"


def test_shuffle_options_preserves_duplicate_reverse_mapping():
    options = {
        "A": "8 năm",
        "B": "8 năm",
        "C": "10 năm",
    }

    seen_reverse_maps = set()
    for sample_idx in range(6):
        shuffled, reverse_map = shuffle_options(options, sample_idx)
        assert shuffled.keys() == options.keys()
        assert set(reverse_map) == set(options)
        for shuffled_label, original_label in reverse_map.items():
            assert shuffled[shuffled_label] == options[original_label]
        seen_reverse_maps.add(tuple(sorted(reverse_map.items())))

    assert len(seen_reverse_maps) > 1


def test_option_disambiguation_instruction_mentions_duplicate_and_combination_handling():
    options = {
        "A": "Paris",
        "B": "Paris",
        "C": "Cả A, B, C",
        "D": "London",
    }

    instruction = option_disambiguation_instruction(options)

    assert "trùng hệt nội dung" in instruction
    assert "tất cả/cả A, B, C" in instruction


def test_sc_prompt_adds_disambiguation_guidance_for_tricky_knowledge_options():
    parsed = _parsed(
        is_quantitative=False,
        options={
            "A": "Độ co giãn cầu theo giá lớn hơn 1",
            "B": "Độ co giãn cầu theo giá lớn hơn 1",
            "C": "Cả A, B, C",
            "D": "Độ co giãn cầu theo giá nhỏ hơn 1",
        },
    )

    prompt = build_sc_reasoning_prompt(parsed, "knowledge", parsed.options)

    assert "trùng hệt nội dung" in prompt
    assert "chỉ chọn phương án gộp khi mọi thành phần đều đúng" in prompt


def test_fit_extraction_prompt_compacts_long_reading_prompts():
    agent = _FakeAgent()
    prompt = _fit_extraction_prompt(
        agent,
        route="reading",
        question="Theo ngữ cảnh, điều gì đã xảy ra?",
        options={"A": "X", "B": "Y"},
        reasoning_prompt="Đoạn thông tin:\n---\n" + ("chi tiết " * 80) + "\n---\n\nCâu hỏi:\nTheo ngữ cảnh...",
        reasoning="kết luận " * 80,
    )

    assert "Bạn đang chốt đáp án trắc nghiệm tiếng Việt." in prompt
    assert "Lời giải nháp rút gọn:" in prompt
    assert "Đoạn thông tin:\n---\n" not in prompt


def test_wave2_uses_seven_stem_samples_for_low_margin_when_adaptive():
    agent = _FakeAgent()
    parsed = _parsed()
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="stem",
            answer="B",
            margin=0.01,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].answer in parsed.options
    assert len(wave2[parsed.qid].votes) > 0
    assert wave2[parsed.qid].escalation_reason.startswith("stem_sc_adaptive")
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_STEM["low"]
    assert len(agent.scored_prompts) == SC_N_STEM["low"]
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_wave2_uses_three_stem_samples_when_adaptive_disabled():
    agent = _FakeAgent()
    parsed = _parsed()
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="stem",
            answer="B",
            margin=0.01,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=False)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].answer in parsed.options
    assert len(wave2[parsed.qid].votes) > 0
    assert wave2[parsed.qid].escalation_reason.startswith("stem_sc_fixed")
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_STEM["high"]
    assert len(agent.scored_prompts) == SC_N_STEM["high"]


def test_wave2_escalates_high_choice_knowledge_even_with_high_margin():
    agent = _FakeAgent()
    options = {chr(ord("A") + i): f"Lựa chọn {i}" for i in range(10)}
    parsed = _parsed(options=options, is_quantitative=False)
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="knowledge",
            answer="J",
            margin=1.0,
            per_letter_logprob={label: (-10.0 if label != "J" else 0.0) for label in options},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].answer in parsed.options
    assert wave2[parsed.qid].escalation_reason == "knowledge_high_choice_n10"
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_HIGH_CHOICE_KNOWLEDGE
    assert len(agent.scored_prompts) == SC_N_HIGH_CHOICE_KNOWLEDGE
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_wave2_escalates_ambiguous_knowledge_even_with_high_margin():
    agent = _FakeAgent()
    parsed = _parsed(
        qid="ambiguous-knowledge",
        is_quantitative=False,
        options={
            "A": "Độ thỏa dụng biên trên mỗi đồng là lớn nhất",
            "B": "Tổng độ thỏa dụng trên mỗi đồng là lớn nhất",
            "C": "Độ thỏa dụng trung bình trên mỗi đồng là lớn nhất",
            "D": "Tổng độ thỏa dụng trên mỗi đồng là nhỏ nhất",
        },
    )
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="knowledge",
            answer="B",
            margin=1.0,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0, "D": -4.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].escalation_reason == "knowledge_ambiguous_options_sc"
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == SC_N_HIGH_CHOICE_KNOWLEDGE
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_wave2_escalates_reading_detail_lookup_questions():
    agent = _FakeAgent()
    parsed = ParsedQuestion(
        qid="reading-detail",
        original_question="Theo ngữ cảnh, lần đầu tiên sự kiện này xảy ra vào năm nào?",
        query="Theo ngữ cảnh, lần đầu tiên sự kiện này xảy ra vào năm nào?",
        context="Đoạn thông tin dài",
        options={"A": "1990", "B": "1991", "C": "1992"},
        refusal_labels=(),
        n_choices=3,
        has_context=True,
        is_quantitative=False,
        is_legal=False,
        has_refusal_choice=False,
        is_harmful=False,
    )
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="reading",
            answer="B",
            margin=1.0,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert set(wave2) == {parsed.qid}
    assert wave2[parsed.qid].escalation_reason == "reading_detail_lookup_sc"
    assert len(agent.generated) == 1
    assert len(agent.generated[0]["prompts"]) == 3
    assert agent.generated[0]["kwargs"]["mode"] == "think"


def test_wave2_does_not_escalate_high_margin_low_choice_knowledge():
    agent = _FakeAgent()
    parsed = _parsed(is_quantitative=False)
    wave1 = {
        parsed.qid: Wave1Result(
            qid=parsed.qid,
            route="knowledge",
            answer="B",
            margin=1.0,
            per_letter_logprob={"A": -2.0, "B": 0.0, "C": -3.0},
        )
    }

    wave2 = run_wave2(agent, [parsed], wave1, adaptive_sc=True)

    assert wave2 == {}
    assert agent.generated == []
    assert agent.scored_prompts == []
