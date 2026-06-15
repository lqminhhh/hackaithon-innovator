"""Tests for vLLM guided-choice label mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reasoning_agent import ReasoningAgent


class _DualVariantTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" ") and len(text) == 2:
            return [100 + ord(text.strip())]
        if len(text) == 1:
            return [ord(text)]
        return [999, 1000]


class _TokenMapAgent(ReasoningAgent):
    def __init__(self):
        super().__init__(model=object(), tokenizer=_DualVariantTokenizer())


class TestVllmLabelMap:
    def test_uses_one_token_id_per_label(self):
        agent = _TokenMapAgent()
        labels = tuple("ABCDEFGHIJ")

        token_map = agent._build_label_token_map(labels)

        assert len(token_map) == len(labels)
        assert sorted(token_map.values()) == list(labels)
