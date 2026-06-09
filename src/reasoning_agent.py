"""Reasoning agent — wraps an LLM to perform CoT inference.

Supports two modes:
  1. No-context (pure CoT) — uses the question + options only.
  2. With-context — injects retrieved chunks before the question.

Returns the raw output text.  Confidence extraction and answer parsing
are handled downstream by the normaliser and confidence gate.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizerBase

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"


def _load_prompts() -> dict[str, str]:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


class ReasoningAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = _load_prompts()
        self.cfg = _load_config()["inference"]

    def infer_no_context(
        self,
        question: str,
        options: dict[str, str],
        temperature: float | None = None,
    ) -> str:
        """Run CoT with no retrieved context."""
        prompt = self.prompts["cot_no_context"].format(
            question=question,
            A=options["A"],
            B=options["B"],
            C=options["C"],
            D=options["D"],
        )
        return self._generate(prompt, temperature)

    def infer_with_context(
        self,
        question: str,
        options: dict[str, str],
        context: str,
        temperature: float | None = None,
    ) -> str:
        """Run CoT with retrieved context injected."""
        prompt = self.prompts["cot_with_context"].format(
            question=question,
            A=options["A"],
            B=options["B"],
            C=options["C"],
            D=options["D"],
            retrieved_context=context,
        )
        return self._generate(prompt, temperature)

    def _generate(self, prompt: str, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        max_new = self.cfg["max_new_tokens"]

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
