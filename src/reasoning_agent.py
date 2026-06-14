"""Reasoning agent — wraps an LLM to perform CoT inference.

Supports two backends:
  - vLLM  (fast, batched, CUDA only) — pass ``llm`` arg
  - HuggingFace Transformers (fallback) — pass ``model`` + ``tokenizer``

And two inference modes per backend:
  1. No-context (pure CoT) — uses the question + options only.
  2. With-context — injects retrieved chunks before the question.

Returns the raw output text.  Confidence extraction and answer parsing
are handled downstream by the normaliser and confidence gate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"


def _load_prompts() -> dict[str, str]:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


class ReasoningAgent:
    """Unified reasoning agent supporting vLLM and HuggingFace backends."""

    def __init__(self, llm=None, model=None, tokenizer=None):
        self.prompts = _load_prompts()
        self.cfg = _load_config()["inference"]

        self._llm = llm
        self._model = model
        self._tokenizer = tokenizer

        if llm is None and model is None:
            raise ValueError("Provide either a vLLM LLM instance or a HF model + tokenizer")

    @property
    def is_vllm(self) -> bool:
        return self._llm is not None

    # ── prompt construction ───────────────────────────────────────────

    @staticmethod
    def _format_options(options: dict[str, str]) -> tuple[str, str]:
        labels = sorted(options.keys())
        block = "\n".join(f"{l}) {options[l]}" for l in labels)
        hint = "/".join(labels)
        return block, hint

    def build_prompt(
        self,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> str:
        """Build the user-facing prompt text (without chat template wrapping)."""
        options_block, valid_labels = self._format_options(options)
        kwargs = dict(
            question=question,
            options_block=options_block,
            valid_labels=valid_labels,
        )
        if context is not None:
            return self.prompts["cot_with_context"].format(
                retrieved_context=context, **kwargs
            )
        return self.prompts["cot_no_context"].format(**kwargs)

    def build_guided_choice_prompt(
        self,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> str:
        """Build a short prompt for constrained answer selection.

        This path is intentionally concise: the model should decide among the
        existing choices and output only one legal option label.
        """
        options_block, valid_labels = self._format_options(options)
        kwargs = dict(
            question=question,
            options_block=options_block,
            valid_labels=valid_labels,
            label_list=", ".join(sorted(options.keys())),
        )

        if context is not None:
            return self.prompts["guided_choice_with_context"].format(
                retrieved_context=context, **kwargs
            )
        return self.prompts["guided_choice_no_context"].format(**kwargs)

    def build_route_prompt(
        self,
        route: str,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> str:
        """Build a route-specific direct-answer prompt."""
        route_to_template = {
            "reading": "reading_direct",
            "stem": "stem_direct",
            "safety": "safety_direct",
            "knowledge": "knowledge_direct",
        }
        template_name = route_to_template[route]

        options_block, valid_labels = self._format_options(options)
        kwargs = dict(
            question=question,
            options_block=options_block,
            valid_labels=valid_labels,
            label_list=", ".join(sorted(options.keys())),
        )

        if route == "reading":
            return self.prompts[template_name].format(
                retrieved_context=context or "", **kwargs
            )
        return self.prompts[template_name].format(**kwargs)

    # ── generation (batch + single) ───────────────────────────────────

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts.

        With vLLM this is truly batched (continuous batching).
        With HF this falls back to sequential generation.
        """
        if self.is_vllm:
            return self._vllm_batch(prompts, temperature)
        return [self._hf_generate(p, temperature) for p in prompts]

    def _vllm_batch(self, prompts: list[str], temperature: float | None) -> list[str]:
        from vllm import SamplingParams

        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        params = SamplingParams(
            temperature=temp,
            max_tokens=self.cfg["max_new_tokens"],
            top_p=0.9 if temp > 0 else 1.0,
        )
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self._llm.chat(conversations, params)
        return [o.outputs[0].text for o in outputs]

    def _hf_generate(self, prompt: str, temperature: float | None) -> str:
        import torch

        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        max_new = self.cfg["max_new_tokens"]

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new,
            do_sample=temp > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── guided-choice decoding ───────────────────────────────────────

    def score_valid_labels(
        self,
        prompt: str,
        valid_labels: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        """Score only the legal answer labels for one prompt.

        Returns a dict of label -> logprob score. Higher is better.
        Currently implemented for HuggingFace backends; vLLM support can be
        added later with token-level logprob extraction.
        """
        if self.is_vllm:
            raise NotImplementedError(
                "Guided-choice scoring is not implemented for vLLM yet."
            )

        return {
            label: self._hf_score_completion(prompt, f" {label}")
            for label in valid_labels
        }

    def predict_guided_choice(
        self,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Select the best legal label using constrained completion scoring."""
        valid_labels = tuple(sorted(options.keys()))
        prompt = self.build_guided_choice_prompt(question, options, context)
        scores = self.score_valid_labels(prompt, valid_labels)
        best = max(scores, key=scores.get)
        return best, scores

    def _hf_score_completion(self, prompt: str, completion: str) -> float:
        import torch

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + completion

        prompt_inputs = self._tokenizer(prompt_text, return_tensors="pt")
        full_inputs = self._tokenizer(full_text, return_tensors="pt").to(self._model.device)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        target_ids = full_inputs["input_ids"][:, prompt_len:]

        with torch.no_grad():
            outputs = self._model(**full_inputs)

        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_scores = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_scores.sum().item())

    # ── backward-compatible single-question methods ───────────────────

    def infer_no_context(
        self,
        question: str,
        options: dict[str, str],
        temperature: float | None = None,
    ) -> str:
        """Run CoT with no retrieved context (single question)."""
        prompt = self.build_prompt(question, options)
        return self.generate_batch([prompt], temperature)[0]

    def infer_with_context(
        self,
        question: str,
        options: dict[str, str],
        context: str,
        temperature: float | None = None,
    ) -> str:
        """Run CoT with retrieved context injected (single question)."""
        prompt = self.build_prompt(question, options, context)
        return self.generate_batch([prompt], temperature)[0]
