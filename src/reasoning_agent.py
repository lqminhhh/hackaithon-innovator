"""Reasoning agent — wraps an LLM to perform CoT inference.

Supports two backends:
  - vLLM  (fast, batched, CUDA only) — pass ``llm`` arg
  - HuggingFace Transformers (fallback) — pass ``model`` + ``tokenizer``
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.extract import ChoiceResult, GuidedChoiceExtractor, best_label, safe_margin

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

    _VLLM_MAX_LOGPROBS = 20

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

    @property
    def tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._llm is not None:
            return self._llm.get_tokenizer()
        raise ValueError("Tokenizer is not available.")

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
    ) -> str:
        """Build the user-facing prompt text (without chat template wrapping)."""
        options_block, valid_labels = self._format_options(options)
        return self.prompts["cot_no_context"].format(
            question=question,
            options_block=options_block,
            valid_labels=valid_labels,
        )

    def build_guided_choice_prompt(
        self,
        question: str,
        options: dict[str, str],
    ) -> str:
        """Build a short prompt for constrained answer selection."""
        options_block, valid_labels = self._format_options(options)
        return self.prompts["guided_choice_no_context"].format(
            question=question,
            options_block=options_block,
            valid_labels=valid_labels,
            label_list=", ".join(sorted(options.keys())),
        )

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

    def generate_freeform(
        self,
        prompts: list[str],
        *,
        mode: str = "no_think",
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        """Generate unconstrained text for reasoning/escalation passes."""
        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        max_new = max_tokens if max_tokens is not None else self.cfg["max_new_tokens"]

        if self.is_vllm and hasattr(self._llm, "generate_text"):
            outputs = self._llm.generate_text(
                prompts,
                mode=mode,  # type: ignore[arg-type]
                max_tokens=max_new,
                temperature=temp,
                top_p=top_p,
            )
            return [output.text for output in outputs]

        tagged = [self._tag_mode(prompt, mode) for prompt in prompts]
        if self.is_vllm:
            return self._vllm_batch(tagged, temp, max_tokens=max_new, top_p=top_p)
        return [
            self._hf_generate(prompt, temp, max_tokens=max_new, top_p=top_p)
            for prompt in tagged
        ]

    def _vllm_batch(
        self,
        prompts: list[str],
        temperature: float | None,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        from vllm import SamplingParams

        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        params = SamplingParams(
            temperature=temp,
            max_tokens=max_tokens if max_tokens is not None else self.cfg["max_new_tokens"],
            top_p=top_p if top_p is not None else (0.9 if temp > 0 else 1.0),
        )
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self._llm.chat(conversations, params)
        return [o.outputs[0].text for o in outputs]

    def _hf_generate(
        self,
        prompt: str,
        temperature: float | None,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> str:
        import torch

        temp = temperature if temperature is not None else self.cfg["temperature_deterministic"]
        max_new = max_tokens if max_tokens is not None else self.cfg["max_new_tokens"]

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
            gen_kwargs["top_p"] = top_p if top_p is not None else 0.9

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _tag_mode(prompt: str, mode: str) -> str:
        tag = "/think" if mode == "think" else "/no_think"
        stripped = prompt.lstrip()
        if stripped.startswith("/think") or stripped.startswith("/no_think"):
            return prompt
        return f"{tag}\n{prompt}"

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
            result = GuidedChoiceExtractor(self._llm, self.tokenizer).extract(
                prompt, valid_labels
            )
            return result.per_letter_logprob

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
        result = self.predict_guided_choice_result(question, options, context)
        return result.letter, result.per_letter_logprob

    def predict_guided_choice_result(
        self,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> ChoiceResult:
        """Select the best legal label and return logprob margin evidence."""
        valid_labels = tuple(sorted(options.keys()))
        prompt = self.build_guided_choice_prompt(question, options)
        scores = self.score_valid_labels(prompt, valid_labels)
        return ChoiceResult(
            letter=best_label(scores),
            margin=safe_margin(scores, len(valid_labels)),
            per_letter_logprob=scores,
        )

    def predict_route_choice(
        self,
        route: str,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Select the best legal label using a route-specific direct-answer prompt."""
        result = self.predict_route_choice_result(route, question, options, context)
        return result.letter, result.per_letter_logprob

    def predict_route_choice_result(
        self,
        route: str,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> ChoiceResult:
        """Route-specific constrained choice with margin evidence."""
        valid_labels = tuple(sorted(options.keys()))
        prompt = self.build_route_prompt(route, question, options, context)
        scores = self.score_valid_labels(prompt, valid_labels)
        return ChoiceResult(
            letter=best_label(scores),
            margin=safe_margin(scores, len(valid_labels)),
            per_letter_logprob=scores,
        )

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

    def _build_label_token_map(
        self,
        valid_labels: tuple[str, ...] | list[str],
    ) -> dict[int, str]:
        """Map one vLLM token id per legal label for constrained decoding.

        We intentionally keep a single token id for each label. Some tokenizers
        expose both ``"A"`` and ``" A"`` as separate one-token variants, but
        passing every variant through to vLLM can exceed its logprob cap on
        questions with many choices (for example A-J).
        """
        token_map: dict[int, str] = {}
        tokenizer = self.tokenizer

        for label in valid_labels:
            # Prefer the whitespace-prefixed form because the label is generated
            # after prompt text, but fall back to the bare label when needed.
            for variant in (f" {label}", label):
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_map[int(token_ids[0])] = label
                    break
        return token_map

    # ── backward-compatible single-question methods ───────────────────

    def infer_no_context(
        self,
        question: str,
        options: dict[str, str],
        temperature: float | None = None,
    ) -> str:
        """Run CoT for a single question."""
        prompt = self.build_prompt(question, options)
        return self.generate_batch([prompt], temperature)[0]
