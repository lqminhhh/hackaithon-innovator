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

import re
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
            return self._vllm_score_valid_labels(prompt, valid_labels)

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

    def predict_route_choice(
        self,
        route: str,
        question: str,
        options: dict[str, str],
        context: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Select the best legal label using a route-specific direct-answer prompt."""
        valid_labels = tuple(sorted(options.keys()))
        prompt = self.build_route_prompt(route, question, options, context)
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

    def _vllm_score_valid_labels(
        self,
        prompt: str,
        valid_labels: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        """Approximate guided-choice scoring with constrained one-token decoding in vLLM.

        This path is designed for route-aware direct answering on GPU. It constrains
        the next token to the legal labels and returns whatever token-level logprobs
        vLLM exposes for those candidates.
        """
        from vllm import SamplingParams

        token_map = self._build_label_token_map(valid_labels)
        if not token_map:
            raise ValueError("Could not derive any legal single-token labels for guided choice.")

        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            top_p=1.0,
            logprobs=min(len(token_map), self._VLLM_MAX_LOGPROBS),
            allowed_token_ids=list(token_map.keys()),
        )
        outputs = self._llm.generate([prompt], params)
        output = outputs[0].outputs[0]

        scores = {label: float("-inf") for label in valid_labels}
        chosen_text = output.text.strip()
        if chosen_text in scores:
            scores[chosen_text] = 0.0

        candidate_logprobs = getattr(output, "logprobs", None) or []
        if candidate_logprobs:
            first_step = candidate_logprobs[0]
            for token_id, entry in first_step.items():
                label = token_map.get(int(token_id))
                if label is None:
                    continue
                logprob = getattr(entry, "logprob", None)
                if logprob is None and isinstance(entry, dict):
                    logprob = entry.get("logprob")
                if logprob is not None:
                    scores[label] = float(logprob)

        # If logprobs are unavailable, keep the chosen label as the only finite score.
        if all(value == float("-inf") for value in scores.values()):
            normalized = self._extract_valid_label(output.text, valid_labels)
            if normalized is None:
                raise ValueError(f"Could not normalize vLLM guided-choice output: {output.text!r}")
            scores[normalized] = 0.0

        return scores

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

    @staticmethod
    def _extract_valid_label(
        text: str,
        valid_labels: tuple[str, ...] | list[str],
    ) -> str | None:
        cleaned = text.strip().upper()
        if cleaned in valid_labels:
            return cleaned
        match = re.search(r"\b([A-Z])\b", cleaned)
        if match and match.group(1) in valid_labels:
            return match.group(1)
        return None

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
