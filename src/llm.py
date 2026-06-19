"""S1 vLLM wrapper for the v2 pipeline.

The rest of the project should not need to know the exact vLLM constructor
flags or the Qwen thinking-mode mechanics. This wrapper owns those details and
keeps a small compatibility surface for the older ``ReasoningAgent`` code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.config import GPU_MEM_UTIL, LLM_MODEL

ThinkingMode = Literal["think", "no_think"]


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Text plus optional token logprobs returned by vLLM."""

    text: str
    logprobs: Any | None = None


class LLM:
    """Thin wrapper around ``vllm.LLM`` with Qwen thinking-mode support."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        *,
        gpu_memory_utilization: float = GPU_MEM_UTIL,
        max_model_len: int = 8192,
        max_num_seqs: int | None = None,
        quantization: str | None = None,
        enable_prefix_caching: bool = True,
        dtype: str = "half",
        trust_remote_code: bool = True,
        engine: Any | None = None,
        engine_cls: Any | None = None,
        sampling_params_cls: Any | None = None,
    ):
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.quantization = quantization
        self.enable_prefix_caching = enable_prefix_caching
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._sampling_params_cls = sampling_params_cls

        self.init_kwargs = {
            "model": model,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "trust_remote_code": trust_remote_code,
        }
        if max_num_seqs is not None:
            self.init_kwargs["max_num_seqs"] = max_num_seqs
        if quantization is not None:
            self.init_kwargs["quantization"] = quantization

        if engine is not None:
            self.engine = engine
        else:
            if engine_cls is None:
                from vllm import LLM as VllmLLM

                engine_cls = VllmLLM
            self.engine = engine_cls(**self.init_kwargs)

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def generate_text(
        self,
        prompts: list[str],
        *,
        mode: ThinkingMode = "no_think",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float | None = None,
        logprobs: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate a batched list of completions.

        ``mode`` is passed through Qwen's chat template when supported. If the
        installed tokenizer/vLLM version does not accept ``enable_thinking``,
        the wrapper falls back to explicit ``/think`` or ``/no_think`` tags.
        """
        params = self.sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        chat_kwargs = self._chat_template_kwargs(mode)

        try:
            raw_outputs = self.engine.chat(
                conversations,
                params,
                chat_template_kwargs=chat_kwargs,
            )
        except TypeError:
            tagged_conversations = [
                [{"role": "user", "content": self._mode_prompt(prompt, mode)}]
                for prompt in prompts
            ]
            raw_outputs = self.engine.chat(tagged_conversations, params)

        return [
            GenerationOutput(
                text=o.outputs[0].text,
                logprobs=getattr(o.outputs[0], "logprobs", None),
            )
            for o in raw_outputs
        ]

    def sampling_params(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        logprobs: int | None = None,
        **extra: Any,
    ):
        """Build vLLM ``SamplingParams`` with optional logprob exposure."""
        SamplingParams = self._sampling_params_cls
        if SamplingParams is None:
            from vllm import SamplingParams

        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p if top_p is not None else (0.9 if temperature > 0 else 1.0),
        }
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        kwargs.update(extra)
        return SamplingParams(**kwargs)

    @staticmethod
    def _chat_template_kwargs(mode: ThinkingMode) -> dict[str, bool]:
        return {"enable_thinking": mode == "think"}

    @staticmethod
    def _mode_prompt(prompt: str, mode: ThinkingMode) -> str:
        tag = "/think" if mode == "think" else "/no_think"
        stripped = prompt.lstrip()
        if stripped.startswith("/think") or stripped.startswith("/no_think"):
            return prompt
        return f"{tag}\n{prompt}"

    # Compatibility methods for existing code that still treats this as vllm.LLM.
    def chat(self, *args: Any, **kwargs: Any):
        return self.engine.chat(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        return self.engine.generate(*args, **kwargs)
