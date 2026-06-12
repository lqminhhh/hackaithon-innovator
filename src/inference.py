"""Inference backend abstraction for the Entropy-Gated Jury.

Public API
----------
InferenceResult      dataclass: letter, logprob_dist, raw_text
InferenceBackend     abstract base class
LlamaCppBackend      llama-cpp-python + Metal (local MacBook dev)
VllmBackend          vLLM main-branch, pinned commit (Docker/CUDA prod)

Usage
-----
backend = build_backend(cfg)   # reads configs/pipeline_config.yaml
results = backend.generate_batch(requests)

All gate / jury logic lives above this layer and is backend-agnostic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"

ALPHABET = "ABCDEFGHIJK"


@dataclass
class InferenceRequest:
    prompt: str                          # raw string for LlamaCppBackend
    allowed_letters: list[str]           # e.g. ["A","B","C","D"]
    thinking_budget: int = 600           # max thinking tokens
    temperature: float = 0.0             # 0 = greedy
    top_p: float = 1.0
    n_samples: int = 1                   # >1 → multiple independent samples (jury)
    messages: list[dict] | None = None   # structured messages for VllmBackend.chat()


@dataclass
class InferenceResult:
    letter: str                                  # top-1 answer letter
    logprob_dist: dict[str, float] = field(default_factory=dict)  # letter → logprob
    raw_text: str = ""

    @property
    def top1_logprob(self) -> float:
        return self.logprob_dist.get(self.letter, 0.0)

    @property
    def margin(self) -> float:
        """top1 − top2 logprob margin (used by gate.py)."""
        vals = sorted(self.logprob_dist.values(), reverse=True)
        if len(vals) < 2:
            return 10.0
        return vals[0] - vals[1]


class InferenceBackend(ABC):
    """Abstract base — backends implement generate_batch only."""

    @abstractmethod
    def generate_batch(
        self, requests: list[InferenceRequest]
    ) -> list[list[InferenceResult]]:
        """Process a batch of requests.

        Returns a list of length == len(requests).
        Each element is a list of InferenceResult of length request.n_samples.
        For n_samples=1 (default) the inner list has exactly one item.
        """

    def warmup(self, dummy_letters: list[str] = None) -> None:
        """Pre-warm CUDA graphs / compilation. Call before the timed run."""
        if dummy_letters is None:
            dummy_letters = ["A", "B", "C", "D"]
        req = InferenceRequest(
            prompt="Warmup.",
            allowed_letters=dummy_letters,
            thinking_budget=1,
            temperature=0.0,
        )
        self.generate_batch([req])


# ── llama-cpp-python backend (local, Metal) ───────────────────────────


class LlamaCppBackend(InferenceBackend):
    """llama-cpp-python with Metal acceleration for local MacBook development.

    Supports logit_bias for answer-token masking and logprobs for the
    confidence margin gate — the full EGJ architecture is testable locally,
    just slower than vLLM on CUDA.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,  # -1 = all layers to Metal
        n_ctx: int = 8192,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/metal"
            ) from e

        self._model_path = model_path
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,   # required to retrieve per-token logprobs for the margin gate
            verbose=verbose,
        )
        self._tok = self._llm
        print(f"LlamaCppBackend loaded: {Path(model_path).name}", flush=True)

    def _letter_token_ids(self, letters: list[str]) -> list[int]:
        """Return the first token ID for each answer letter."""
        ids: list[int] = []
        for letter in letters:
            toks = self._llm.tokenize(letter.encode(), add_bos=False, special=False)
            if toks:
                ids.append(toks[0])
        return ids

    def _build_logit_bias(
        self, allowed_letters: list[str], vocab_size: int
    ) -> dict[int, float]:
        """Build logit_bias dict: allowed letters → +0, all else → -inf."""
        allowed_ids = set(self._letter_token_ids(allowed_letters))
        # llama-cpp logit_bias keys are token IDs (int), values are bias float
        bias: dict[int, float] = {}
        for tid in range(vocab_size):
            if tid not in allowed_ids:
                bias[tid] = -1e9
        return bias

    def generate_batch(
        self, requests: list[InferenceRequest]
    ) -> list[list[InferenceResult]]:
        results: list[list[InferenceResult]] = []
        for req in requests:
            samples: list[InferenceResult] = []
            for _ in range(req.n_samples):
                sample = self._generate_one(req)
                samples.append(sample)
            results.append(samples)
        return results

    def _generate_one(self, req: InferenceRequest) -> InferenceResult:
        vocab_size = self._llm.n_vocab()
        logit_bias = self._build_logit_bias(req.allowed_letters, vocab_size)

        gen_kwargs: dict[str, Any] = dict(
            prompt=req.prompt,
            max_tokens=1,
            temperature=max(req.temperature, 1e-4),
            top_p=req.top_p,
            logit_bias=logit_bias,
            logprobs=len(req.allowed_letters),
        )

        output = self._llm(**gen_kwargs)
        choice = output["choices"][0]
        token_text = choice.get("text", "").strip().upper()

        logprob_dist: dict[str, float] = {}
        top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [])
        if top_logprobs:
            for tok, lp in top_logprobs[0].items():
                letter = tok.strip().upper()
                if letter in req.allowed_letters:
                    logprob_dist[letter] = float(lp)

        # Normalise to letter keys; fill missing with large negative
        for letter in req.allowed_letters:
            if letter not in logprob_dist:
                logprob_dist[letter] = -100.0

        letter = token_text if token_text in req.allowed_letters else req.allowed_letters[0]

        return InferenceResult(
            letter=letter,
            logprob_dist=logprob_dist,
            raw_text=choice.get("text", ""),
        )


# ── vLLM backend (CUDA, production) ───────────────────────────────────


class VllmBackend(InferenceBackend):
    """vLLM backend for Linux/CUDA production environment.

    Uses the llm.chat() API so the tokenizer's built-in chat template is
    applied — no hand-rolled prompt strings needed.  For Qwen3.5 the
    template is invoked with enable_thinking=False so that the very first
    generated token is forced to be an answer letter (A-K).

    n_samples > 1 is handled via SamplingParams(n=...) — one vLLM request,
    N outputs — instead of duplicating the prompt N times.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 16384,
        enable_prefix_caching: bool = True,
        tensor_parallel_size: int = 1,
    ):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError("vLLM not installed: pip install vllm") from e

        from vllm import LLM

        self._model_id = model_id
        self._model_family = get_model_family_by_id(model_id)
        self._llm = LLM(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._tokenizer = self._llm.get_tokenizer()
        print(f"[VllmBackend] loaded {model_id} (family={self._model_family})", flush=True)

    def _allowed_token_ids(self, letters: list[str]) -> list[int]:
        ids: list[int] = []
        for letter in letters:
            toks = self._tokenizer.encode(letter, add_special_tokens=False)
            # Also try with a leading space (some tokenizers encode " A" differently)
            if toks:
                ids.append(toks[0])
            toks2 = self._tokenizer.encode(f" {letter}", add_special_tokens=False)
            if toks2 and toks2[-1] not in ids:
                ids.append(toks2[-1])
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

    def _chat_extra_kwargs(self) -> dict:
        """Extra kwargs for llm.chat() that suppress thinking on Qwen3.5."""
        if self._model_family == "qwen3":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def generate_batch(
        self, requests: list[InferenceRequest]
    ) -> list[list[InferenceResult]]:
        from vllm import SamplingParams

        if not requests:
            return []

        results: list[list[InferenceResult]] = [[] for _ in requests]

        for req_idx, req in enumerate(requests):
            if req.messages is None:
                raise ValueError(
                    "VllmBackend requires InferenceRequest.messages to be set. "
                    "Use build_messages() from src.prompts."
                )

            allowed_ids = self._allowed_token_ids(req.allowed_letters)
            params = SamplingParams(
                n=req.n_samples,
                temperature=req.temperature if req.temperature > 0 else 1e-6,
                top_p=req.top_p if req.temperature > 0 else 1.0,
                max_tokens=1,
                logprobs=max(len(req.allowed_letters), 5),
                allowed_token_ids=allowed_ids,
            )

            extra = self._chat_extra_kwargs()
            outputs = self._llm.chat(
                messages=[req.messages],
                sampling_params=params,
                **extra,
            )
            output = outputs[0]  # one conversation → one RequestOutput

            for choice in output.outputs:
                token_text = (choice.text or "").strip().upper()

                logprob_dist: dict[str, float] = {}
                if choice.logprobs:
                    lp_map = choice.logprobs[0]
                    for _tid, lp_obj in lp_map.items():
                        decoded = (lp_obj.decoded_token or "").strip().upper()
                        if decoded in req.allowed_letters:
                            logprob_dist[decoded] = float(lp_obj.logprob)

                for letter in req.allowed_letters:
                    if letter not in logprob_dist:
                        logprob_dist[letter] = -100.0

                letter = (
                    token_text if token_text in req.allowed_letters
                    else req.allowed_letters[0]
                )
                results[req_idx].append(
                    InferenceResult(
                        letter=letter,
                        logprob_dist=logprob_dist,
                        raw_text=choice.text or "",
                    )
                )

        return results

    def generate_text(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Free-form text generation (code_exec, RAG extraction).

        Does NOT apply allowed_token_ids — returns the full output text.
        Uses thinking=True so the model can reason before generating code.
        """
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        extra = {}
        if self._model_family == "qwen3":
            extra = {"chat_template_kwargs": {"enable_thinking": True}}

        outputs = self._llm.chat(
            messages=[messages],
            sampling_params=params,
            **extra,
        )
        return outputs[0].outputs[0].text or ""


# ── thinking-prefix helpers ────────────────────────────────────────────

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def build_chat_prompt_llamacpp(
    system: str,
    user: str,
    thinking_budget: int,
    model_family: str = "qwen3",
) -> str:
    """Render a chat-style prompt for llama.cpp (no tokenizer available).

    For the answer-extraction position we need exactly 1 token (the letter),
    so the thinking block must be closed BEFORE the forced token.

    Qwen3.5 pattern:
      <|im_start|>assistant
      <think>
      </think>          ← thinking block immediately closed
      ← model emits answer letter here (forced via logit_bias)

    Gemma 4: no thinking tokens; the answer letter follows <start_of_turn>model directly.
    """
    if model_family == "qwen3":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{_THINK_OPEN}\n{_THINK_CLOSE}\n"
        )
    # Gemma 4 (it) format — no thinking prefix
    return (
        f"<start_of_turn>system\n{system}<end_of_turn>\n"
        f"<start_of_turn>user\n{user}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def build_chat_messages(system: str, user: str) -> list[dict]:
    """Return messages list for vLLM / HF chat template."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── factory ────────────────────────────────────────────────────────────


def get_model_family(cfg: dict, model_key: str = "primary") -> str:
    """Return the chat-template family tag for a model key (qwen3 or gemma4)."""
    return cfg.get("model_family", {}).get(model_key, "qwen3")


def get_model_family_by_id(model_id: str) -> str:
    """Infer model family from model_id string."""
    mid = model_id.lower()
    if "qwen" in mid:
        return "qwen3"
    if "gemma" in mid:
        return "gemma4"
    return "qwen3"


def build_backend(cfg: dict | None = None, model_key: str = "primary") -> InferenceBackend:
    """Instantiate the correct backend from pipeline_config.yaml."""
    if cfg is None:
        with open(_CFG_PATH) as f:
            cfg = yaml.safe_load(f)

    backend_type: str = cfg.get("backend", "llamacpp")
    model_cfg = cfg["models"]

    if backend_type == "vllm":
        model_id = model_cfg[model_key]
        vllm_cfg = cfg.get("vllm", {})
        return VllmBackend(
            model_id=model_id,
            dtype=vllm_cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.92),
            max_model_len=vllm_cfg.get("max_model_len", 16384),
            enable_prefix_caching=vllm_cfg.get("enable_prefix_caching", True),
            tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        )

    # llamacpp
    gguf_paths = cfg.get("gguf_paths", {})
    model_path = gguf_paths.get(model_key)
    if not model_path:
        raise ValueError(
            f"No GGUF path configured for model_key='{model_key}'. "
            "Set gguf_paths.<key> in pipeline_config.yaml after running:\n"
            "  bash scripts/download_weights.sh gguf"
        )
    llamacpp_cfg = cfg.get("llamacpp", {})
    return LlamaCppBackend(
        model_path=model_path,
        n_gpu_layers=llamacpp_cfg.get("n_gpu_layers", -1),
        n_ctx=llamacpp_cfg.get("n_ctx", 8192),
        verbose=llamacpp_cfg.get("verbose", False),
    )
