"""S1 tests for the vLLM/Qwen wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GPU_MEM_UTIL, LLM_MODEL
from src.llm import LLM
from src.models import load_vllm_primary


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEngine:
    def __init__(self, *, reject_chat_template_kwargs: bool = False):
        self.reject_chat_template_kwargs = reject_chat_template_kwargs
        self.calls = []

    def get_tokenizer(self):
        return "tokenizer"

    def chat(self, conversations, params, **kwargs):
        self.calls.append(
            {
                "conversations": conversations,
                "params": params,
                "kwargs": kwargs,
            }
        )
        if self.reject_chat_template_kwargs and "chat_template_kwargs" in kwargs:
            raise TypeError("chat_template_kwargs unsupported")
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"out-{i}",
                        logprobs=[{"A": -0.1}],
                    )
                ]
            )
            for i, _ in enumerate(conversations)
        ]

    def generate(self, prompts, params, **kwargs):
        self.calls.append(
            {
                "prompts": prompts,
                "params": params,
                "kwargs": kwargs,
            }
        )
        return [("generated", prompts, params, kwargs)]


class _FakeEngineCls:
    last_kwargs = None

    def __new__(cls, **kwargs):
        cls.last_kwargs = kwargs
        return _FakeEngine()


def test_llm_constructor_uses_s1_vllm_defaults():
    llm = LLM(engine_cls=_FakeEngineCls, sampling_params_cls=_FakeSamplingParams)

    assert llm.model == LLM_MODEL
    assert _FakeEngineCls.last_kwargs["model"] == LLM_MODEL
    assert "quantization" not in _FakeEngineCls.last_kwargs
    assert _FakeEngineCls.last_kwargs["gpu_memory_utilization"] == GPU_MEM_UTIL
    assert _FakeEngineCls.last_kwargs["enable_prefix_caching"] is True
    assert _FakeEngineCls.last_kwargs["trust_remote_code"] is True


def test_generate_text_batches_prompts_and_passes_thinking_flag():
    engine = _FakeEngine()
    llm = LLM(engine=engine, sampling_params_cls=_FakeSamplingParams)

    outputs = llm.generate_text(
        ["p1", "p2"],
        mode="think",
        max_tokens=123,
        temperature=0.6,
        top_p=0.95,
        logprobs=5,
    )

    assert [o.text for o in outputs] == ["out-0", "out-1"]
    assert outputs[0].logprobs == [{"A": -0.1}]
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["kwargs"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert call["conversations"][0][0]["content"] == "p1"
    assert call["params"].kwargs == {
        "temperature": 0.6,
        "max_tokens": 123,
        "top_p": 0.95,
        "logprobs": 5,
    }


def test_generate_text_falls_back_to_mode_tags_when_template_kwargs_unsupported():
    engine = _FakeEngine(reject_chat_template_kwargs=True)
    llm = LLM(engine=engine, sampling_params_cls=_FakeSamplingParams)

    outputs = llm.generate_text(["prompt"], mode="no_think")

    assert [o.text for o in outputs] == ["out-0"]
    assert len(engine.calls) == 2
    fallback_call = engine.calls[1]
    assert fallback_call["kwargs"] == {}
    assert fallback_call["conversations"][0][0]["content"].startswith("/no_think\n")


def test_compatibility_methods_delegate_to_engine():
    engine = _FakeEngine()
    llm = LLM(engine=engine, sampling_params_cls=_FakeSamplingParams)
    params = _FakeSamplingParams(max_tokens=1, temperature=0)

    assert llm.get_tokenizer() == "tokenizer"
    assert llm.generate(["p"], params) == [("generated", ["p"], params, {})]


def test_lora_request_is_passed_to_chat_and_raw_generate():
    class _FakeLoRARequest:
        def __init__(self, name, int_id, path):
            self.name = name
            self.int_id = int_id
            self.path = path

    engine = _FakeEngine()
    llm = LLM(
        engine=engine,
        sampling_params_cls=_FakeSamplingParams,
        lora_adapter_path="/tmp/adapter",
        lora_name="adapter",
        lora_int_id=7,
        lora_request_cls=_FakeLoRARequest,
    )

    assert llm.init_kwargs["enable_lora"] is True

    llm.generate_text(["prompt"], mode="no_think")
    chat_kwargs = engine.calls[-1]["kwargs"]
    assert chat_kwargs["lora_request"].name == "adapter"
    assert chat_kwargs["lora_request"].int_id == 7
    assert chat_kwargs["lora_request"].path == "/tmp/adapter"

    params = _FakeSamplingParams(max_tokens=1, temperature=0)
    llm.raw_generate([{"prompt_token_ids": [1, 2, 3]}], params)
    generate_kwargs = engine.calls[-1]["kwargs"]
    assert generate_kwargs["lora_request"].name == "adapter"


def test_load_vllm_primary_uses_s1_wrapper_defaults(monkeypatch):
    captured = {}

    class _CapturedLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import src.llm as llm_module

    monkeypatch.setattr(llm_module, "LLM", _CapturedLLM)

    loaded = load_vllm_primary()

    assert isinstance(loaded, _CapturedLLM)
    assert captured["model"] == LLM_MODEL
    assert captured["quantization"] is None
    assert captured["gpu_memory_utilization"] == GPU_MEM_UTIL
    assert captured["enable_prefix_caching"] is True


def test_load_vllm_primary_uses_awq_only_for_awq_model_names(monkeypatch):
    captured = {}

    class _CapturedLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import src.llm as llm_module

    monkeypatch.setattr(llm_module, "LLM", _CapturedLLM)

    load_vllm_primary(model_id="Qwen/Qwen3-8B-AWQ")

    assert captured["model"] == "Qwen/Qwen3-8B-AWQ"
    assert captured["quantization"] == "awq"


def test_load_vllm_primary_forwards_lora_adapter(monkeypatch):
    captured = {}

    class _CapturedLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import src.llm as llm_module

    monkeypatch.setattr(llm_module, "LLM", _CapturedLLM)

    load_vllm_primary(
        model_id="Qwen/Qwen3.5-4B",
        lora_adapter_path="/tmp/adapter",
        lora_name="adapter",
        lora_int_id=3,
    )

    assert captured["model"] == "Qwen/Qwen3.5-4B"
    assert captured["lora_adapter_path"] == "/tmp/adapter"
    assert captured["lora_name"] == "adapter"
    assert captured["lora_int_id"] == 3
