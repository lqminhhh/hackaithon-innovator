"""Train a LoRA adapter for the Vietnamese MCQ model.

This script intentionally trains only adapter weights. It does not modify the
base model and does not change the inference pipeline. Merge the adapter with
``scripts/merge_lora.py`` before testing it as a single model directory.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

QWEN35_TRANSFORMERS_HELP = """
This environment cannot load Qwen/Qwen3.5 checkpoints because its
Transformers install does not recognize model_type='qwen3_5'.

In Colab, run these commands, then restart the runtime:

  pip uninstall -y transformers tokenizers
  pip install --no-cache-dir git+https://github.com/huggingface/transformers.git
  pip install -U peft accelerate datasets safetensors sentencepiece protobuf

After the restart, rerun scripts/train_lora.py.
""".strip()

TORCHAO_COMPAT_HELP = """
PEFT found an incompatible torchao install while injecting LoRA adapters.
This project trains plain bf16 LoRA and does not need torchao.

In Colab, run this, then restart the runtime:

  pip uninstall -y torchao

After the restart, rerun scripts/train_lora.py.
""".strip()


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _torch_dtype(name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_tokenizer(model_id: str, *, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _is_qwen35_support_error(exc: Exception) -> bool:
    message = str(exc)
    return "qwen3_5" in message or "does not recognize this architecture" in message


def _assert_model_supported(model_id: str, *, trust_remote_code: bool) -> None:
    try:
        AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        if _is_qwen35_support_error(exc):
            raise RuntimeError(QWEN35_TRANSFORMERS_HELP) from exc
        raise


def _load_model(model_cfg: dict[str, Any], *, trust_remote_code: bool):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            torch_dtype=_torch_dtype(model_cfg["dtype"]),
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        if _is_qwen35_support_error(exc):
            raise RuntimeError(QWEN35_TRANSFORMERS_HELP) from exc
        raise


def _apply_lora(model, peft_config: LoraConfig):
    try:
        return get_peft_model(model, peft_config)
    except ImportError as exc:
        if "torchao" in str(exc).lower():
            raise RuntimeError(TORCHAO_COMPAT_HELP) from exc
        raise


def _chat_text(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    chunks: list[str] = []
    for message in messages:
        chunks.append(f"{message['role']}: {message['content']}")
    if add_generation_prompt:
        chunks.append("assistant:")
    return "\n\n".join(chunks)


def _tokenize_record(record: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    messages = record["messages"]
    prompt_messages = messages[:-1]

    prompt_text = _chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = _chat_text(tokenizer, messages, add_generation_prompt=False)

    full = tokenizer(
        full_text,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )
    prompt = tokenizer(
        prompt_text,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )

    labels = list(full["input_ids"])
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    has_target = any(label != -100 for label in labels)
    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
        "has_target": has_target,
    }


def _load_tokenized_datasets(cfg: dict[str, Any], tokenizer) -> DatasetDict:
    data_cfg = cfg["data"]
    raw = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_path"],
            "validation": data_cfg["val_path"],
            "test": data_cfg["test_path"],
        },
    )

    max_seq_length = int(data_cfg["max_seq_length"])
    tokenized = raw.map(
        lambda record: _tokenize_record(record, tokenizer, max_seq_length),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing chat records",
    )
    tokenized = tokenized.filter(lambda record: bool(record["has_target"]))
    tokenized = tokenized.remove_columns(["has_target"])
    return tokenized


def _build_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    train_cfg = cfg["training"]
    dtype = cfg["model"]["dtype"].lower()
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": train_cfg["output_dir"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "weight_decay": train_cfg["weight_decay"],
        "logging_steps": train_cfg["logging_steps"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "seed": train_cfg["seed"],
        "report_to": train_cfg["report_to"],
        "bf16": dtype in {"bf16", "bfloat16"},
        "fp16": dtype in {"fp16", "float16", "half"},
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "logging_first_step": True,
        "save_safetensors": True,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    # Transformers has renamed/removed a few TrainingArguments fields across
    # releases. Build only the arguments supported by the active install.
    if "eval_strategy" in supported_args:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported_args:
        kwargs["evaluation_strategy"] = "steps"

    if "eval_steps" in supported_args:
        kwargs["eval_steps"] = train_cfg["eval_steps"]
    if "save_strategy" in supported_args:
        kwargs["save_strategy"] = "steps"

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_args}
    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    _assert_model_supported(model_cfg["base_model"], trust_remote_code=trust_remote_code)

    tokenizer = _load_tokenizer(
        model_cfg["base_model"],
        trust_remote_code=trust_remote_code,
    )
    datasets = _load_tokenized_datasets(cfg, tokenizer)

    model = _load_model(model_cfg, trust_remote_code=trust_remote_code)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = _apply_lora(model, peft_config)
    model.print_trainable_parameters()

    training_args = _build_training_args(cfg)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
