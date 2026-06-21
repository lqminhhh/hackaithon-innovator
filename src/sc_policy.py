"""Self-consistency policy helpers for final-compliant runners."""

from __future__ import annotations

import random

from src.config import GPU_MEM_UTIL, SC_TEMP
from src.parser import ParsedQuestion

# Per-route low-margin thresholds. These differ significantly by route.
MARGIN_LOW_BY_ROUTE = {
    "READING": 0.10,
    "STEM": 0.15,
    "KNOWLEDGE": 0.20,
    "SAFETY": 0.05,
}

# STEM SC depth is adaptive, not an early-exit toggle.
SC_N_STEM = {"high": 3, "low": 7}

SC_N_DEFAULT = 5
SC_N_HIGH_CHOICE_KNOWLEDGE = 3
HIGH_CHOICE_KNOWLEDGE_MIN_CHOICES = 8
SC_TOP_P = 0.95
SC_SEED = 1234
SHUFFLE_OPTIONS = True

TOKENS_BY_ROUTE = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

GAMMA_GPU_MEM_UTIL = GPU_MEM_UTIL
GAMMA_MAX_MODEL_LEN = 4096


def stem_sc_n(margin: float | None, adaptive_sc: bool) -> int:
    """Return the SC sample count for a STEM question."""
    if adaptive_sc and margin is not None and margin < MARGIN_LOW_BY_ROUTE["STEM"]:
        return SC_N_STEM["low"]
    return SC_N_STEM["high"]


def shuffle_options(
    options: dict[str, str],
    sample_idx: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Shuffle option values across labels and return a reverse label map.

    ``reverse_map[new_label] = original_label`` so voted letters can be
    remapped after extraction.
    """
    if not SHUFFLE_OPTIONS:
        identity = {key: key for key in options}
        return options, identity

    labels = sorted(options.keys())
    values = [options[label] for label in labels]
    random.Random(SC_SEED + sample_idx).shuffle(values)
    shuffled = dict(zip(labels, values))
    value_to_original = {value: key for key, value in options.items()}
    reverse_map = {label: value_to_original[shuffled[label]] for label in labels}
    return shuffled, reverse_map


def build_sc_reasoning_prompt(
    parsed: ParsedQuestion,
    route: str,
    options: dict[str, str],
) -> str:
    """Build a reasoning prompt with explicit options for SC samples."""
    options_block = "\n".join(f"{label}) {options[label]}" for label in sorted(options))
    route_instruction: dict[str, str] = {
        "reading": (
            "Chỉ dựa vào đoạn thông tin trong câu hỏi. "
            "Nếu câu hỏi hỏi lý do, mục đích hoặc nguyên nhân, hãy tìm câu nêu trực tiếp quan hệ đó "
            "và loại chi tiết chỉ được nhắc đến nhưng không trả lời đúng quan hệ."
        ),
        "stem": "Giải từng bước, kiểm tra phép tính hoặc lập luận định lượng.",
        "knowledge": "Suy luận cẩn thận bằng kiến thức chung.",
        "safety": "Đánh giá ranh giới an toàn và từ chối nếu yêu cầu gây hại.",
    }[route]

    context = ""
    if route == "reading" and parsed.context:
        context = f"Đoạn thông tin:\n---\n{parsed.context}\n---\n\n"

    return (
        "Bạn là một chuyên gia giải câu hỏi trắc nghiệm tiếng Việt.\n"
        "Đây là bài thi trắc nghiệm. Chọn phương án đúng nhất.\n"
        "Chỉ chọn phương án từ chối khi câu hỏi thực sự yêu cầu hướng dẫn hành vi phạm pháp hoặc gây hại.\n"
        f"{route_instruction}\n\n"
        f"{context}"
        f"Câu hỏi:\n{parsed.query}\n\n"
        f"Các lựa chọn:\n{options_block}\n\n"
        "Hãy suy nghĩ ngắn gọn nhưng đủ chắc chắn trước khi chọn đáp án."
    )
