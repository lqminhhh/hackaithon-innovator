"""Self-consistency policy helpers for final-compliant runners."""

from __future__ import annotations

import random
import re

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

_READING_REASON_MARKERS = (
    "lý do",
    "lí do",
    "mục đích",
    "nguyên nhân",
    "vì sao",
    "tại sao",
    "do đâu",
    "nhằm mục đích",
    "để làm gì",
)

_READING_DETAIL_MARKERS = (
    "theo ngữ cảnh",
    "theo nội dung",
    "theo thông tin",
    "theo đoạn",
    "lần đầu tiên",
    "đầu tiên",
    "vào năm nào",
    "ngày nào",
    "khi nào",
    "bao giờ",
    "ở đâu",
    "điều gì đã xảy ra",
    "hệ quả",
    "nhận định nào",
    "thuộc nhóm nào",
    "biệt danh",
    "nguồn gốc",
)

_COMBINATION_OPTION_MARKERS = (
    "tất cả các đáp án trên",
    "tất cả các phương án trên",
    "cả a, b, c",
    "cả a,b,c",
    "cả ba đáp án",
    "cả 3 đáp án",
)

_OPTION_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_OPTION_STOPWORDS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "các", "đáp", "án", "phương", "trả", "lời", "một", "những", "sau",
    "đây", "là", "và", "cho", "của", "trên", "dưới", "với", "trong",
    "về", "do", "từ", "được", "không", "có", "nào", "gì", "theo",
}


def stem_sc_n(margin: float | None, adaptive_sc: bool) -> int:
    """Return the SC sample count for a STEM question."""
    if adaptive_sc and margin is not None and margin < MARGIN_LOW_BY_ROUTE["STEM"]:
        return SC_N_STEM["low"]
    return SC_N_STEM["high"]


def reading_escalation_reason(query: str) -> str | None:
    """Return the SC reason for reading questions that need rereading."""
    lowered = query.lower()
    if any(marker in lowered for marker in _READING_REASON_MARKERS):
        return "reading_reason_purpose_sc"
    if any(marker in lowered for marker in _READING_DETAIL_MARKERS):
        return "reading_detail_lookup_sc"
    return None


def has_combination_option(options: dict[str, str]) -> bool:
    """Return True when an option collapses multiple other choices."""
    lowered_options = [text.lower() for text in options.values()]
    return any(
        any(marker in text for marker in _COMBINATION_OPTION_MARKERS)
        for text in lowered_options
    )


def has_ambiguous_options(options: dict[str, str]) -> bool:
    """Return True when options are duplicated or lexically too similar."""
    normalized = [
        " ".join(_OPTION_TOKEN_RE.findall(text.lower()))
        for text in options.values()
    ]
    if len(set(normalized)) < len(normalized):
        return True

    token_sets = [
        {
            token for token in _OPTION_TOKEN_RE.findall(text.lower())
            if len(token) > 1 and token not in _OPTION_STOPWORDS
        }
        for text in options.values()
    ]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            shared = token_sets[i] & token_sets[j]
            if len(shared) < 2:
                continue
            smaller = min(len(token_sets[i]), len(token_sets[j]))
            if smaller and len(shared) / smaller >= 0.6:
                return True
    return False


def knowledge_requires_extra_compute(parsed: ParsedQuestion) -> bool:
    """Return True for knowledge questions that deserve extra compute."""
    return (
        parsed.n_choices >= HIGH_CHOICE_KNOWLEDGE_MIN_CHOICES
        or has_ambiguous_options(parsed.options)
        or has_combination_option(parsed.options)
    )


def knowledge_escalation_reason(
    parsed: ParsedQuestion,
    margin: float | None,
) -> str | None:
    """Return the SC reason for knowledge questions that need more compute."""
    if margin is not None and margin < MARGIN_LOW_BY_ROUTE["KNOWLEDGE"]:
        return f"knowledge_low_margin_{margin:.3f}"
    if parsed.n_choices >= HIGH_CHOICE_KNOWLEDGE_MIN_CHOICES:
        return f"knowledge_high_choice_n{parsed.n_choices}"
    if has_ambiguous_options(parsed.options):
        return "knowledge_ambiguous_options_sc"
    if has_combination_option(parsed.options):
        return "knowledge_combination_option_sc"
    return None


def should_use_think_mode(
    parsed: ParsedQuestion,
    route: str,
    *,
    stage: str = "wave1",
) -> bool:
    """Return whether a route should use think-mode generation."""
    if route == "stem":
        return True
    if route == "knowledge":
        if stage == "wave1":
            return parsed.n_choices >= HIGH_CHOICE_KNOWLEDGE_MIN_CHOICES
        return knowledge_requires_extra_compute(parsed)
    if route == "reading":
        return (
            stage == "wave2"
            and reading_escalation_reason(parsed.query) is not None
        )
    return False


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
