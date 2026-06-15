"""Question parsing utilities for route-aware MCQ handling.

This module prepares question context before any LLM call:
  - splits embedded passage/context from the actual question
  - derives lightweight rule-based flags
  - normalises useful metadata for downstream routing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

_PASSAGE_START_PATTERNS = (
    "đoạn thông tin",
    "nội dung",
    "title:",
    "content:",
    "tiêu đề",
    "-- document --",
)

_QUESTION_SPLIT_RE = re.compile(
    r"(?is)\b(câu hỏi|question)\s*:\s*"
)

_LEGAL_TERMS = (
    "luật",
    "nghị định",
    "thông tư",
    "điều ",
    "khoản ",
    "quy định",
    "thủ tục",
    "hành chính",
    "công dân",
    "căn cước",
    "xử phạt",
)

_REFUSAL_TERMS = (
    "tôi không thể",
    "không thể cung cấp",
    "không thể hỗ trợ",
    "từ chối",
    "không được phép hỗ trợ",
)

_HARMFUL_TERMS = (
    "hack",
    "đánh cắp",
    "trộm",
    "lừa đảo",
    "vượt mặt",
    "qua mặt",
    "tấn công",
    "tránh bị phát hiện",
    "chế tạo bom",
    "ma túy",
    "vũ khí",
    "xâm nhập",
    "phá hoại",
)

_QUANT_TERMS = (
    "tính",
    "xác suất",
    "đạo hàm",
    "tích phân",
    "phương trình",
    "vi phân",
    "nồng độ",
    "điện trở",
    "điện áp",
    "công suất",
    "gdp",
    "lạm phát",
    "độ co giãn",
    "mol",
    "hằng số",
    "phản ứng",
    "vận tốc",
)

_QUANT_SYMBOL_RE = re.compile(r"[\d=+\-*/%^√π∞$<>]|\\frac|\\int|\\sum|ax|bx|dx|dt")


@dataclass(slots=True)
class ParsedQuestion:
    qid: str
    original_question: str
    query: str
    context: str | None
    options: dict[str, str]
    refusal_labels: tuple[str, ...]
    n_choices: int
    has_context: bool
    is_quantitative: bool
    is_legal: bool
    has_refusal_choice: bool
    is_harmful: bool

    def to_dict(self) -> dict:
        return asdict(self)


def parse_question(question: dict) -> ParsedQuestion:
    """Parse one normalized question dict from data_loader."""
    raw_question = question["question"].strip()
    options = question["options"]
    context, query = _split_context_and_query(raw_question)

    refusal_labels = tuple(
        label for label, value in options.items()
        if _is_refusal_option(value)
    )
    option_text = " ".join(options.values()).lower()
    query_plus_options = f"{query}\n{option_text}"
    full_text = f"{raw_question}\n{option_text}".lower()
    query_text = query_plus_options.lower()

    has_context = context is not None
    is_quantitative = _looks_quantitative(query, options)
    is_legal = any(term in full_text for term in _LEGAL_TERMS)
    has_refusal_choice = any(term in option_text for term in _REFUSAL_TERMS)
    is_harmful = any(term in query_text for term in _HARMFUL_TERMS)

    return ParsedQuestion(
        qid=question["qid"],
        original_question=raw_question,
        query=query,
        context=context,
        options=options,
        refusal_labels=refusal_labels,
        n_choices=len(options),
        has_context=has_context,
        is_quantitative=is_quantitative,
        is_legal=is_legal,
        has_refusal_choice=bool(refusal_labels),
        is_harmful=is_harmful,
    )


def _split_context_and_query(text: str) -> tuple[str | None, str]:
    """Split passage/document context from the actual question if present."""
    lowered = text.lower()
    has_passage_marker = any(marker in lowered for marker in _PASSAGE_START_PATTERNS)
    match = _QUESTION_SPLIT_RE.search(text)

    if has_passage_marker and match:
        context = text[: match.start()].strip()
        query = text[match.end() :].strip()
        if context and query:
            return context, query

    return None, text


def _looks_quantitative(text: str, options: dict[str, str]) -> bool:
    body = f"{text}\n" + "\n".join(options.values())
    lowered = body.lower()

    keyword_hits = sum(term in lowered for term in _QUANT_TERMS)
    symbol_hits = len(_QUANT_SYMBOL_RE.findall(body))
    digit_count = sum(ch.isdigit() for ch in body)
    n_choices = len(options)

    # 10-option questions in this set are overwhelmingly STEM-like; use that as a hint.
    if n_choices >= 8 and (keyword_hits >= 1 or digit_count >= 3):
        return True

    return keyword_hits >= 2 or symbol_hits >= 3 or digit_count >= 8


def _is_refusal_option(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _REFUSAL_TERMS)
