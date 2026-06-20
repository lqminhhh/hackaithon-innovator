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
    "đoạn văn",
    "nội dung:",
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

_HARMFUL_INTENT_PHRASES = (
    "làm thế nào để hack",
    "làm thế nào để chủ trì",
    "làm thế nào để phá hoại",
    "làm thế nào để phá vỡ",
    "như thế nào để phá hoại",
    "cách hack",
    "cách đánh cắp",
    "cách lừa đảo",
    "cách chế tạo bom",
    "cách sản xuất ma túy",
    "cách vượt mặt",
    "cách qua mặt",
    "cách tấn công",
    "cách xâm nhập",
    "cách phá hoại",
    "cách trốn tránh",
    "cách hiệu quả nhất",
    "hiệu quả nhất để",
    "tránh bị phát hiện khi",
    "làm giả",
    "làm suy yếu",
    "gây khó khăn cho",
    "để phá hoại",
    "để kích động",
    "để xúc phạm",
    "xúc phạm các biểu tượng",
    "ban hành văn bản có nội dung trái",
)

_HARMFUL_KEYWORDS = (
    "chế tạo bom",
    "sản xuất ma túy",
    "hack hệ thống",
    "đánh cắp dữ liệu",
    "đánh cắp thông tin",
    "phát tán tài liệu mật",
    "tiết lộ bí mật nhà nước",
)

_LONG_CONTEXT_CHARS = 600

_QUANT_TERMS = (
    "tính",
    "bao nhiêu",
    "xác suất",
    "kỳ vọng",
    "đạo hàm",
    "tích phân",
    "phương trình",
    "hệ phương trình",
    "vi phân",
    "nồng độ",
    "mol",
    "phản ứng",
    "điện trở",
    "điện áp",
    "công suất",
    "vận tốc",
    "tốc độ",
    "gia tốc",
    "lực",
    "khối lượng",
    "gdp",
    "lạm phát",
    "độ co giãn",
    "lãi suất",
    "ma trận",
    "hằng số",
    "latex",
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
    is_harmful = _is_actionable_harmful(query_text)

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

    if has_passage_marker:
        return text, text

    lowered_options_free = lowered
    has_non_reading_signals = (
        any(term in lowered_options_free for term in _LEGAL_TERMS)
        or any(phrase in lowered_options_free for phrase in _HARMFUL_INTENT_PHRASES)
        or any(kw in lowered_options_free for kw in _HARMFUL_KEYWORDS)
        or any(term in lowered_options_free for term in _QUANT_TERMS)
        or bool(_QUANT_SYMBOL_RE.search(text))
    )
    if len(text) > _LONG_CONTEXT_CHARS and not has_non_reading_signals:
        return text, text

    return None, text


def _looks_quantitative(text: str, options: dict[str, str]) -> bool:
    body = f"{text}\n" + "\n".join(options.values())
    lowered = body.lower()

    keyword_hits = sum(term in lowered for term in _QUANT_TERMS)
    symbol_hits = len(_QUANT_SYMBOL_RE.findall(body))
    digit_count = sum(ch.isdigit() for ch in body)

    return keyword_hits >= 2 or symbol_hits >= 3 or digit_count >= 8


def _is_actionable_harmful(text: str) -> bool:
    """High-precision harmful intent detection.

    Only fires on questions that contain an actionable harmful request (intent
    phrases like "cách hack", "làm thế nào để phá hoại") or specific dangerous
    keywords ("chế tạo bom", "sản xuất ma túy"). Historical, encyclopedic, or
    academic mentions of weapons, drugs, war, etc. do NOT trigger this.
    """
    lowered = text.lower()
    return (
        any(phrase in lowered for phrase in _HARMFUL_INTENT_PHRASES)
        or any(kw in lowered for kw in _HARMFUL_KEYWORDS)
    )


def _is_refusal_option(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _REFUSAL_TERMS)
