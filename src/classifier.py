"""Question type classifier.

Routes each question to one of three types so the pipeline can choose
the optimal agent combination.  Uses a fast regex/heuristic approach
rather than a model call to avoid latency overhead.
"""

from __future__ import annotations

import re
from typing import Literal

QuestionType = Literal["FACTUAL", "REASONING", "ARITHMETIC"]

_ARITHMETIC_RE = re.compile(
    r"\d+\s*[\+\-\*/×÷%]\s*\d+"
    r"|tính"
    r"|bao nhiêu"
    r"|phần trăm"
    r"|xác suất"
    r"|diện tích"
    r"|thể tích"
    r"|phương trình"
    r"|tổng\b"
    r"|hiệu\b"
    r"|tích\b"
    r"|thương\b",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"vì sao"
    r"|tại sao"
    r"|giải thích"
    r"|so sánh"
    r"|phân tích"
    r"|nhận xét"
    r"|đánh giá"
    r"|hậu quả"
    r"|nguyên nhân"
    r"|suy luận"
    r"|nếu.*thì"
    r"|ảnh hưởng",
    re.IGNORECASE,
)


def classify(question: str, options: dict[str, str] | None = None) -> QuestionType:
    """Classify a question into FACTUAL, REASONING, or ARITHMETIC."""
    text = question
    if options:
        text = f"{question} {' '.join(options.values())}"

    if _ARITHMETIC_RE.search(text):
        return "ARITHMETIC"
    if _REASONING_RE.search(text):
        return "REASONING"
    return "FACTUAL"
