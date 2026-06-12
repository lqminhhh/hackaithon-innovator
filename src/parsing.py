"""Question parser — passage split, type flags, choice normalisation.

For each question this module produces a ParsedQuestion with:
  - context:             extracted passage text, or None
  - query:              the actual question sentence
  - options:            {A: ..., B: ..., ...} with letters A..K
  - flags:
      has_context       passage markers found
      is_quantitative   digit/unit/formula-token density above threshold
      has_refusal_choice at least one choice matches a refusal pattern
      is_legal          Vietnamese legal vocabulary density above threshold

Flags are *hints* for template/tool selection; the confidence gate does
the actual routing (planning doc Lesson C).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

# ── passage-split markers ─────────────────────────────────────────────

_PASSAGE_MARKERS: list[re.Pattern] = [
    re.compile(r"Đoạn\s+thông\s+tin\s*:", re.IGNORECASE),
    re.compile(r"Tiêu\s+đề\s*:", re.IGNORECASE),
    re.compile(r"Nội\s+dung\s*:", re.IGNORECASE),
    re.compile(r"Đoạn\s+văn\s*:", re.IGNORECASE),
    re.compile(r"Văn\s+bản\s*:", re.IGNORECASE),
    re.compile(r"Bài\s+đọc\s*:", re.IGNORECASE),
]
_QUESTION_MARKER = re.compile(r"Câu\s+hỏi\s*:", re.IGNORECASE)

# ── quantitative detection ────────────────────────────────────────────

_QUANT_PATTERNS = [
    re.compile(r"\d+[,\.]\d+"),       # decimal numbers
    re.compile(r"\d+\s*%"),           # percentages
    re.compile(r"\d+\s*[kKmMgG][wWhHjJ]"),  # energy/power units
    re.compile(r"\b(?:kg|m|km|cm|mm|s|N|J|W|K|mol|Pa|Hz|Ω|ohm|V|A|F|T|cd)\b"),
    re.compile(r"[=\+\-\×\÷\*\/^√∑∫∂]"),
    re.compile(r"\b(?:GDP|CPI|lãi\s*suất|đạo\s*hàm|xác\s*suất|phương\s*trình|"
               r"công\s*thức|tính|giải|nhiệt|entropy|điện|vận\s*tốc|gia\s*tốc|"
               r"khối\s*lượng|áp\s*suất|nồng\s*độ|mol)\b", re.IGNORECASE),
]
_QUANT_DENSITY_THRESHOLD = 2   # how many pattern matches trigger the flag

# ── refusal patterns ──────────────────────────────────────────────────

_REFUSAL_PATTERNS = [
    re.compile(r"Tôi\s+không\s+thể", re.IGNORECASE),
    re.compile(r"không\s+thể\s+cung\s+cấp", re.IGNORECASE),
    re.compile(r"từ\s+chối", re.IGNORECASE),
    re.compile(r"không\s+được\s+phép", re.IGNORECASE),
    re.compile(r"vi\s+phạm\s+pháp\s+luật", re.IGNORECASE),
    re.compile(r"không\s+hỗ\s+trợ", re.IGNORECASE),
    re.compile(r"không\s+cung\s+cấp", re.IGNORECASE),
]

# ── legal vocabulary ──────────────────────────────────────────────────

_LEGAL_TOKENS = re.compile(
    r"\b(?:Luật|Bộ\s+luật|Pháp\s+lệnh|Nghị\s+định|Thông\s+tư|Quyết\s+định|"
    r"Hiến\s+pháp|Điều\s+lệ|Điều\s*\d+|Khoản\s*\d+|Điểm\s*[a-z]|"
    r"Nghị\s+quyết|Chỉ\s+thị|Biên\s+bản|Hợp\s+đồng|Tòa\s+án|Kiểm\s+sát|"
    r"Thi\s+hành|Vi\s+phạm|Xử\s+phạt|Hình\s+sự|Dân\s+sự|Hành\s+chính)\b",
    re.IGNORECASE,
)
_LEGAL_DENSITY_THRESHOLD = 2

ALL_LABELS = list(string.ascii_uppercase)  # A–Z


@dataclass
class ParsedQuestion:
    qid: str
    context: str | None
    query: str
    options: dict[str, str]
    # flags
    has_context: bool = False
    is_quantitative: bool = False
    has_refusal_choice: bool = False
    is_legal: bool = False
    # metadata
    n_choices: int = 0
    raw_question: str = ""
    audit: dict = field(default_factory=dict)

    @property
    def valid_letters(self) -> list[str]:
        return sorted(self.options.keys())


def parse_question(raw: dict) -> ParsedQuestion:
    """Parse a single question dict from the data loader into a ParsedQuestion."""
    qid = str(raw.get("qid", raw.get("id", "")))
    question_text = str(raw.get("question", ""))
    options_raw: dict[str, str] = raw.get("options", {})

    context, query = _split_passage(question_text)
    options = _normalise_options(options_raw)
    n_choices = len(options)

    has_context = context is not None
    is_quantitative = _detect_quantitative(query, options, context)
    has_refusal_choice = _detect_refusal(options)
    is_legal = _detect_legal(query, options, context)

    return ParsedQuestion(
        qid=qid,
        context=context,
        query=query,
        options=options,
        has_context=has_context,
        is_quantitative=is_quantitative,
        has_refusal_choice=has_refusal_choice,
        is_legal=is_legal,
        n_choices=n_choices,
        raw_question=question_text,
        audit={
            "has_context": has_context,
            "is_quantitative": is_quantitative,
            "has_refusal_choice": has_refusal_choice,
            "is_legal": is_legal,
            "n_choices": n_choices,
        },
    )


def parse_questions(raws: list[dict]) -> list[ParsedQuestion]:
    return [parse_question(r) for r in raws]


# ── internals ─────────────────────────────────────────────────────────


def _split_passage(text: str) -> tuple[str | None, str]:
    """Separate passage context from the question sentence.

    Returns (context, query). If no passage markers are found, context is None
    and query is the full text.
    """
    passage_start = None
    for pat in _PASSAGE_MARKERS:
        m = pat.search(text)
        if m:
            passage_start = m.start()
            break

    q_match = _QUESTION_MARKER.search(text)

    if passage_start is None and q_match is None:
        return None, text.strip()

    if q_match is not None:
        # Everything before Câu hỏi: is context; everything after is query
        context_part = text[:q_match.start()].strip()
        query_part = text[q_match.end():].strip()
        context = context_part if context_part else None
        return context, query_part

    # Has a passage marker but no "Câu hỏi:" — treat whole text as context
    # (the question is embedded in the passage; keep full text as query but flag)
    return None, text.strip()


def _normalise_options(options: dict[str, str]) -> dict[str, str]:
    """Ensure option keys are single uppercase letters A..K, values stripped."""
    result: dict[str, str] = {}
    for i, (k, v) in enumerate(sorted(options.items())):
        label = ALL_LABELS[i] if len(k) != 1 or k not in ALL_LABELS else k
        result[label] = str(v).strip()
    return result


def _detect_quantitative(
    query: str, options: dict[str, str], context: str | None
) -> bool:
    combined = query + " " + " ".join(options.values())
    if context:
        combined += " " + context
    hits = sum(1 for p in _QUANT_PATTERNS if p.search(combined))
    return hits >= _QUANT_DENSITY_THRESHOLD


def _detect_refusal(options: dict[str, str]) -> bool:
    for v in options.values():
        for pat in _REFUSAL_PATTERNS:
            if pat.search(v):
                return True
    return False


def _detect_legal(
    query: str, options: dict[str, str], context: str | None
) -> bool:
    combined = query + " " + " ".join(options.values())
    if context:
        combined += " " + context
    hits = len(_LEGAL_TOKENS.findall(combined))
    return hits >= _LEGAL_DENSITY_THRESHOLD
