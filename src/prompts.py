"""Prompt builder for the Entropy-Gated Jury.

Three tier-1 templates (Vietnamese system prompts):
  1. context_grounded  — has_context=True  — "trả lời CHỈ dựa trên đoạn"
  2. quantitative      — is_quantitative=True — step-by-step calculation
  3. general_knowledge — default — expert persona + safety handling

Plus tier-2 variants:
  - self_consistency   — same template, different temperature
  - evidence_extract   — re-ask escalated comprehension questions
  - code_exec_prompt   — ask the model to emit runnable Python

Template selection hierarchy (applied in parse order):
  has_context                → context_grounded
  is_quantitative            → quantitative
  else                       → general_knowledge

Thinking budgets (tokens) per template — tunable in config:
  context_grounded:   400
  quantitative:      1000
  general_knowledge:  600
"""

from __future__ import annotations

import string
from pathlib import Path

import yaml

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_raw_prompts() -> dict:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


# ── system prompt templates ────────────────────────────────────────────

_SYS_CONTEXT_GROUNDED = (
    "Bạn là chuyên gia phân tích văn bản. "
    "Trả lời CHỈ dựa trên đoạn thông tin được cung cấp. "
    "Không sử dụng kiến thức bên ngoài. "
    "Đọc kỹ đoạn văn, xác định bằng chứng, rồi chọn đáp án."
)

_SYS_QUANTITATIVE = (
    "Bạn là chuyên gia giải toán, vật lý, hóa học và kinh tế. "
    "Giải từng bước một, kiểm tra lại phép tính trước khi chọn đáp án. "
    "Trình bày rõ ràng: đặt công thức, thay số, tính kết quả, đối chiếu các lựa chọn."
)

_SYS_GENERAL_KNOWLEDGE = (
    "Bạn là chuyên gia đa lĩnh vực với kiến thức sâu rộng về luật pháp, lịch sử, "
    "địa lý, khoa học, văn hóa và các vấn đề xã hội Việt Nam. "
    "Nếu câu hỏi yêu cầu thực hiện hành vi vi phạm pháp luật hoặc gây hại "
    "và có lựa chọn từ chối, hãy chọn lựa chọn từ chối đó."
)

_SYS_EVIDENCE_EXTRACT = (
    "Bạn là chuyên gia phân tích văn bản. "
    "Trước tiên, trích dẫn câu hoặc đoạn chứa bằng chứng trực tiếp từ đoạn văn. "
    "Sau đó, dựa trên bằng chứng đó, chọn đáp án đúng. "
    "Trả lời CHỈ dựa trên đoạn thông tin được cung cấp."
)

_SYS_CODE_EXEC = (
    "Bạn là chuyên gia lập trình Python. "
    "Viết một chương trình Python ngắn để tính kết quả của bài toán. "
    "Chương trình phải tự chứa (không import ngoài whitelist), "
    "kết thúc bằng lệnh print(answer) trong đó answer là giá trị số của đáp án. "
    "Chỉ viết code, không giải thích."
)

SYSTEM_PROMPTS = {
    "context_grounded": _SYS_CONTEXT_GROUNDED,
    "quantitative": _SYS_QUANTITATIVE,
    "general_knowledge": _SYS_GENERAL_KNOWLEDGE,
    "evidence_extract": _SYS_EVIDENCE_EXTRACT,
    "code_exec": _SYS_CODE_EXEC,
}


def select_template(
    has_context: bool,
    is_quantitative: bool,
    tier2_evidence: bool = False,
    code_exec: bool = False,
) -> str:
    """Return the template key for a question's flags."""
    if code_exec:
        return "code_exec"
    if tier2_evidence and has_context:
        return "evidence_extract"
    if has_context:
        return "context_grounded"
    if is_quantitative:
        return "quantitative"
    return "general_knowledge"


def thinking_budget(template: str, cfg: dict | None = None) -> int:
    """Return the max thinking tokens for a template from config."""
    if cfg is None:
        cfg = _load_config()
    budgets = cfg.get("thinking_budgets", {})
    defaults = {
        "context_grounded": 400,
        "quantitative": 1000,
        "general_knowledge": 600,
        "evidence_extract": 500,
        "code_exec": 800,
    }
    return budgets.get(template, defaults.get(template, 600))


def build_user_message(
    query: str,
    options: dict[str, str],
    context: str | None = None,
    template: str = "general_knowledge",
) -> str:
    """Build the user-turn message for any template."""
    labels = sorted(options.keys())
    options_block = "\n".join(f"{l}. {options[l]}" for l in labels)
    valid_hint = "/".join(labels)

    if template in ("context_grounded", "evidence_extract") and context:
        return (
            f"Đoạn thông tin:\n---\n{context}\n---\n\n"
            f"Câu hỏi: {query}\n\n"
            f"Các lựa chọn:\n{options_block}\n\n"
            f"Chỉ chọn một đáp án ({valid_hint})."
        )

    if template == "code_exec":
        return (
            f"Bài toán: {query}\n\n"
            f"Các lựa chọn:\n{options_block}\n\n"
            "Viết code Python để tính và in ra giá trị số của đáp án đúng."
        )

    base = f"Câu hỏi: {query}\n\nCác lựa chọn:\n{options_block}\n\nChọn đáp án đúng ({valid_hint})."
    return base


def get_system_prompt(template: str) -> str:
    return SYSTEM_PROMPTS.get(template, _SYS_GENERAL_KNOWLEDGE)
