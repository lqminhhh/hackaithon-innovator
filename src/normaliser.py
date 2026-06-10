"""Robust answer extraction from raw model output.

Handles many output formats: 'ĐÁP ÁN: B', 'Đáp án là C', 'Câu trả lời: D',
'(A)', bare letters, and completely garbled output.

Supports any set of option labels (not limited to A/B/C/D).
"""

from __future__ import annotations

import re

_DEFAULT_LABELS = ("A", "B", "C", "D")


def normalise_answer(
    raw_output: str,
    valid_labels: tuple[str, ...] | list[str] = _DEFAULT_LABELS,
) -> str:
    """Extract a single answer letter from any model output format.

    ``valid_labels`` should be the sorted label keys for this question
    (e.g. ("A","B","C") for a 3-choice question, or ("A","B","C","D","E")
    for a 5-choice one).  Defaults to A-D for backwards compatibility.
    """
    labels_str = "".join(valid_labels)           # e.g. "ABCDE"
    label_class = f"[{labels_str}]"              # e.g. "[ABCDE]"

    # Layer 1: explicit ĐÁP ÁN / Đáp án format
    m = re.search(
        rf"[ĐĐđ][ÁÁáa]P\s*[ÁÁáa]N[:\s]*({label_class})",
        raw_output,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Layer 2: Vietnamese phrasing variants
    m = re.search(
        rf"(?:đáp\s*án|câu\s*trả\s*lời|chọn|là)[:\s]*({label_class})\b",
        raw_output,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Layer 3: parenthesised letter like (B) or [C]
    m = re.search(rf"[\(\[]\s*({label_class})\s*[\)\]]", raw_output)
    if m:
        return m.group(1).upper()

    # Layer 4: last standalone valid label in the text
    matches = re.findall(rf"\b({label_class})\b", raw_output.upper())
    if matches:
        return matches[-1]

    # Layer 5: most frequently mentioned valid label
    counts = {c: raw_output.upper().count(c) for c in valid_labels}
    return max(counts, key=counts.get)


def parse_confidence(raw_output: str) -> float:
    """Extract the confidence score from model output, defaulting to 0.5."""
    m = re.search(
        r"[ĐĐđ][ỘỘộo]\s*T[ỰỰựu]\s*TIN[:\s]*([-+]?\d*\.?\d+)",
        raw_output,
        re.IGNORECASE,
    )
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return 0.5
