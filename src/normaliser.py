"""Robust answer extraction from raw model output.

Handles many output formats: 'ĐÁP ÁN: B', 'Đáp án là C', 'Câu trả lời: D',
'(A)', bare letters, and completely garbled output.
"""

from __future__ import annotations

import re


def normalise_answer(raw_output: str) -> str:
    """Extract a single A/B/C/D letter from any model output format."""
    # Layer 1: explicit ĐÁP ÁN / Đáp án format
    m = re.search(r"[ĐĐđ][ÁÁáa]P\s*[ÁÁáa]N[:\s]*([ABCD])", raw_output, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Layer 2: Vietnamese phrasing variants
    m = re.search(
        r"(?:đáp\s*án|câu\s*trả\s*lời|chọn|là)[:\s]*([ABCD])\b",
        raw_output,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Layer 3: parenthesised letter like (B) or [C]
    m = re.search(r"[\(\[]\s*([ABCD])\s*[\)\]]", raw_output)
    if m:
        return m.group(1).upper()

    # Layer 4: last standalone A/B/C/D in the text
    matches = re.findall(r"\b([ABCD])\b", raw_output.upper())
    if matches:
        return matches[-1]

    # Layer 5: most frequently mentioned letter
    counts = {c: raw_output.upper().count(c) for c in "ABCD"}
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
