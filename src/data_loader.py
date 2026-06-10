"""Universal data loader for competition input files.

Handles both JSON and CSV formats. Normalises all inputs into a
consistent internal representation regardless of source format.
Supports any number of choices (not limited to A/B/C/D).

Supported formats:

1. JSON (organiser format):
   [{"qid": "test_0001", "question": "...", "choices": ["...", "...", "..."]}, ...]

2. CSV with separate columns (auto-detected by single uppercase letter columns):
   qid,question,A,B,C,D   (or A,B,C or A,B,C,D,E, etc.)

3. CSV with choices embedded in the question text:
   qid,question
   1,"What is X? A. foo B. bar C. baz D. qux"
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import pandas as pd

ALL_LABELS = list(string.ascii_uppercase)  # A-Z

_LABEL_PREFIX_RE = re.compile(r"^[A-Z][:\.\)]\s*")

_INLINE_CHOICES_RE = re.compile(
    r"^(.*?)\s+"
    r"([A-Z][.\):\s]+.+)$",
    re.DOTALL,
)

_SINGLE_INLINE_CHOICE_RE = re.compile(
    r"([A-Z])\s*[.\):\s]\s*(.+?)(?=\s+[A-Z][.\):\s]|$)",
)


def _strip_label_prefix(text: str) -> str:
    """Remove leading 'A: ', 'B. ', 'C) ' etc. from a choice string."""
    return _LABEL_PREFIX_RE.sub("", text).strip()


def _choices_have_positional_prefixes(choices: list[str]) -> bool:
    """Return True if every choice starts with its expected label (A., B., ...)."""
    if not choices:
        return False
    for i, choice in enumerate(choices):
        expected = ALL_LABELS[i]
        if not re.match(rf"^{expected}\s*[:\.\)]\s*", str(choice)):
            return False
    return True


def _parse_inline_choices(text: str) -> tuple[str, dict[str, str]] | None:
    """Try to split 'question A. x B. y ...' into question + options dict.

    Works with any number of choices (A through Z).
    """
    m = _INLINE_CHOICES_RE.match(text)
    if not m:
        return None
    question = m.group(1).strip()
    choices_block = m.group(2)
    pairs = _SINGLE_INLINE_CHOICE_RE.findall(choices_block)
    if len(pairs) < 2:
        return None
    options = {label: value.strip() for label, value in pairs}
    return question, options


def load_questions(path: str | Path) -> list[dict]:
    """Load questions from JSON or CSV, returning a normalised list.

    Each returned dict has:
        qid:      str
        question: str
        options:  {"A": "...", "B": "...", ...}  (key count matches number of choices)
    """
    path = Path(path)
    if path.suffix == ".json":
        return _load_json(path)
    if path.suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}  (expected .json or .csv)")


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    questions = []
    for item in raw:
        choices = [str(c) for c in item.get("choices", [])]
        has_prefixes = _choices_have_positional_prefixes(choices)

        options = {}
        for i, c in enumerate(choices):
            text = _strip_label_prefix(c) if has_prefixes else c.strip()
            options[ALL_LABELS[i]] = text

        questions.append({
            "qid": str(item.get("qid", item.get("id", ""))),
            "question": str(item.get("question", "")),
            "options": options,
        })
    return questions


def _load_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    questions = []

    id_col = "qid" if "qid" in df.columns else "id"
    choice_cols = [c for c in df.columns if len(c) == 1 and c in ALL_LABELS]
    choice_cols.sort()

    for _, row in df.iterrows():
        if choice_cols:
            question_text = str(row["question"])
            raw_choices = [str(row[label]) for label in choice_cols]
            has_prefixes = _choices_have_positional_prefixes(raw_choices)
            options = {}
            for label, raw in zip(choice_cols, raw_choices):
                options[label] = _strip_label_prefix(raw) if has_prefixes else raw.strip()
        else:
            parsed = _parse_inline_choices(str(row["question"]))
            if parsed:
                question_text, options = parsed
            else:
                question_text = str(row["question"])
                options = {}

        questions.append({
            "qid": str(row[id_col]),
            "question": question_text,
            "options": options,
        })
    return questions


def write_submission(results: list[dict], output_path: str | Path):
    """Write submission file matching the input format.

    Always writes CSV with columns: id, answer
    (competition standard format).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if "qid" in df.columns:
        df = df.rename(columns={"qid": "id"})
    df.to_csv(output_path, index=False)
