"""Universal data loader for competition input files.

Handles both JSON and CSV formats. Normalises all inputs into a
consistent internal representation regardless of source format.

Supported formats:

1. JSON (organiser format):
   [{"qid": "test_0001", "question": "...", "choices": ["...", "...", "..."]}, ...]

2. CSV with separate columns:
   qid,question,A,B,C,D

3. CSV with choices embedded in the question text:
   qid,question
   1,"What is X? A. foo B. bar C. baz D. qux"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["A", "B", "C", "D"]

_LABEL_PREFIX_RE = re.compile(r"^[A-D][:\.\)]\s*")

_INLINE_CHOICES_RE = re.compile(
    r"^(.*?)\s*"
    r"A[.\):\s]+(.+?)\s+"
    r"B[.\):\s]+(.+?)\s+"
    r"C[.\):\s]+(.+?)\s+"
    r"D[.\):\s]+(.+?)\s*$",
    re.DOTALL,
)


def _strip_label_prefix(text: str) -> str:
    """Remove leading 'A: ', 'B. ', 'C) ' etc. from a choice string."""
    return _LABEL_PREFIX_RE.sub("", text).strip()


def _parse_inline_choices(text: str) -> tuple[str, dict[str, str]] | None:
    """Try to split 'question A. x B. y C. z D. w' into question + options."""
    m = _INLINE_CHOICES_RE.match(text)
    if not m:
        return None
    question = m.group(1).strip()
    options = {
        "A": m.group(2).strip(),
        "B": m.group(3).strip(),
        "C": m.group(4).strip(),
        "D": m.group(5).strip(),
    }
    return question, options


def load_questions(path: str | Path) -> list[dict]:
    """Load questions from JSON or CSV, returning a normalised list.

    Each returned dict has:
        qid:      str
        question: str
        options:  {"A": "...", "B": "...", "C": "...", "D": "..."}
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
        choices = item.get("choices", [])
        while len(choices) < 4:
            choices.append("")

        options = {}
        for i, label in enumerate(LABELS):
            options[label] = _strip_label_prefix(str(choices[i]))

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
    has_separate_choices = all(label in df.columns for label in LABELS)

    for _, row in df.iterrows():
        if has_separate_choices:
            question_text = str(row["question"])
            options = {}
            for label in LABELS:
                options[label] = _strip_label_prefix(str(row[label]))
        else:
            # Choices are embedded in the question text: "question? A. x B. y C. z D. w"
            parsed = _parse_inline_choices(str(row["question"]))
            if parsed:
                question_text, options = parsed
            else:
                question_text = str(row["question"])
                options = {"A": "", "B": "", "C": "", "D": ""}

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
