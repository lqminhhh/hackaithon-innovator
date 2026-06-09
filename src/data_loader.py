"""Universal data loader for competition input files.

Handles both JSON and CSV formats. Normalises all inputs into a
consistent internal representation regardless of source format.

Expected JSON format (from organiser):
    [{"qid": "test_0001", "question": "...", "choices": ["...", "...", "...", "..."]}, ...]

Expected CSV format:
    id,question,A,B,C,D

Choice text may optionally have "A: " / "B: " prefixes — these are stripped.
Some questions may have fewer than 4 choices — missing slots are filled with empty strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["A", "B", "C", "D"]

_LABEL_PREFIX_RE = re.compile(r"^[A-D][:\.\)]\s*")


def _strip_label_prefix(text: str) -> str:
    """Remove leading 'A: ', 'B. ', 'C) ' etc. from a choice string."""
    return _LABEL_PREFIX_RE.sub("", text).strip()


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
        # Pad to 4 choices if fewer
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

    # Detect column naming: "qid" or "id"
    id_col = "qid" if "qid" in df.columns else "id"

    for _, row in df.iterrows():
        options = {}
        for label in LABELS:
            val = str(row.get(label, "")) if label in row else ""
            options[label] = _strip_label_prefix(val)

        questions.append({
            "qid": str(row[id_col]),
            "question": str(row["question"]),
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
