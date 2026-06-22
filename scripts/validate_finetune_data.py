"""Validate fine-tuning JSONL splits before LoRA training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = {
    "id",
    "source",
    "bucket",
    "question",
    "choices",
    "answer",
    "messages",
}


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _expected_labels(n: int) -> list[str]:
    return [chr(ord("A") + idx) for idx in range(n)]


def _canonical_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _validate_record(record: dict[str, Any], *, path: Path, index: int) -> list[str]:
    prefix = f"{path}:{index + 1}:{record.get('id', '<missing-id>')}"
    errors: list[str] = []

    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"{prefix}: missing fields {sorted(missing)}")

    choices = record.get("choices")
    if not isinstance(choices, dict):
        errors.append(f"{prefix}: choices must be an object")
        return errors

    labels = sorted(choices)
    if not 2 <= len(labels) <= 26:
        errors.append(f"{prefix}: choice count must be 2..26, got {len(labels)}")

    expected = _expected_labels(len(labels))
    if labels != expected:
        errors.append(f"{prefix}: labels must be contiguous A.., got {labels}")

    answer = record.get("answer")
    if answer not in choices:
        errors.append(f"{prefix}: answer {answer!r} is not in choices")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append(f"{prefix}: messages must contain exactly user + assistant")
    else:
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            errors.append(f"{prefix}: message roles must be user, assistant")
        assistant_text = str(assistant.get("content", ""))
        if answer and f"Đáp án: {answer}" not in assistant_text:
            errors.append(f"{prefix}: assistant message must end/include final answer")

    return errors


def validate_splits(paths: dict[str, Path]) -> tuple[list[str], dict[str, Any]]:
    all_errors: list[str] = []
    all_ids: list[str] = []
    canonical_by_split: dict[str, dict[str, list[str]]] = {}
    stats: dict[str, Any] = {"splits": {}}

    for split, path in paths.items():
        rows = _load_jsonl(path)
        bucket_counts = Counter(row.get("bucket") for row in rows)
        source_counts = Counter(row.get("source") for row in rows)
        answer_counts = Counter(row.get("answer") for row in rows)
        quality_counts = Counter(row.get("explanation_quality") for row in rows)
        choice_counts = Counter(len(row.get("choices", {})) for row in rows)

        stats["splits"][split] = {
            "count": len(rows),
            "buckets": dict(bucket_counts),
            "sources": dict(source_counts),
            "answers": dict(answer_counts),
            "explanation_quality": dict(quality_counts),
            "choice_counts": dict(choice_counts),
        }

        canonical_by_split[split] = defaultdict(list)
        for idx, row in enumerate(rows):
            all_errors.extend(_validate_record(row, path=path, index=idx))
            all_ids.append(str(row.get("id")))
            question = str(row.get("question", ""))
            canonical_by_split[split][_canonical_question(question)].append(str(row.get("id")))

    id_counts = Counter(all_ids)
    duplicated_ids = sorted(item for item, count in id_counts.items() if count > 1)
    if duplicated_ids:
        all_errors.append(f"duplicate ids: {duplicated_ids[:20]}")

    split_names = sorted(canonical_by_split)
    cross_split_dupes: list[tuple[str, list[str]]] = []
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap = set(canonical_by_split[left]) & set(canonical_by_split[right])
            for question in sorted(overlap):
                ids = canonical_by_split[left][question] + canonical_by_split[right][question]
                cross_split_dupes.append((question, ids))
    if cross_split_dupes:
        all_errors.append(
            f"exact duplicate questions across splits: {len(cross_split_dupes)} "
            f"(first ids: {cross_split_dupes[0][1]})"
        )

    stats["total"] = sum(split_stats["count"] for split_stats in stats["splits"].values())
    stats["duplicate_id_count"] = len(duplicated_ids)
    stats["exact_cross_split_question_duplicates"] = len(cross_split_dupes)
    return all_errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LoRA fine-tune JSONL splits")
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--stats-output", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    paths = {
        "train": Path(cfg["data"]["train_path"]),
        "val": Path(cfg["data"]["val_path"]),
        "test": Path(cfg["data"]["test_path"]),
    }

    errors, stats = validate_splits(paths)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats_output:
        Path(args.stats_output).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if errors:
        print("\nValidation errors:")
        for error in errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
