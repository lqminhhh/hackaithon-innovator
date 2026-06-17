#!/usr/bin/env python
"""Run Layer-2 semantic router diagnostics without loading the LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_questions
from src.semantic_router import SemanticRouter
from src.semantic_shadow import (
    build_shadow_records,
    summarize_shadow_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shadow-run the S5 semantic router and write diagnostics."
    )
    parser.add_argument(
        "--input",
        default="data/public-test_1780368312.json",
        help="Input questions JSON/CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/semantic_shadow_v02_gamma.jsonl",
        help="Output JSONL diagnostics path.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional embedding model override. Defaults to semantic_router_config.yaml.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device override, e.g. cpu, cuda, mps.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional semantic router config path.",
    )
    parser.add_argument(
        "--prototypes",
        default=None,
        help="Optional route prototypes YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    if args.limit is not None:
        questions = questions[: args.limit]

    router = SemanticRouter(
        model_name=args.model_name,
        device=args.device,
        config_path=args.config,
        prototypes_path=args.prototypes,
    )
    records = build_shadow_records(questions, router=router)
    summary = summarize_shadow_records(records)

    write_jsonl(records, args.output)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} semantic shadow records to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

