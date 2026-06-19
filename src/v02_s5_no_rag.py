"""v02 S5-only runner.

This entrypoint is for measuring Layer-2 semantic routing on top of the
current v02/S4 solver while keeping S6 RAG disabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.v02_alpha import run_v02_alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v02 with S5 semantic routing enabled and RAG disabled"
    )
    parser.add_argument("--input", required=True, help="Path to input file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to output submission CSV")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model override. Defaults to configs/v02_alpha_config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions to process",
    )
    args = parser.parse_args()

    run_v02_alpha(
        args.input,
        args.output,
        args.model_id,
        args.limit,
        use_rag=False,
        use_reranker=False,
        use_semantic_router=True,
    )


if __name__ == "__main__":
    main()
