"""v03 epsilon: v03_delta-compatible runner with safer continuation batching.

This wrapper keeps the current v02_gamma/v03_delta logic and only adds a safer
default continuation chunk size for extraction, while writing epsilon-named
submission and trace files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sc_policy import SC_N_STEM
from src.v02_gamma import run_v02_gamma
from src.version_runner import add_common_args


def main() -> None:
    os.environ.setdefault("EXTRACT_CONTINUATION_CHUNK_SIZE", "32")

    parser = argparse.ArgumentParser(
        description="Run v03_epsilon (v03_delta-compatible, chunked extraction)"
    )
    add_common_args(
        parser,
        default_output="data/submissions/submission_v03_epsilon.csv",
        default_trace="data/traces/trace_v03_epsilon.jsonl",
        include_sc_batch=False,
    )
    parser.add_argument(
        "--no-adaptive-sc",
        dest="adaptive_sc",
        action="store_false",
        default=True,
        help=(
            "Disable adaptive SC depth for STEM. "
            "When set, STEM always uses SC_N_STEM['high'] "
            f"(n={SC_N_STEM['high']}) regardless of margin. "
            "Default: adaptive (n=3 if margin high, n=7 if low)."
        ),
    )
    args = parser.parse_args()

    run_v02_gamma(
        input_path=args.input,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        limit=args.limit,
        safe_mode=args.safe_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        adaptive_sc=args.adaptive_sc,
    )


if __name__ == "__main__":
    main()
