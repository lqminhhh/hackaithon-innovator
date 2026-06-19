"""v01 baseline: simplest one-model LLM pipeline.

This preserves the baseline architecture but uses the same configured primary
model as the v02 runners instead of the old Qwen2.5 instruct model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.version_runner import add_common_args, run_v01_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v01_baseline LLM-only pipeline")
    add_common_args(
        parser,
        default_output="data/submissions/submission_v01_baseline.csv",
        default_trace="data/traces/trace_v01_baseline.jsonl",
    )
    args = parser.parse_args()

    run_v01_baseline(
        input_path=args.input,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        limit=args.limit,
        safe_mode=args.safe_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
