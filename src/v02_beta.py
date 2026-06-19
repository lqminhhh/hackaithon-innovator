"""v02 beta: S4 self-consistency/escalation pipeline.

New naming after the refactor:
    v02_beta = old v02_gamma

Final-inference compliant:
    one primary LLM only, no S5 embedder, no RAG, no reranker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.version_runner import add_common_args, run_v02_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v02_beta S4 escalation pipeline")
    add_common_args(
        parser,
        default_output="data/submissions/submission_v02_beta.csv",
        default_trace="data/traces/trace_v02_beta.jsonl",
        include_sc_batch=True,
    )
    args = parser.parse_args()

    run_v02_beta(
        input_path=args.input,
        output_path=args.output,
        trace_output=args.trace_output,
        model_id=args.model_id,
        limit=args.limit,
        safe_mode=args.safe_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        sc_batch_size=args.sc_batch_size,
    )


if __name__ == "__main__":
    main()
