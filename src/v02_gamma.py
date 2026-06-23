"""Compatibility shim for the historical `v02_gamma` entrypoint.

The final runner was promoted to `src.v03_gamma`. Keep this module so older
scripts, docs, and evaluation flows can still import or execute `v02_gamma`
without breaking immediately.
"""

from __future__ import annotations

from src.v03_gamma import main as _v03_main
from src.v03_gamma import run_v03_gamma


def run_v02_gamma(
    *,
    input_path: str,
    output_path: str,
    trace_output: str,
    model_id: str | None = None,
    limit: int | None = None,
    safe_mode: bool = False,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    adaptive_sc: bool = True,
) -> None:
    """Compatibility wrapper for callers still importing `run_v02_gamma`."""
    return run_v03_gamma(
        input_path=input_path,
        output_path=output_path,
        trace_output=trace_output,
        model_id=model_id,
        limit=limit,
        safe_mode=safe_mode,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        adaptive_sc=adaptive_sc,
    )


def main() -> None:
    _v03_main()


if __name__ == "__main__":
    main()
