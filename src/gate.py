"""Confidence gate — decides whether to accept tier-1 or escalate to tier-2.

Two signals:
  1. Generative margin:  logprob(top-1 letter) − logprob(top-2 letter)
     Margin ≥ τ → accept tier-1.  Margin < τ → escalate.

  2. Discriminative cross-check (only for questions with ≥10 choices):
     Score each letter via a single forced-token pass; if the
     discriminative winner disagrees with the generative winner,
     escalate regardless of the margin.

τ is read from configs/pipeline_config.yaml → gate.tau (default 1.5).
Set conservatively — over-escalation is cheaper than wrong-but-confident.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from src.inference import InferenceResult

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
_DEFAULT_TAU = 1.5
_DISC_CHOICE_THRESHOLD = 10   # use discriminative cross-check for n_choices ≥ this


def _load_tau() -> float:
    try:
        with open(_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("gate", {}).get("tau", _DEFAULT_TAU))
    except Exception:
        return _DEFAULT_TAU


def compute_margin(result: InferenceResult) -> float:
    """Return top1 − top2 logprob margin from an InferenceResult."""
    return result.margin


def should_escalate(
    result: InferenceResult,
    n_choices: int,
    disc_result: InferenceResult | None = None,
    tau: float | None = None,
) -> tuple[bool, str]:
    """Return (escalate: bool, reason: str).

    Parameters
    ----------
    result:       tier-1 generative InferenceResult
    n_choices:    number of answer choices for this question
    disc_result:  discriminative forward-pass result (required when n_choices ≥ 10)
    tau:          override τ (uses config value if None)
    """
    if tau is None:
        tau = _load_tau()

    margin = compute_margin(result)

    if margin < tau:
        return True, f"margin={margin:.3f} < tau={tau:.3f}"

    # Discriminative cross-check for 10+-choice questions
    if n_choices >= _DISC_CHOICE_THRESHOLD and disc_result is not None:
        if disc_result.letter != result.letter:
            return (
                True,
                f"disc_check disagreement: gen={result.letter} disc={disc_result.letter}",
            )

    return False, f"margin={margin:.3f} >= tau={tau:.3f}"


def escalation_rate(
    results: list[InferenceResult],
    n_choices_list: list[int],
    disc_results: list[InferenceResult | None] | None = None,
    tau: float | None = None,
) -> tuple[list[bool], float]:
    """Compute per-question escalation decisions and the overall escalation rate.

    Returns (escalate_flags, rate).
    """
    if disc_results is None:
        disc_results = [None] * len(results)

    flags: list[bool] = []
    for result, n, disc in zip(results, n_choices_list, disc_results):
        escalate, _ = should_escalate(result, n, disc, tau)
        flags.append(escalate)

    rate = sum(flags) / len(flags) if flags else 0.0
    return flags, rate
