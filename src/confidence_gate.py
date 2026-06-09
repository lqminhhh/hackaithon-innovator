"""Confidence gate — routes questions to the appropriate path based on
the model's self-reported confidence score.

Paths:
  >= 0.85  → fast exit (use first-pass answer directly)
  0.55–0.84 → adaptive consistency sampling
  < 0.55   → dual-model ensemble
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"

Route = Literal["fast_exit", "consistency", "ensemble"]


def _load_thresholds() -> dict:
    with open(_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["confidence_gate"]


_THRESHOLDS: dict | None = None


def route(confidence: float) -> Route:
    """Determine which downstream path to take based on confidence."""
    global _THRESHOLDS
    if _THRESHOLDS is None:
        _THRESHOLDS = _load_thresholds()

    if confidence >= _THRESHOLDS["fast_exit_threshold"]:
        return "fast_exit"
    if confidence >= _THRESHOLDS["ensemble_threshold"]:
        return "consistency"
    return "ensemble"
