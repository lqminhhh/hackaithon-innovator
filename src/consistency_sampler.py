"""Adaptive consistency sampler.

Samples the model N times at higher temperature and takes the majority
vote.  Uses an adaptive strategy: start with N=2; if they disagree,
escalate to N=5; if still no majority, go to N=7.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from src.reasoning_agent import ReasoningAgent
from src.normaliser import normalise_answer, parse_confidence

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _majority(answers: list[str]) -> str | None:
    """Return the majority answer, or None if tied."""
    if not answers:
        return None
    count = Counter(answers)
    top = count.most_common(2)
    if len(top) == 1:
        return top[0][0]
    if top[0][1] > top[1][1]:
        return top[0][0]
    return None


def adaptive_consistency(
    agent: ReasoningAgent,
    question: str,
    options: dict[str, str],
    context: str | None = None,
) -> tuple[str, float]:
    """Run adaptive N=2→5→7 consistency sampling.

    Returns (answer_letter, confidence) where confidence = winning_votes / total.
    """
    cfg = _load_config()
    inf_cfg = cfg["inference"]
    sc_cfg = cfg["consistency_sampler"]
    temp = inf_cfg["temperature_sampling"]

    answers: list[str] = []

    def _sample() -> str:
        if context:
            raw = agent.infer_with_context(question, options, context, temperature=temp)
        else:
            raw = agent.infer_no_context(question, options, temperature=temp)
        return normalise_answer(raw)

    # Phase 1: N=2
    for _ in range(sc_cfg["n_initial"]):
        answers.append(_sample())
    winner = _majority(answers)
    if winner is not None:
        return winner, len([a for a in answers if a == winner]) / len(answers)

    # Phase 2: escalate to N=5
    for _ in range(sc_cfg["n_second"] - sc_cfg["n_initial"]):
        answers.append(_sample())
    winner = _majority(answers)
    if winner is not None:
        return winner, len([a for a in answers if a == winner]) / len(answers)

    # Phase 3: escalate to N=7
    for _ in range(sc_cfg["n_max"] - sc_cfg["n_second"]):
        answers.append(_sample())

    count = Counter(answers)
    best, best_count = count.most_common(1)[0]
    return best, best_count / len(answers)
