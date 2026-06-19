"""Shadow-mode diagnostics for the Layer-2 semantic router.

This module never changes the answer path. It records what the semantic router
would have recommended so we can inspect disagreements before enabling any
active override policy.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

from src.parser import parse_question
from src.router import route_l1, route_question
from src.semantic_router import SemanticRouter


def build_shadow_records(
    questions: Iterable[dict],
    *,
    router: SemanticRouter | None = None,
) -> list[dict]:
    """Return JSON-serializable semantic-router shadow records."""
    semantic_router = router or SemanticRouter()
    records: list[dict] = []

    for question in questions:
        parsed = parse_question(question)
        layer1_special = route_l1(parsed)
        layer1_final = route_question(parsed)
        result = semantic_router.decide_route(parsed, layer1_route=layer1_final)

        records.append(
            {
                "qid": parsed.qid,
                "layer1_special_route": layer1_special,
                "layer1_final_route": layer1_final,
                "semantic_route": result.layer2_route,
                "final_route_if_enabled": result.final_route,
                "would_change_route": result.final_route != layer1_final,
                "should_override": result.should_override,
                "was_ambiguous": result.was_ambiguous,
                "route_scores": result.route_scores,
                "score_margin": result.score_margin,
                "top_gap": result.top_gap,
                "override_blockers": list(result.override_blockers),
                "n_choices": parsed.n_choices,
                "has_context": parsed.has_context,
                "is_quantitative": parsed.is_quantitative,
                "is_legal": parsed.is_legal,
                "has_refusal_choice": parsed.has_refusal_choice,
                "is_harmful": parsed.is_harmful,
                "query": parsed.query,
                "options": parsed.options,
            }
        )

    return records


def summarize_shadow_records(records: Iterable[dict]) -> dict:
    """Build compact counts for CLI/notebook inspection."""
    rows = list(records)
    layer1_counts = Counter(row["layer1_final_route"] for row in rows)
    semantic_counts = Counter(row["semantic_route"] for row in rows)
    pair_counts = Counter(
        (row["layer1_final_route"], row["semantic_route"]) for row in rows
    )
    override_counts = Counter(row["should_override"] for row in rows)

    return {
        "total": len(rows),
        "layer1_counts": dict(layer1_counts),
        "semantic_counts": dict(semantic_counts),
        "layer1_semantic_pairs": {
            f"{layer1}->{semantic}": count
            for (layer1, semantic), count in sorted(pair_counts.items())
        },
        "should_override_counts": {
            str(key): value for key, value in override_counts.items()
        },
        "would_change_count": sum(row["would_change_route"] for row in rows),
        "should_override_count": sum(row["should_override"] for row in rows),
    }


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write shadow records as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
