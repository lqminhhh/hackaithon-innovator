"""Layer 2 semantic router scaffold based on BGE-M3-style embeddings.

This module is intentionally standalone for early validation. It does not
modify the existing v02_alpha pipeline yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from src.parser import ParsedQuestion
from src.router import Route

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "semantic_router_config.yaml"
_PROTOTYPES_PATH = Path(__file__).resolve().parent.parent / "data" / "route_prototypes.yaml"


@dataclass(frozen=True, slots=True)
class RoutePrototype:
    route: Route
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class SemanticRouterResult:
    layer1_route: Route | None
    layer2_route: Route
    final_route: Route
    route_scores: dict[Route, float]
    should_override: bool
    was_ambiguous: bool
    query_text: str


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_route_prototypes(path: str | Path | None = None) -> list[RoutePrototype]:
    """Load semantic route prototypes from YAML."""
    payload = _load_yaml(Path(path) if path is not None else _PROTOTYPES_PATH)
    routes = payload["routes"]

    prototypes: list[RoutePrototype] = []
    for route_name, route_payload in routes.items():
        route = route_name  # type: ignore[assignment]
        description = route_payload.get("description", "").strip()
        if description:
            prototypes.append(
                RoutePrototype(route=route, text=description, source="description")
            )
        for example in route_payload.get("examples", []):
            text = str(example).strip()
            if text:
                prototypes.append(RoutePrototype(route=route, text=text, source="example"))
    return prototypes


class SemanticRouter:
    """Standalone semantic route scorer built on sentence embeddings."""

    def __init__(
        self,
        embedder=None,
        model_name: str | None = None,
        config_path: str | Path | None = None,
        prototypes_path: str | Path | None = None,
    ):
        cfg = _load_yaml(Path(config_path) if config_path is not None else _CFG_PATH)
        self.cfg = cfg
        self.model_name = model_name or cfg["model"]["name"]
        self.top_k_per_route = int(cfg["scoring"]["top_k_per_route"])
        self.min_top_score = float(cfg["scoring"]["min_top_score"])
        self.min_override_margin = float(cfg["scoring"]["min_override_margin"])
        self.review_routes = set(cfg["policy"].get("review_routes", []))
        self.allow_override_routes = set(cfg["policy"].get("allow_override_routes", []))
        self.require_ambiguity = bool(cfg["policy"].get("require_ambiguity", True))

        self._embedder = embedder
        self._prototypes = load_route_prototypes(prototypes_path)
        self._prototype_embeddings: np.ndarray | None = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    @property
    def prototypes(self) -> list[RoutePrototype]:
        return self._prototypes

    def build_query_text(self, parsed: ParsedQuestion) -> str:
        """Build route text for semantic embedding.

        We include lightweight metadata markers to help the embedder separate
        question styles without depending on the LLM prompt.
        """
        lines = [
            f"question: {parsed.query}",
            f"choices: {len(parsed.options)}",
        ]
        if parsed.context:
            lines.append(f"context: {parsed.context}")
        if parsed.has_refusal_choice:
            lines.append("has_refusal_choice: yes")
        if parsed.is_legal:
            lines.append("domain: legal")
        if parsed.is_quantitative:
            lines.append("domain: quantitative")
        return "\n".join(lines)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _get_prototype_embeddings(self) -> np.ndarray:
        if self._prototype_embeddings is None:
            self._prototype_embeddings = self.encode_texts(p.text for p in self.prototypes)
        return self._prototype_embeddings

    def score_routes(self, parsed: ParsedQuestion) -> dict[Route, float]:
        """Return one semantic score per route."""
        query_text = self.build_query_text(parsed)
        query_embedding = self.encode_texts([query_text])[0]
        prototype_embeddings = self._get_prototype_embeddings()
        similarities = prototype_embeddings @ query_embedding

        grouped: dict[Route, list[float]] = {
            "reading": [],
            "stem": [],
            "knowledge": [],
            "safety": [],
        }
        for prototype, sim in zip(self.prototypes, similarities):
            grouped[prototype.route].append(float(sim))

        scores: dict[Route, float] = {}
        for route, values in grouped.items():
            if not values:
                scores[route] = float("-inf")
                continue
            top_values = sorted(values, reverse=True)[: self.top_k_per_route]
            scores[route] = float(sum(top_values) / len(top_values))
        return scores

    def is_ambiguous(self, parsed: ParsedQuestion, layer1_route: Route) -> bool:
        """Cheap ambiguity heuristic for deciding when to review Layer 1."""
        if parsed.has_refusal_choice and not parsed.is_harmful:
            return True
        if parsed.is_legal and layer1_route in {"knowledge", "stem"}:
            return True
        if parsed.n_choices >= 8 and not parsed.has_context:
            return True
        if layer1_route in {"knowledge", "stem"} and not parsed.has_context:
            return True
        return False

    def decide_route(
        self,
        parsed: ParsedQuestion,
        layer1_route: Route | None = None,
    ) -> SemanticRouterResult:
        """Return semantic routing result and an override recommendation."""
        route_scores = self.score_routes(parsed)
        layer2_route = max(route_scores, key=route_scores.get)
        query_text = self.build_query_text(parsed)

        if layer1_route is None:
            return SemanticRouterResult(
                layer1_route=None,
                layer2_route=layer2_route,
                final_route=layer2_route,
                route_scores=route_scores,
                should_override=False,
                was_ambiguous=False,
                query_text=query_text,
            )

        ambiguous = self.is_ambiguous(parsed, layer1_route)
        top_score = route_scores[layer2_route]
        layer1_score = route_scores.get(layer1_route, float("-inf"))
        score_margin = top_score - layer1_score

        eligible_for_review = layer1_route in self.review_routes
        eligible_for_override = layer1_route in self.allow_override_routes

        should_override = (
            layer2_route != layer1_route
            and eligible_for_review
            and eligible_for_override
            and top_score >= self.min_top_score
            and score_margin >= self.min_override_margin
            and (ambiguous or not self.require_ambiguity)
        )
        final_route = layer2_route if should_override else layer1_route

        return SemanticRouterResult(
            layer1_route=layer1_route,
            layer2_route=layer2_route,
            final_route=final_route,
            route_scores=route_scores,
            should_override=should_override,
            was_ambiguous=ambiguous,
            query_text=query_text,
        )
