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
    score_margin: float
    top_gap: float
    override_blockers: tuple[str, ...]


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
        device: str | None = None,
        config_path: str | Path | None = None,
        prototypes_path: str | Path | None = None,
    ):
        cfg = _load_yaml(Path(config_path) if config_path is not None else _CFG_PATH)
        self.cfg = cfg
        self.model_name = model_name or cfg["model"]["name"]
        self.device = device if device is not None else cfg["model"].get("device")
        self.max_context_chars = int(cfg["model"].get("max_context_chars", 1200))
        self.top_k_per_route = int(cfg["scoring"]["top_k_per_route"])
        self.min_top_score = float(cfg["scoring"]["min_top_score"])
        self.min_override_margin = float(cfg["scoring"]["min_override_margin"])
        self.min_top_gap = float(cfg["scoring"].get("min_top_gap", 0.0))
        self.review_routes = set(cfg["policy"].get("review_routes", []))
        self.allow_override_routes = set(cfg["policy"].get("allow_override_routes", []))
        self.allow_override_targets = set(
            cfg["policy"].get("allow_override_targets", self.allow_override_routes)
        )
        self.require_ambiguity = bool(cfg["policy"].get("require_ambiguity", True))
        self.reading_requires_context = bool(
            cfg["policy"].get("reading_requires_context", True)
        )
        self.safety_requires_harmful_refusal = bool(
            cfg["policy"].get("safety_requires_harmful_refusal", True)
        )
        self.stem_requires_evidence = bool(
            cfg["policy"].get("stem_requires_evidence", True)
        )

        self._embedder = embedder
        self._prototypes = load_route_prototypes(prototypes_path)
        self._prototype_embeddings: np.ndarray | None = None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "S5 semantic routing requires sentence-transformers. "
                    "Install project requirements or run: "
                    "`pip install 'sentence-transformers>=3.3.1,<4.0.0'`."
                ) from exc

            kwargs = {"device": self.device} if self.device else {}
            self._embedder = SentenceTransformer(self.model_name, **kwargs)
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
            context = parsed.context[: self.max_context_chars]
            lines.append(f"context: {context}")
        if parsed.has_refusal_choice:
            lines.append("abstention_option_present: yes")
        lines.append(f"passage_present: {'yes' if parsed.has_context else 'no'}")
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

    def warmup(self) -> None:
        """Load the embedder and cache prototype embeddings."""
        self._get_prototype_embeddings()

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
                score_margin=0.0,
                top_gap=_top_gap(route_scores),
                override_blockers=("no_layer1_route",),
            )

        ambiguous = self.is_ambiguous(parsed, layer1_route)
        top_score = route_scores[layer2_route]
        layer1_score = route_scores.get(layer1_route, float("-inf"))
        score_margin = top_score - layer1_score
        top_gap = _top_gap(route_scores)

        blockers = self._override_blockers(
            parsed=parsed,
            layer1_route=layer1_route,
            layer2_route=layer2_route,
            top_score=top_score,
            score_margin=score_margin,
            top_gap=top_gap,
            ambiguous=ambiguous,
        )

        should_override = not blockers
        final_route = layer2_route if should_override else layer1_route

        return SemanticRouterResult(
            layer1_route=layer1_route,
            layer2_route=layer2_route,
            final_route=final_route,
            route_scores=route_scores,
            should_override=should_override,
            was_ambiguous=ambiguous,
            query_text=query_text,
            score_margin=score_margin,
            top_gap=top_gap,
            override_blockers=tuple(blockers),
        )

    def _override_blockers(
        self,
        *,
        parsed: ParsedQuestion,
        layer1_route: Route,
        layer2_route: Route,
        top_score: float,
        score_margin: float,
        top_gap: float,
        ambiguous: bool,
    ) -> list[str]:
        blockers: list[str] = []

        if layer2_route == layer1_route:
            blockers.append("same_route")
        if layer1_route not in self.review_routes:
            blockers.append("source_not_reviewed")
        if layer1_route not in self.allow_override_routes:
            blockers.append("source_not_allowed")
        if layer2_route not in self.allow_override_targets:
            blockers.append("target_not_allowed")
        if top_score < self.min_top_score:
            blockers.append("top_score_below_floor")
        if score_margin < self.min_override_margin:
            blockers.append("margin_below_floor")
        if top_gap < self.min_top_gap:
            blockers.append("top_gap_below_floor")
        if self.require_ambiguity and not ambiguous:
            blockers.append("not_ambiguous")
        blockers.extend(self._target_blockers(parsed, layer2_route))
        return blockers

    def _target_blockers(self, parsed: ParsedQuestion, target: Route) -> list[str]:
        if target == "reading" and self.reading_requires_context and not parsed.has_context:
            return ["reading_without_context"]
        if (
            target == "safety"
            and self.safety_requires_harmful_refusal
            and not _has_safety_evidence(parsed)
        ):
            return ["safety_without_harmful_refusal"]
        if target == "stem" and self.stem_requires_evidence and not _has_stem_evidence(parsed):
            return ["stem_without_evidence"]
        return []


def _top_gap(scores: dict[Route, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


_STEM_EVIDENCE_TERMS = (
    "lãi kép",
    "lãi suất",
    "số tiền",
    "dòng điện",
    "điện cảm ứng",
    "đinamô",
    "độ co giãn",
    "độ co dãn",
    "co giãn",
    "co dãn",
    "ngoại tệ",
    "đường cầu",
    "đường cung",
    "chi phí",
    "doanh thu",
    "lợi nhuận",
    "độ thỏa dụng",
    "thỏa dụng biên",
    "phương trình",
    "xác suất",
    "đạo hàm",
    "tích phân",
    "điện áp",
    "điện trở",
    "công suất",
    "nồng độ",
)

_SAFETY_EVIDENCE_TERMS = (
    "phát tán tài liệu mật",
    "tài liệu mật",
    "tham nhũng",
    "lợi dụng quyền lực",
    "lợi dụng quyền hạn",
    "chống phá",
    "chống lại đường lối",
    "vi phạm lệnh đình chỉ",
    "bị đình chỉ vĩnh viễn",
    "xâm nhập",
    "đánh cắp",
    "trộm",
    "lừa đảo",
    "vượt mặt",
    "qua mặt",
    "tấn công",
    "chế tạo bom",
    "ma túy",
    "vũ khí",
    "phá hoại",
)


def _question_with_options(parsed: ParsedQuestion) -> str:
    return f"{parsed.query}\n" + "\n".join(parsed.options.values())


def _has_stem_evidence(parsed: ParsedQuestion) -> bool:
    if parsed.is_quantitative:
        return True
    lowered = _question_with_options(parsed).lower()
    return any(term in lowered for term in _STEM_EVIDENCE_TERMS)


def _has_safety_evidence(parsed: ParsedQuestion) -> bool:
    if not parsed.has_refusal_choice:
        return False
    if parsed.is_harmful:
        return True
    lowered = _question_with_options(parsed).lower()
    return any(term in lowered for term in _SAFETY_EVIDENCE_TERMS)
