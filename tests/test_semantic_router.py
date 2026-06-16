"""Tests for the standalone Layer 2 semantic router scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import ParsedQuestion
from src.semantic_router import SemanticRouter, load_route_prototypes


class _KeywordEmbedder:
    def __init__(self):
        self.axes = {
            "reading": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "stem": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "knowledge": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "safety": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vec = np.zeros(4, dtype=np.float32)
            if any(term in lowered for term in ("đoạn", "văn bản", "context")):
                vec += self.axes["reading"]
            if any(term in lowered for term in ("tính", "phương trình", "dòng điện", "xác suất")):
                vec += self.axes["stem"]
            if any(term in lowered for term in ("quy định", "căn cước", "gdp", "kiến thức", "legal")):
                vec += self.axes["knowledge"]
            if any(term in lowered for term in ("xâm nhập", "chế tạo bom", "từ chối", "gây hại")):
                vec += self.axes["safety"]
            if not vec.any():
                vec += self.axes["knowledge"]
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.stack(vectors)


def _parsed(
    query: str,
    *,
    context: str | None = None,
    is_quantitative: bool = False,
    is_legal: bool = False,
    has_refusal_choice: bool = False,
    is_harmful: bool = False,
    n_choices: int = 4,
) -> ParsedQuestion:
    options = {chr(65 + i): f"option {i}" for i in range(n_choices)}
    return ParsedQuestion(
        qid="demo",
        original_question=query,
        query=query,
        context=context,
        options=options,
        refusal_labels=("C",) if has_refusal_choice else (),
        n_choices=n_choices,
        has_context=context is not None,
        is_quantitative=is_quantitative,
        is_legal=is_legal,
        has_refusal_choice=has_refusal_choice,
        is_harmful=is_harmful,
    )


class TestPrototypeLoading:
    def test_load_route_prototypes_has_all_routes(self):
        prototypes = load_route_prototypes()
        routes = {p.route for p in prototypes}
        assert routes == {"reading", "stem", "knowledge", "safety"}


class TestSemanticRouter:
    def test_scores_reading_highest_for_context_question(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        parsed = _parsed(
            "Theo đoạn thông tin trên, tác giả muốn nhấn mạnh điều gì?",
            context="Đoạn thông tin: ...",
        )

        scores = router.score_routes(parsed)

        assert max(scores, key=scores.get) == "reading"

    def test_recommends_override_for_ambiguous_legal_question(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        parsed = _parsed(
            "Theo quy định hiện hành, cơ quan nào cấp căn cước công dân?",
            is_legal=True,
            n_choices=4,
        )

        result = router.decide_route(parsed, layer1_route="stem")

        assert result.layer2_route == "knowledge"
        assert result.was_ambiguous is True
        assert result.should_override is True
        assert result.final_route == "knowledge"
