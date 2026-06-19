"""Tests for the standalone Layer 2 semantic router scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import ParsedQuestion
from src.semantic_router import SemanticRouter, load_route_prototypes
from src.semantic_shadow import build_shadow_records, summarize_shadow_records


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

    def test_blocks_reading_override_without_context(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        parsed = _parsed(
            "Theo đoạn mô tả trong đề, khái niệm nào đúng nhất?",
            is_legal=True,
        )

        result = router.decide_route(parsed, layer1_route="knowledge")

        assert result.layer2_route == "reading"
        assert result.should_override is False
        assert result.final_route == "knowledge"
        assert "reading_without_context" in result.override_blockers

    def test_blocks_safety_override_without_harmful_refusal(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        parsed = _parsed(
            "Câu tục ngữ nào đúng nhất? Một lựa chọn là từ chối.",
            has_refusal_choice=True,
            is_harmful=False,
        )

        result = router.decide_route(parsed, layer1_route="knowledge")

        assert result.layer2_route == "safety"
        assert result.should_override is False
        assert result.final_route == "knowledge"
        assert "safety_without_harmful_refusal" in result.override_blockers

    def test_blocks_stem_override_without_stem_evidence(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        parsed = _parsed(
            "Tính cách nào phù hợp nhất trong giao tiếp?",
            is_legal=True,
        )

        result = router.decide_route(parsed, layer1_route="knowledge")

        assert result.layer2_route == "stem"
        assert result.should_override is False
        assert result.final_route == "knowledge"
        assert "stem_without_evidence" in result.override_blockers

    def test_query_text_truncates_long_context(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        router.max_context_chars = 12
        parsed = _parsed(
            "Theo đoạn thông tin, ý chính là gì?",
            context="Đoạn thông tin: " + ("x" * 100),
        )

        query_text = router.build_query_text(parsed)

        assert "x" * 20 not in query_text
        assert "context: Đoạn thông t" in query_text


class TestSemanticShadow:
    def test_build_shadow_records_does_not_change_active_route(self):
        router = SemanticRouter(embedder=_KeywordEmbedder())
        questions = [
            {
                "qid": "q1",
                "question": "Theo quy định hiện hành, cơ quan nào cấp căn cước công dân?",
                "options": {"A": "Một", "B": "Hai"},
            }
        ]

        records = build_shadow_records(questions, router=router)

        assert len(records) == 1
        assert records[0]["qid"] == "q1"
        assert records[0]["layer1_final_route"] == "knowledge"
        assert records[0]["semantic_route"] == "knowledge"
        assert records[0]["final_route_if_enabled"] == "knowledge"
        assert records[0]["should_override"] is False
        assert set(records[0]["route_scores"]) == {"reading", "stem", "knowledge", "safety"}
        assert "score_margin" in records[0]
        assert "top_gap" in records[0]
        assert "override_blockers" in records[0]

    def test_shadow_summary_counts_disagreements(self):
        rows = [
            {
                "layer1_final_route": "knowledge",
                "semantic_route": "stem",
                "would_change_route": True,
                "should_override": True,
            },
            {
                "layer1_final_route": "reading",
                "semantic_route": "reading",
                "would_change_route": False,
                "should_override": False,
            },
        ]

        summary = summarize_shadow_records(rows)

        assert summary["total"] == 2
        assert summary["layer1_counts"] == {"knowledge": 1, "reading": 1}
        assert summary["semantic_counts"] == {"stem": 1, "reading": 1}
        assert summary["would_change_count"] == 1
        assert summary["should_override_count"] == 1
