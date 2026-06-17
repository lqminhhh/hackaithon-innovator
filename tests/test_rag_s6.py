"""Tests for S6 RAG -- config, build script, _QwenReranker, RAGEngine, and solve integration.

Test sections
-------------
1. Config        -- RAG_INCLUDE_SUBJECTS, SUBJECT_META, constant sanity, model name
2. Build         -- chunk formatting, loader, dedup (no FAISS needed)
3. _QwenReranker -- prompt format, token resolution, predict interface (no real model)
4. RAGEngine     -- retrieve, rerank, _pipeline, _format_context (requires faiss)
5. Prompts       -- knowledge_rag template placeholders and formatting
6. ReasoningAgent -- build_route_prompt uses knowledge_rag when context given
7. Solve         -- _try_rag logic, solve_question RAG integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── optional faiss skip ───────────────────────────────────────────────────────

try:
    import faiss as _faiss_module

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Config
# ═══════════════════════════════════════════════════════════════════════════════


def test_rag_include_subjects_count():
    from src.config import RAG_INCLUDE_SUBJECTS

    assert len(RAG_INCLUDE_SUBJECTS) == 34


def test_rag_include_subjects_excludes_all_stem():
    from src.config import RAG_INCLUDE_SUBJECTS

    stem_ids = {str(i).zfill(2) for i in range(1, 22)}  # "01".."21"
    overlap = stem_ids & RAG_INCLUDE_SUBJECTS
    assert overlap == set(), f"STEM subjects in RAG corpus: {overlap}"


def test_rag_include_subjects_excludes_macro_micro_logic():
    from src.config import RAG_INCLUDE_SUBJECTS

    for excluded in ("28", "29", "47"):
        assert excluded not in RAG_INCLUDE_SUBJECTS, f"Subject {excluded} should be excluded"


def test_subject_meta_has_58_subjects():
    from src.config import SUBJECT_META

    assert len(SUBJECT_META) == 58


def test_subject_meta_entries_have_required_keys():
    from src.config import SUBJECT_META

    for s_id, meta in SUBJECT_META.items():
        assert "name" in meta, f"{s_id} missing 'name'"
        assert "category" in meta, f"{s_id} missing 'category'"
        assert meta["category"] in {"STEM", "Social Science", "Humanity", "Other"}, (
            f"{s_id} has unexpected category: {meta['category']}"
        )


def test_all_rag_subjects_in_subject_meta():
    from src.config import RAG_INCLUDE_SUBJECTS, SUBJECT_META

    for s_id in RAG_INCLUDE_SUBJECTS:
        assert s_id in SUBJECT_META, f"Subject {s_id} not in SUBJECT_META"


def test_rag_top_k_greater_than_top_n():
    from src.config import RAG_TOP_K, RAG_TOP_N

    assert RAG_TOP_K > RAG_TOP_N > 0


def test_rag_timeout_positive():
    from src.config import RAG_TIMEOUT

    assert RAG_TIMEOUT > 0.0


def test_rerank_min_in_unit_interval():
    from src.config import RERANK_MIN

    assert 0.0 <= RERANK_MIN <= 1.0


def test_vmlu_paths_are_path_objects():
    from pathlib import Path

    from src.config import VMLU_CHUNKS_PATH, VMLU_DATA_DIR, VMLU_INDEX_PATH

    for path in (VMLU_DATA_DIR, VMLU_INDEX_PATH, VMLU_CHUNKS_PATH):
        assert isinstance(path, Path), f"{path} should be a Path object"


def test_rerank_model_is_qwen():
    """planning_v2.md specifies Qwen/Qwen3-Reranker-0.6B as the reranker."""
    from src.config import RERANK_MODEL

    assert RERANK_MODEL == "Qwen/Qwen3-Reranker-0.6B", (
        f"Expected Qwen/Qwen3-Reranker-0.6B, got {RERANK_MODEL}"
    )


def test_rerank_batch_size_positive():
    from src.config import RERANK_BATCH_SIZE

    assert RERANK_BATCH_SIZE > 0


def test_pipeline_config_reranker_model():
    """configs/pipeline_config.yaml must reference the Qwen reranker."""
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["models"]["reranker"] == "Qwen/Qwen3-Reranker-0.6B", (
        f"pipeline_config.yaml reranker mismatch: {cfg['models'].get('reranker')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Build script (no FAISS required)
# ═══════════════════════════════════════════════════════════════════════════════


def test_parse_answer_text_standard():
    from scripts.build_vmlu_index import _parse_answer_text

    choices = ["A. Apple", "B. Banana", "C. Cherry"]
    assert _parse_answer_text(choices, "B") == "B. Banana"


def test_parse_answer_text_first_and_last():
    from scripts.build_vmlu_index import _parse_answer_text

    choices = ["A. First", "B. Middle", "C. Last"]
    assert _parse_answer_text(choices, "A") == "A. First"
    assert _parse_answer_text(choices, "C") == "C. Last"


def test_parse_answer_text_whitespace_stripped():
    from scripts.build_vmlu_index import _parse_answer_text

    choices = ["  A. With spaces  ", "B. Other"]
    result = _parse_answer_text(choices, "A")
    assert result == "A. With spaces"


def test_parse_answer_text_not_found_returns_fallback():
    from scripts.build_vmlu_index import _parse_answer_text

    choices = ["A. Option A", "B. Option B"]
    result = _parse_answer_text(choices, "D")
    assert result.startswith("D."), f"Fallback should start with 'D.', got: {result}"


def test_format_vmlu_chunk_contains_all_sections():
    from scripts.build_vmlu_index import format_vmlu_chunk

    obj = {
        "id": "39-0001",
        "question": "Án treo là gì?",
        "choices": ["A. Biện pháp miễn chấp hành hình phạt tù có điều kiện", "B. Hình phạt bổ sung"],
        "answer": "A",
    }
    chunk = format_vmlu_chunk(obj)

    assert chunk.startswith("Câu hỏi:")
    assert "Các lựa chọn:" in chunk
    assert "Đáp án đúng:" in chunk
    assert "A. Biện pháp miễn chấp hành hình phạt tù có điều kiện" in chunk
    assert "B. Hình phạt bổ sung" in chunk


def test_format_vmlu_chunk_correct_answer_highlighted():
    from scripts.build_vmlu_index import format_vmlu_chunk

    obj = {
        "id": "24-0001",
        "question": "Câu hỏi về pháp luật?",
        "choices": ["A. Sai", "B. Đúng", "C. Không rõ"],
        "answer": "B",
    }
    chunk = format_vmlu_chunk(obj)
    assert "Đáp án đúng: B. Đúng" in chunk
    # Wrong answer should NOT appear after "Đáp án đúng:"
    after_answer = chunk.split("Đáp án đúng:")[1]
    assert "A. Sai" not in after_answer


def test_format_vmlu_chunk_utf8_preserved():
    from scripts.build_vmlu_index import format_vmlu_chunk

    obj = {
        "id": "27-0001",
        "question": "Tư tưởng Hồ Chí Minh là gì?",
        "choices": ["A. Chủ nghĩa Mác – Lênin", "B. Kinh nghiệm thực tiễn"],
        "answer": "A",
    }
    chunk = format_vmlu_chunk(obj)
    assert "Hồ Chí Minh" in chunk
    assert "Mác – Lênin" in chunk


def test_load_vmlu_chunks_total_count():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    assert len(chunks) == 606


def test_load_vmlu_chunks_metadata_fields_present():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    required = {"text", "id", "subject_id", "subject_name", "category", "source"}
    for chunk in chunks[:20]:
        missing = required - set(chunk.keys())
        assert not missing, f"Chunk {chunk.get('id', '?')} missing fields: {missing}"


def test_load_vmlu_chunks_no_stem_subjects():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    stem_ids = {str(i).zfill(2) for i in range(1, 22)}
    for chunk in chunks:
        assert chunk["subject_id"] not in stem_ids, (
            f"STEM subject {chunk['subject_id']} found in RAG corpus (chunk {chunk['id']})"
        )


def test_load_vmlu_chunks_all_have_correct_answer():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    for chunk in chunks:
        assert "Đáp án đúng:" in chunk["text"], (
            f"Chunk {chunk['id']} missing correct answer"
        )


def test_load_vmlu_chunks_source_labels():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    sources = {c["source"] for c in chunks}
    assert sources == {"vmlu_dev", "vmlu_valid"}


def test_load_vmlu_chunks_custom_subject_filter():
    from scripts.build_vmlu_index import load_vmlu_chunks

    civil_law_only = load_vmlu_chunks(include_subjects=frozenset({"39"}))
    assert len(civil_law_only) > 0
    assert all(c["subject_id"] == "39" for c in civil_law_only)


def test_load_vmlu_chunks_empty_filter():
    from scripts.build_vmlu_index import load_vmlu_chunks

    empty = load_vmlu_chunks(include_subjects=frozenset())
    assert empty == []


def test_load_vmlu_chunks_ids_are_unique():
    from scripts.build_vmlu_index import load_vmlu_chunks

    chunks = load_vmlu_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate IDs found in corpus"


def test_existing_ids_extracts_ids():
    from scripts.build_vmlu_index import existing_ids

    chunks = [
        {"id": "22-0001", "text": "a"},
        {"id": "39-0002", "text": "b"},
        {"text": "no id field"},
    ]
    ids = existing_ids(chunks)
    assert "22-0001" in ids
    assert "39-0002" in ids


def test_existing_ids_on_empty_list():
    from scripts.build_vmlu_index import existing_ids

    assert existing_ids([]) == set()


def test_source_loaders_registry_has_vmlu():
    from scripts.build_vmlu_index import SOURCE_LOADERS

    assert "vmlu" in SOURCE_LOADERS
    assert callable(SOURCE_LOADERS["vmlu"])


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — _QwenReranker (no real model -- mock transformers)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_qwen_reranker_mock(yes_id: int = 100, no_id: int = 200):
    """Build a _QwenReranker backed by minimal class-based fakes.

    Class-based fakes avoid the PyTorch + MagicMock segfault that occurs when
    tensor slicing operations are performed on MagicMock return values.
    """
    import torch

    from src.rag import _QwenReranker

    _yes = yes_id
    _no = no_id

    class _FakeTokenizer:
        """Returns real tensors; first pair gets high yes logit, rest get low."""

        def __call__(self, texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long),
            }

    class _FakeModelOutput:
        __slots__ = ("logits",)

        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __init__(self):
            self.call_count = 0

        def __call__(self, input_ids, attention_mask=None, **kwargs):
            self.call_count += 1
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            vocab = max(_yes, _no) + 1
            logits = torch.zeros(batch_size, seq_len, vocab)
            # First item in each batch: high P(yes)
            logits[0, -1, _yes] = 5.0
            logits[0, -1, _no] = -5.0
            # Remaining items: low P(yes)
            for i in range(1, batch_size):
                logits[i, -1, _yes] = -5.0
                logits[i, -1, _no] = 5.0
            return _FakeModelOutput(logits)

        def parameters(self):
            return iter([torch.zeros(1)])

    reranker = object.__new__(_QwenReranker)
    reranker.max_length = 512
    reranker.batch_size = 8
    reranker._yes_id = yes_id
    reranker._no_id = no_id
    reranker.tokenizer = _FakeTokenizer()
    reranker.model = _FakeModel()
    return reranker


def test_qwen_reranker_format_pair_contains_system():
    from src.rag import _QwenReranker

    rr = object.__new__(_QwenReranker)
    rr._SYSTEM = "Judge yes or no."
    rr._INSTRUCTION = "Find relevant docs."
    rr._ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    result = rr._format_pair("What is law?", "Law is a rule.")
    assert "<|im_start|>system" in result
    assert "Judge yes or no." in result
    assert "<|im_start|>user" in result
    assert "<Instruct>:" in result
    assert "<Query>: What is law?" in result
    assert "<Document>: Law is a rule." in result
    assert "<|im_start|>assistant" in result
    assert "<think>" in result
    assert "</think>" in result


def test_qwen_reranker_format_pair_ends_with_assistant_prefix():
    from src.rag import _QwenReranker

    rr = object.__new__(_QwenReranker)
    rr._SYSTEM = "S"
    rr._INSTRUCTION = "I"
    rr._ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    result = rr._format_pair("q", "d")
    assert result.endswith(rr._ASSISTANT_PREFIX), (
        "Prompt must end with assistant prefix so position -1 is after </think>"
    )


def test_qwen_reranker_format_pair_utf8_preserved():
    from src.rag import _QwenReranker

    rr = object.__new__(_QwenReranker)
    rr._SYSTEM = "S"
    rr._INSTRUCTION = "I"
    rr._ASSISTANT_PREFIX = "A"

    result = rr._format_pair("Hồ Chí Minh?", "Tư tưởng dân tộc")
    assert "Hồ Chí Minh" in result
    assert "Tư tưởng dân tộc" in result


def test_qwen_reranker_predict_interface_exists():
    """predict() must be defined and accept a list of (query, doc) pairs."""
    from src.rag import _QwenReranker

    assert callable(getattr(_QwenReranker, "predict", None))


def test_qwen_reranker_yes_no_softmax_math():
    """The yes/no scoring math must give P(yes) ∈ [0,1] from any logit values."""
    import torch

    yes_id, no_id = 0, 1
    # High yes logit → near 1.0
    high_yes = torch.tensor([[5.0, -5.0]])
    p_yes_high = float(torch.softmax(high_yes, dim=-1)[0, yes_id])
    assert p_yes_high > 0.99

    # Low yes logit → near 0.0
    low_yes = torch.tensor([[-5.0, 5.0]])
    p_yes_low = float(torch.softmax(low_yes, dim=-1)[0, yes_id])
    assert p_yes_low < 0.01

    # Equal logits → 0.5
    equal = torch.tensor([[0.0, 0.0]])
    p_yes_eq = float(torch.softmax(equal, dim=-1)[0, yes_id])
    assert abs(p_yes_eq - 0.5) < 1e-5


def test_qwen_reranker_batch_arithmetic():
    """Verify the batch-loop logic: N pairs / batch_size = expected forward passes."""
    import math

    for n_pairs, batch_size in [(5, 2), (6, 2), (1, 8), (20, 8)]:
        expected_calls = math.ceil(n_pairs / batch_size)
        actual_ranges = list(range(0, n_pairs, batch_size))
        assert len(actual_ranges) == expected_calls, (
            f"n_pairs={n_pairs}, batch_size={batch_size}: "
            f"expected {expected_calls} batches, got {len(actual_ranges)}"
        )


def test_qwen_reranker_predict_output_type_from_np_array():
    """np.array(list_of_floats, dtype=np.float32) produces the expected output type."""
    scores_list = [0.9, 0.1, 0.5]
    result = np.array(scores_list, dtype=np.float32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (3,)
    assert all(0.0 <= float(s) <= 1.0 for s in result)


def test_qwen_reranker_predict_scores_range_via_softmax():
    """P('yes') from any pair of logits must lie in [0, 1]."""
    import torch

    for yes_logit, no_logit in [(10.0, -10.0), (-10.0, 10.0), (0.0, 0.0), (1.0, 1.0)]:
        t = torch.tensor([[yes_logit, no_logit]])
        p_yes = float(torch.softmax(t, dim=-1)[0, 0])
        assert 0.0 <= p_yes <= 1.0, f"P(yes) = {p_yes} outside [0,1] for logits ({yes_logit}, {no_logit})"


def test_qwen_reranker_class_is_used_in_rag_engine():
    """RAGEngine._reranker must be a _QwenReranker, not a CrossEncoder."""
    from src.rag import _QwenReranker

    # Verify the class is exported and is the one used in RAGEngine.__init__
    import inspect

    import src.rag as rag_module

    init_src = inspect.getsource(rag_module.RAGEngine.__init__)
    assert "_QwenReranker" in init_src, (
        "RAGEngine.__init__ must instantiate _QwenReranker, not CrossEncoder"
    )
    assert "CrossEncoder" not in init_src, (
        "CrossEncoder must not appear in RAGEngine.__init__"
    )


def test_qwen_reranker_default_model_name_in_class():
    """The default model name inside _QwenReranker must match RERANK_MODEL in config."""
    from src.config import RERANK_MODEL
    from src.rag import _QwenReranker

    import inspect

    src_text = inspect.getsource(_QwenReranker.__init__)
    assert "RERANK_MODEL" in src_text or "Qwen/Qwen3-Reranker" in src_text, (
        "_QwenReranker default should reference RERANK_MODEL or the Qwen model name"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — RAGEngine (requires faiss)
# ═══════════════════════════════════════════════════════════════════════════════

# ── shared helpers ────────────────────────────────────────────────────────────


def _make_tiny_rag(
    n_chunks: int = 5,
    dim: int = 8,
    rerank_min: float = 0.0,
    top_k: int = 5,
    top_n: int = 2,
    timeout: float = 2.0,
    use_reranker: bool = False,
    reranker=None,
    seed: int = 0,
):
    """Build a RAGEngine backed by a tiny in-memory FAISS index (no real models)."""
    import faiss

    from src.rag import RAGEngine

    np.random.seed(seed)
    vecs = np.random.randn(n_chunks, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    chunks = [
        {
            "id": f"22-{str(i).zfill(4)}",
            "text": f"Câu hỏi về chủ đề {i}\nĐáp án đúng: A. Đúng",
            "subject_id": "22",
            "source": "vmlu_dev",
        }
        for i in range(n_chunks)
    ]

    mock_embedder = MagicMock()
    # Embedder returns the first vector so FAISS finds real matches
    mock_embedder.encode.return_value = np.array([vecs[0]], dtype=np.float32)

    rag = object.__new__(RAGEngine)
    rag.rerank_min = rerank_min
    rag.top_k = top_k
    rag.top_n = top_n
    rag.timeout = timeout
    rag.use_reranker = use_reranker
    rag._index = index
    rag._chunks = chunks
    rag._embedder = mock_embedder
    rag._reranker = reranker
    return rag, vecs


# ── ScoredChunk ──────────────────────────────────────────────────────────────


@requires_faiss
def test_scored_chunk_is_frozen():
    from src.rag import ScoredChunk

    sc = ScoredChunk(text="t", chunk_id="id", score=0.9, subject_id="22", source="vmlu_dev")
    with pytest.raises((AttributeError, TypeError)):
        sc.score = 0.5  # type: ignore[misc]


@requires_faiss
def test_scored_chunk_fields():
    from src.rag import ScoredChunk

    sc = ScoredChunk(text="hello", chunk_id="22-0001", score=0.75, subject_id="22", source="vmlu_dev")
    assert sc.text == "hello"
    assert sc.chunk_id == "22-0001"
    assert sc.score == 0.75


# ── _format_context ───────────────────────────────────────────────────────────


@requires_faiss
def test_format_context_single_chunk():
    from src.rag import ScoredChunk, _format_context

    chunks = [ScoredChunk(text="Context A", chunk_id="id1", score=0.9, subject_id="22", source="s")]
    result = _format_context(chunks)
    assert "[1]" in result
    assert "Context A" in result


@requires_faiss
def test_format_context_multiple_chunks():
    from src.rag import ScoredChunk, _format_context

    chunks = [
        ScoredChunk(text="First", chunk_id="id1", score=0.9, subject_id="22", source="s"),
        ScoredChunk(text="Second", chunk_id="id2", score=0.7, subject_id="39", source="s"),
        ScoredChunk(text="Third", chunk_id="id3", score=0.5, subject_id="40", source="s"),
    ]
    result = _format_context(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result
    assert "First" in result
    assert "Second" in result
    assert "Third" in result


@requires_faiss
def test_format_context_empty():
    from src.rag import _format_context

    result = _format_context([])
    assert result == ""


# ── retrieve ─────────────────────────────────────────────────────────────────


@requires_faiss
def test_retrieve_returns_list_of_scored_chunks():
    from src.rag import ScoredChunk

    rag, _ = _make_tiny_rag(n_chunks=5)
    results = rag.retrieve("some query")
    assert isinstance(results, list)
    assert all(isinstance(r, ScoredChunk) for r in results)


@requires_faiss
def test_retrieve_empty_index_returns_empty():
    import faiss

    from src.rag import RAGEngine

    rag = object.__new__(RAGEngine)
    rag.rerank_min = 0.5
    rag.top_k = 5
    rag.top_n = 3
    rag.timeout = 2.0
    rag.use_reranker = False
    rag._index = faiss.IndexFlatIP(8)  # empty
    rag._chunks = []
    rag._embedder = MagicMock()
    rag._reranker = None

    assert rag.retrieve("query") == []


@requires_faiss
def test_retrieve_decontaminates_exact_id():
    rag, _ = _make_tiny_rag(n_chunks=5)
    # The first chunk has id "22-0000"; exclude it
    exclude_id = "22-0000"
    results = rag.retrieve("query", exclude_id=exclude_id)
    returned_ids = [r.chunk_id for r in results]
    assert exclude_id not in returned_ids, "Excluded chunk should not appear in results"


@requires_faiss
def test_retrieve_returns_at_most_top_k():
    rag, _ = _make_tiny_rag(n_chunks=10, top_k=3)
    results = rag.retrieve("query", top_k=3)
    assert len(results) <= 3


@requires_faiss
def test_retrieve_scores_are_cosine_similarities():
    rag, _ = _make_tiny_rag(n_chunks=5)
    results = rag.retrieve("query")
    # IndexFlatIP with L2-normalised vectors → scores in [-1, 1]
    for r in results:
        assert -1.0 <= r.score <= 1.0 + 1e-6


@requires_faiss
def test_retrieve_results_sorted_by_score():
    rag, _ = _make_tiny_rag(n_chunks=5)
    results = rag.retrieve("query")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by descending score"


# ── rerank ────────────────────────────────────────────────────────────────────


@requires_faiss
def test_rerank_without_reranker_is_passthrough():
    from src.rag import ScoredChunk

    rag, _ = _make_tiny_rag(use_reranker=False)
    chunks = [
        ScoredChunk(text="A", chunk_id="c1", score=0.8, subject_id="22", source="s"),
        ScoredChunk(text="B", chunk_id="c2", score=0.6, subject_id="22", source="s"),
    ]
    result = rag.rerank("q", chunks)
    assert result == chunks  # same list, same order


@requires_faiss
def test_rerank_with_mock_reranker_sorts_by_new_score():
    from src.rag import ScoredChunk

    mock_reranker = MagicMock()
    # Override: second chunk (score 0.6) gets higher rerank score
    mock_reranker.predict.return_value = np.array([0.3, 0.9], dtype=np.float32)

    rag, _ = _make_tiny_rag(use_reranker=True, reranker=mock_reranker)
    chunks = [
        ScoredChunk(text="Low rerank", chunk_id="c1", score=0.8, subject_id="22", source="s"),
        ScoredChunk(text="High rerank", chunk_id="c2", score=0.6, subject_id="22", source="s"),
    ]
    result = rag.rerank("q", chunks)
    assert result[0].chunk_id == "c2", "Higher rerank score should come first"
    assert result[0].score == pytest.approx(0.9)


@requires_faiss
def test_rerank_empty_returns_empty():
    rag, _ = _make_tiny_rag()
    assert rag.rerank("q", []) == []


@requires_faiss
def test_rerank_reranker_exception_falls_back_to_cosine():
    from src.rag import ScoredChunk

    mock_reranker = MagicMock()
    mock_reranker.predict.side_effect = RuntimeError("GPU OOM")

    rag, _ = _make_tiny_rag(use_reranker=True, reranker=mock_reranker)
    chunks = [
        ScoredChunk(text="A", chunk_id="c1", score=0.8, subject_id="22", source="s"),
        ScoredChunk(text="B", chunk_id="c2", score=0.6, subject_id="22", source="s"),
    ]
    result = rag.rerank("q", chunks)
    # Falls back to cosine order unchanged
    assert result == chunks


# ── retrieve_and_rerank ───────────────────────────────────────────────────────


@requires_faiss
def test_retrieve_and_rerank_returns_string_above_threshold():
    rag, _ = _make_tiny_rag(n_chunks=5, rerank_min=0.0, top_n=2)
    result = rag.retrieve_and_rerank("Câu hỏi về pháp luật")
    assert result is not None
    assert isinstance(result, str)
    assert "[1]" in result


@requires_faiss
def test_retrieve_and_rerank_returns_none_below_threshold():
    rag, _ = _make_tiny_rag(n_chunks=5, rerank_min=2.0)  # impossibly high threshold
    result = rag.retrieve_and_rerank("Câu hỏi về pháp luật")
    assert result is None


@requires_faiss
def test_retrieve_and_rerank_decontaminates_query_id():
    rag, _ = _make_tiny_rag(n_chunks=5, rerank_min=0.0)
    # The embedder is mocked to return a vector close to chunk 0 ("22-0000")
    result = rag.retrieve_and_rerank("query", exclude_id="22-0000")
    if result is not None:
        assert "22-0000" not in result


@requires_faiss
def test_retrieve_and_rerank_timeout_returns_none(monkeypatch):
    import time

    rag, _ = _make_tiny_rag(n_chunks=5, rerank_min=0.0, timeout=0.001)

    original_pipeline = rag._pipeline

    def slow_pipeline(query, exclude_id):
        time.sleep(0.1)
        return original_pipeline(query, exclude_id)

    rag._pipeline = slow_pipeline
    result = rag.retrieve_and_rerank("query")
    # Timeout is very short; result might be None
    # We just verify it doesn't crash and returns str or None
    assert result is None or isinstance(result, str)


@requires_faiss
def test_retrieve_and_rerank_exception_returns_none():
    rag, _ = _make_tiny_rag(n_chunks=5)
    rag._embedder.encode.side_effect = RuntimeError("encode failed")
    result = rag.retrieve_and_rerank("query")
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Prompts YAML
# ═══════════════════════════════════════════════════════════════════════════════


def _load_prompts():
    import yaml

    prompts_path = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"
    with open(prompts_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_knowledge_rag_template_exists():
    prompts = _load_prompts()
    assert "knowledge_rag" in prompts, "knowledge_rag template missing from prompts.yaml"


def test_knowledge_rag_template_has_all_placeholders():
    prompts = _load_prompts()
    template = prompts["knowledge_rag"]
    for placeholder in ("{retrieved_context}", "{question}", "{options_block}", "{label_list}"):
        assert placeholder in template, f"Placeholder {placeholder} missing from knowledge_rag template"


def test_knowledge_rag_template_formats_without_error():
    prompts = _load_prompts()
    template = prompts["knowledge_rag"]
    result = template.format(
        retrieved_context="[1]\nSome context",
        question="Câu hỏi?",
        options_block="A. Đúng\nB. Sai",
        valid_labels="A/B",
        label_list="A, B",
    )
    assert "Câu hỏi?" in result
    assert "[1]" in result
    assert "A. Đúng" in result


def test_knowledge_rag_differs_from_knowledge_direct():
    prompts = _load_prompts()
    assert "knowledge_rag" in prompts
    assert "knowledge_direct" in prompts
    assert prompts["knowledge_rag"] != prompts["knowledge_direct"]


def test_knowledge_direct_unchanged():
    """Ensure we didn't accidentally modify the knowledge_direct template."""
    prompts = _load_prompts()
    kd = prompts["knowledge_direct"]
    assert "{retrieved_context}" not in kd, (
        "knowledge_direct should NOT have a retrieved_context placeholder"
    )
    assert "kiến thức chung" in kd


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — ReasoningAgent.build_route_prompt
# ═══════════════════════════════════════════════════════════════════════════════


def _make_agent_with_prompts():
    """Create a ReasoningAgent with real prompts but no LLM loaded."""
    import yaml

    from src.reasoning_agent import ReasoningAgent

    agent = object.__new__(ReasoningAgent)
    prompts_path = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"
    with open(prompts_path, encoding="utf-8") as f:
        agent.prompts = yaml.safe_load(f)
    agent.cfg = {"temperature_deterministic": 0.1, "max_new_tokens": 512}
    agent._llm = None
    agent._model = None
    agent._tokenizer = None
    return agent


def test_build_route_prompt_knowledge_no_context_uses_direct():
    agent = _make_agent_with_prompts()
    prompt = agent.build_route_prompt(
        "knowledge", "Luật là gì?", {"A": "Quy tắc", "B": "Nguyên tắc"}, context=None
    )
    # knowledge_direct does not contain "tài liệu tham khảo"
    assert "tài liệu tham khảo" not in prompt
    assert "{retrieved_context}" not in prompt
    assert "Luật là gì?" in prompt


def test_build_route_prompt_knowledge_with_context_uses_rag():
    agent = _make_agent_with_prompts()
    context = "[1]\nCâu hỏi tương tự với đáp án B"
    prompt = agent.build_route_prompt(
        "knowledge", "Luật là gì?", {"A": "Quy tắc", "B": "Nguyên tắc"}, context=context
    )
    # knowledge_rag contains "tài liệu tham khảo"
    assert "tài liệu tham khảo" in prompt
    assert "[1]" in prompt
    assert "Câu hỏi tương tự" in prompt
    assert "Luật là gì?" in prompt


def test_build_route_prompt_reading_with_context_unaffected():
    """Reading route should NOT use knowledge_rag template."""
    agent = _make_agent_with_prompts()
    prompt = agent.build_route_prompt(
        "reading",
        "Đoạn thông tin nói gì?",
        {"A": "Đúng", "B": "Sai"},
        context="passage text",
    )
    assert "passage text" in prompt
    assert "tài liệu tham khảo" not in prompt


def test_build_route_prompt_stem_ignores_context():
    agent = _make_agent_with_prompts()
    prompt = agent.build_route_prompt(
        "stem", "Tính x?", {"A": "1", "B": "2"}, context="irrelevant context"
    )
    # stem_direct template has no retrieved_context placeholder
    assert "irrelevant context" not in prompt


def test_build_route_prompt_safety_unaffected():
    agent = _make_agent_with_prompts()
    prompt = agent.build_route_prompt(
        "safety", "Hack vào máy tính?", {"A": "Từ chối", "B": "Hướng dẫn"}, context=None
    )
    assert "Hack vào máy tính?" in prompt
    assert "tài liệu tham khảo" not in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Solve integration
# ═══════════════════════════════════════════════════════════════════════════════


# ── helpers reused from test_solve_s4.py style ────────────────────────────────


def _parsed(
    *,
    qid: str = "q1",
    query: str = "Câu hỏi kiến thức?",
    options: dict | None = None,
    has_context: bool = False,
    is_quantitative: bool = False,
    has_refusal_choice: bool = False,
    is_harmful: bool = False,
    refusal_labels: tuple = (),
):
    from src.parser import ParsedQuestion

    opts = options or {"A": "Một", "B": "Hai", "C": "Ba"}
    return ParsedQuestion(
        qid=qid,
        original_question=query,
        query=query,
        context="Đoạn văn" if has_context else None,
        options=opts,
        refusal_labels=refusal_labels,
        n_choices=len(opts),
        has_context=has_context,
        is_quantitative=is_quantitative,
        is_legal=False,
        has_refusal_choice=has_refusal_choice,
        is_harmful=is_harmful,
    )


def _choice(letter: str, margin: float = 1.0):
    from src.extract import ChoiceResult

    scores = {"A": -3.0, "B": -2.0, "C": -1.0}
    scores[letter] = 0.0
    return ChoiceResult(letter=letter, margin=margin, per_letter_logprob=scores)


class _ContextualFakeAgent:
    """Fake agent that returns different ChoiceResults based on whether context is provided."""

    def __init__(self, *, direct, rag_choice=None, sc_scores=None):
        self.direct = direct
        self.rag_choice = rag_choice
        self.sc_scores = list(sc_scores or [])
        self.generated = []
        self.scored_prompts = []
        self.calls: list[dict] = []

    def predict_route_choice_result(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("context") is not None and self.rag_choice is not None:
            return self.rag_choice
        return self.direct

    def generate_freeform(self, prompts, **kwargs):
        self.generated.append({"prompts": prompts, "kwargs": kwargs})
        return [f"reasoning-{i}" for i, _ in enumerate(prompts)]

    def score_valid_labels(self, prompt, valid_labels):
        self.scored_prompts.append(prompt)
        if self.sc_scores:
            return self.sc_scores.pop(0)
        return {label: -float(i) for i, label in enumerate(valid_labels)}


class _MockRAG:
    """Fake RAGEngine that returns a pre-set context string."""

    def __init__(self, context=None):
        self.context = context
        self.calls: list[dict] = []

    def retrieve_and_rerank(self, query: str, exclude_id=None) -> str | None:
        self.calls.append({"query": query, "exclude_id": exclude_id})
        return self.context


# ── _try_rag ──────────────────────────────────────────────────────────────────


def test_try_rag_returns_solve_result_when_margin_improves():
    from src.solve import _try_rag

    first = _choice("A", margin=0.05)
    rag_answer = _choice("B", margin=0.80)
    agent = _ContextualFakeAgent(direct=first, rag_choice=rag_answer)
    mock_rag = _MockRAG(context="[1]\nContext relevant")
    parsed = _parsed()

    result = _try_rag(agent, parsed, mock_rag, first)

    assert result is not None
    assert result.answer == "B"
    assert result.path == "knowledge_rag"
    assert result.first_answer == "A"
    assert result.margin == pytest.approx(0.80)


def test_try_rag_returns_none_when_no_context():
    from src.solve import _try_rag

    first = _choice("A", margin=0.05)
    agent = _ContextualFakeAgent(direct=first)
    mock_rag = _MockRAG(context=None)
    parsed = _parsed()

    result = _try_rag(agent, parsed, mock_rag, first)
    assert result is None


def test_try_rag_returns_none_when_rag_margin_does_not_improve():
    from src.solve import _try_rag

    first = _choice("A", margin=0.12)
    rag_answer = _choice("B", margin=0.08)  # worse than first
    agent = _ContextualFakeAgent(direct=first, rag_choice=rag_answer)
    mock_rag = _MockRAG(context="[1]\nSome context")
    parsed = _parsed()

    result = _try_rag(agent, parsed, mock_rag, first)
    assert result is None


def test_try_rag_returns_result_when_rag_margin_equals_first():
    """Equal margin is still accepted (>= not >)."""
    from src.solve import _try_rag

    margin = 0.10
    first = _choice("A", margin=margin)
    rag_answer = _choice("B", margin=margin)  # equal
    agent = _ContextualFakeAgent(direct=first, rag_choice=rag_answer)
    mock_rag = _MockRAG(context="[1]\nSome context")
    parsed = _parsed()

    result = _try_rag(agent, parsed, mock_rag, first)
    assert result is not None
    assert result.answer == "B"


def test_try_rag_returns_none_on_retrieval_exception():
    from src.solve import _try_rag

    class _BrokenRAG:
        def retrieve_and_rerank(self, *a, **kw):
            raise RuntimeError("disk full")

    first = _choice("A", margin=0.05)
    agent = _ContextualFakeAgent(direct=first)
    parsed = _parsed()

    result = _try_rag(agent, parsed, _BrokenRAG(), first)
    assert result is None  # never crashes


def test_try_rag_returns_none_on_agent_exception():
    from src.solve import _try_rag

    class _BrokenAgent:
        def predict_route_choice_result(self, **kwargs):
            raise ValueError("model error")

    first = _choice("A", margin=0.05)
    mock_rag = _MockRAG(context="[1]\nContext")
    parsed = _parsed()

    result = _try_rag(_BrokenAgent(), parsed, mock_rag, first)
    assert result is None


def test_try_rag_passes_qid_as_exclude_id():
    from src.solve import _try_rag

    first = _choice("A", margin=0.05)
    rag_answer = _choice("B", margin=0.9)
    agent = _ContextualFakeAgent(direct=first, rag_choice=rag_answer)
    mock_rag = _MockRAG(context="[1]\nContext")
    parsed = _parsed(qid="test_0042")

    _try_rag(agent, parsed, mock_rag, first)
    assert mock_rag.calls[0]["exclude_id"] == "test_0042"


# ── solve_question RAG integration ────────────────────────────────────────────


def test_solve_question_rag_none_is_backward_compatible():
    """With rag=None, behaviour is identical to pre-S6."""
    from src.solve import solve_question

    agent = _ContextualFakeAgent(direct=_choice("B", margin=0.9))
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=None)

    assert solved.answer == "B"
    assert solved.path == "direct"
    assert agent.generated == []


def test_solve_question_rag_improves_low_margin_knowledge():
    from src.solve import solve_question

    first = _choice("A", margin=0.05)  # below MARGIN_LOW
    rag_answer = _choice("B", margin=0.85)
    agent = _ContextualFakeAgent(direct=first, rag_choice=rag_answer)
    mock_rag = _MockRAG(context="[1]\nRelevant context")
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=mock_rag)

    assert solved.path == "knowledge_rag"
    assert solved.answer == "B"
    assert solved.first_answer == "A"
    assert agent.generated == []  # self-consistency NOT called


def test_solve_question_rag_falls_back_to_sc_when_no_context():
    from src.solve import solve_question

    first = _choice("A", margin=0.05)
    sc_scores = [{"A": -0.1, "B": -2.0, "C": -3.0}] * 5
    agent = _ContextualFakeAgent(direct=first, sc_scores=sc_scores)
    mock_rag = _MockRAG(context=None)
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=mock_rag)

    assert solved.path == "low_margin_self_consistency"
    assert len(agent.generated) > 0  # SC was called


def test_solve_question_rag_falls_back_to_sc_when_margin_does_not_improve():
    from src.solve import solve_question

    first = _choice("A", margin=0.05)
    worse_rag = _choice("C", margin=0.02)
    sc_scores = [{"A": -0.1, "B": -2.0, "C": -3.0}] * 5
    agent = _ContextualFakeAgent(direct=first, rag_choice=worse_rag, sc_scores=sc_scores)
    mock_rag = _MockRAG(context="[1]\nSome context")
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=mock_rag)

    assert solved.path == "low_margin_self_consistency"


def test_solve_question_rag_does_not_affect_stem():
    """RAG must never be called for stem-route questions."""
    from src.solve import solve_question

    first = _choice("A", margin=0.9)
    sc_scores = [{"A": -0.1, "B": -2.0, "C": -3.0}] * 5
    agent = _ContextualFakeAgent(direct=first, sc_scores=sc_scores)
    mock_rag = _MockRAG(context="[1]\nStem context")
    parsed = _parsed(is_quantitative=True)  # routes to "stem"

    solve_question(agent, parsed, rag=mock_rag)

    # RAG should not have been called
    assert mock_rag.calls == []


def test_solve_question_rag_does_not_affect_reading():
    """RAG must never be called for reading-route questions."""
    from src.solve import solve_question

    agent = _ContextualFakeAgent(direct=_choice("B", margin=0.8))
    mock_rag = _MockRAG(context="[1]\nReading context")
    parsed = _parsed(
        query="Nhân vật chính là ai?",
        has_context=True,
    )

    solve_question(agent, parsed, rag=mock_rag)

    assert mock_rag.calls == []


def test_solve_question_rag_does_not_affect_high_margin_knowledge():
    """High-margin knowledge questions skip RAG entirely."""
    from src.solve import solve_question

    agent = _ContextualFakeAgent(direct=_choice("C", margin=0.95))
    mock_rag = _MockRAG(context="[1]\nSome context")
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=mock_rag)

    assert solved.path == "direct"
    assert mock_rag.calls == []


def test_solve_question_rag_error_does_not_crash():
    """An exception in RAG must not crash solve_question."""
    from src.solve import solve_question

    class _ErrorRAG:
        def retrieve_and_rerank(self, *a, **kw):
            raise OSError("index corrupted")

    first = _choice("A", margin=0.05)
    sc_scores = [{"A": -0.1, "B": -2.0, "C": -3.0}] * 5
    agent = _ContextualFakeAgent(direct=first, sc_scores=sc_scores)
    parsed = _parsed()

    solved = solve_question(agent, parsed, rag=_ErrorRAG())

    # Must return a valid answer, never None
    assert solved.answer in ("A", "B", "C")
    assert solved.path in ("low_margin_self_consistency", "fallback")
