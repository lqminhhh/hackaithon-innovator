"""Tests for the retrieval agent (requires FAISS index to be built)."""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _create_test_index(tmp_dir: Path, dim: int = 384):
    """Create a small FAISS index and chunks file for testing."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("keepitreal/vietnamese-sbert")

    texts = [
        "Thủ đô của Việt Nam là Hà Nội, nằm ở phía Bắc",
        "Hồ Chí Minh là thành phố lớn nhất Việt Nam",
        "Sông Mê Kông chảy qua nhiều quốc gia Đông Nam Á",
        "Nguyễn Du là tác giả của Truyện Kiều",
        "Phương trình bậc hai có dạng ax² + bx + c = 0",
    ]

    chunks_path = tmp_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            obj = {"text": text, "source": "test", "chunk_idx": i}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    embeddings = encoder.encode(texts, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_path = tmp_dir / "faiss.index"
    faiss.write_index(index, str(index_path))

    return index_path, chunks_path, encoder


class TestRetrievalAgent:
    def test_retrieve_returns_list(self):
        from src.retrieval_agent import RetrievalAgent
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            idx_path, chunks_path, encoder = _create_test_index(tmp_dir)
            agent = RetrievalAgent(
                embedder=encoder,
                index_path=idx_path,
                chunks_path=chunks_path,
                top_k=3,
                relevance_threshold=0.3,
            )
            results = agent.retrieve("Thủ đô của Việt Nam")
            assert isinstance(results, list)
            assert len(results) <= 3

    def test_empty_index(self):
        from src.retrieval_agent import RetrievalAgent
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("keepitreal/vietnamese-sbert")
            dim = encoder.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dim)
            idx_path = tmp_dir / "faiss.index"
            faiss.write_index(index, str(idx_path))
            chunks_path = tmp_dir / "chunks.jsonl"
            chunks_path.write_text("")
            agent = RetrievalAgent(
                embedder=encoder,
                index_path=idx_path,
                chunks_path=chunks_path,
            )
            results = agent.retrieve("Anything")
            assert results == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
