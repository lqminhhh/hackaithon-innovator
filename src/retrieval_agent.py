"""Hybrid BM25 + dense retrieval agent with relevance gating.

Retrieves the top-k chunks from the FAISS index, fused with BM25
keyword matches via Reciprocal Rank Fusion.  Chunks that fall below
the cosine relevance threshold are dropped silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class RetrievalAgent:
    def __init__(
        self,
        embedder: SentenceTransformer,
        index_path: str | Path | None = None,
        chunks_path: str | Path | None = None,
        top_k: int = 5,
        relevance_threshold: float = 0.65,
    ):
        index_path = Path(index_path) if index_path else _DATA_DIR / "faiss.index"
        chunks_path = Path(chunks_path) if chunks_path else _DATA_DIR / "chunks.jsonl"

        self.embedder = embedder
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

        self.index = faiss.read_index(str(index_path))

        self.chunks: list[dict] = []
        self.chunk_texts: list[str] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(obj)
                self.chunk_texts.append(obj["text"])

        if self.chunk_texts:
            tokenized = [t.lower().split() for t in self.chunk_texts]
            self.bm25: BM25Okapi | None = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    # ── public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[str]:
        """Return top-k relevant chunks that pass the relevance gate."""
        if self.index.ntotal == 0:
            return []

        fused_ids = self._hybrid_search(query)
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]

        results: list[str] = []
        for doc_id in fused_ids:
            chunk_emb = self._get_embedding(doc_id)
            if chunk_emb is not None and self._passes_gate(query_emb, chunk_emb):
                results.append(self.chunk_texts[doc_id])
            if len(results) >= self.top_k:
                break

        return results

    # ── internals ────────────────────────────────────────────────────────

    def _hybrid_search(self, query: str) -> list[int]:
        """BM25 + dense search merged by reciprocal rank fusion."""
        k_fetch = self.top_k * 4

        # Dense retrieval
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        n_search = min(k_fetch, self.index.ntotal)
        _, dense_ids = self.index.search(q_emb, n_search)
        dense_ids = [int(i) for i in dense_ids[0] if i >= 0]

        # BM25 retrieval
        bm25_ids: list[int] = []
        if self.bm25 is not None:
            tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokens)
            bm25_ids = list(np.argsort(bm25_scores)[::-1][:k_fetch])

        # Reciprocal rank fusion
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + 60)
        for rank, doc_id in enumerate(bm25_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + 60)

        return sorted(scores, key=scores.get, reverse=True)

    def _get_embedding(self, doc_id: int) -> np.ndarray | None:
        """Reconstruct a stored embedding from the FAISS index."""
        try:
            vec = np.zeros((1, self.index.d), dtype=np.float32)
            self.index.reconstruct(doc_id, vec[0])
            return vec[0]
        except RuntimeError:
            return None

    def _passes_gate(self, query_emb: np.ndarray, chunk_emb: np.ndarray) -> bool:
        """Check cosine similarity against the relevance threshold."""
        q_norm = np.linalg.norm(query_emb)
        c_norm = np.linalg.norm(chunk_emb)
        if q_norm == 0 or c_norm == 0:
            return False
        cosine = float(np.dot(query_emb, chunk_emb) / (q_norm * c_norm))
        return cosine >= self.relevance_threshold
