"""Gated RAG retrieval using BGE-M3 over the narrow curated corpus.

Design (planning doc §3.4 + Lessons A, C):
  - Corpus is small and surgical (Hiến pháp, core bộ luật, key nghị định/thông tư,
    curriculum summaries for chính trị/triết học). NO general Wikipedia.
  - Relative-delta gate: inject context ONLY when
        (top_score - mean_of_next_k) / mean_of_next_k >= spike_threshold
    A flat score profile → answers parametrically, zero context injected.
  - Optional Qwen-Rerank pass behind a config toggle; disabled by default
    until ablation proves net-positive accuracy per unit latency.

Usage
-----
retriever = GatedRetriever(cfg)
chunk = retriever.retrieve_for_question(question, exclude_qid=question.qid)
# → str | None   (None = no spike detected, answer without context)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_INDEX = _DATA_DIR / "faiss.index"
_DEFAULT_CHUNKS = _DATA_DIR / "chunks.jsonl"


class GatedRetriever:
    """BGE-M3 dense retrieval with relative-delta gating."""

    def __init__(
        self,
        embedder,                         # SentenceTransformer (BGE-M3)
        index_path: Path | str | None = None,
        chunks_path: Path | str | None = None,
        top_k: int = 5,
        spike_threshold: float = 0.3,    # relative delta required to inject
        use_rerank: bool = False,
        reranker=None,
    ):
        self.embedder = embedder
        self.top_k = top_k
        self.spike_threshold = spike_threshold
        self.use_rerank = use_rerank
        self.reranker = reranker
        self._loaded = False

        self._index_path = Path(index_path) if index_path else _DEFAULT_INDEX
        self._chunks_path = Path(chunks_path) if chunks_path else _DEFAULT_CHUNKS

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        if not self._index_path.exists():
            print(
                f"[retrieval] Index not found at {self._index_path}; "
                "RAG will return None for all queries.",
                flush=True,
            )
            self._index = None
            self._chunks: list[dict] = []
            self._loaded = True
            return

        t = time.time()
        self._index = faiss.read_index(str(self._index_path))
        print(
            f"[retrieval] FAISS index loaded: {self._index.ntotal} vectors "
            f"({time.time() - t:.1f}s)",
            flush=True,
        )

        self._chunks = []
        with open(self._chunks_path, encoding="utf-8") as f:
            for line in f:
                self._chunks.append(json.loads(line))
        print(f"[retrieval] {len(self._chunks)} chunks loaded.", flush=True)
        self._loaded = True

    # ── public API ────────────────────────────────────────────────────

    def retrieve_for_question(
        self,
        query: str,
        exclude_qid: str | None = None,
    ) -> Optional[str]:
        """Return the top-k chunks concatenated, or None if the gate rejects.

        Gate logic: inject ONLY when the top-hit cosine score is significantly
        higher than the mean of the remaining k hits (relative-delta gate).
        """
        self._lazy_load()
        if self._index is None or self._index.ntotal == 0:
            return None

        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        fetch_k = min(self.top_k * 3 + 10, self._index.ntotal)
        scores, indices = self._index.search(q_emb, fetch_k)
        scores = scores[0].tolist()
        indices = [int(i) for i in indices[0]]

        # Filter excluded QID and valid indices
        hits: list[tuple[int, float]] = []
        for doc_id, score in zip(indices, scores):
            if doc_id < 0 or doc_id >= len(self._chunks):
                continue
            chunk = self._chunks[doc_id]
            if (
                exclude_qid
                and chunk.get("source") == "cot_chain"
                and chunk.get("qid") == exclude_qid
            ):
                continue
            hits.append((doc_id, score))
            if len(hits) >= self.top_k + 5:
                break

        if not hits:
            return None

        top_score = hits[0][1]
        rest_scores = [s for _, s in hits[1:]]
        if not rest_scores:
            # Only one hit; inject it directly
            return self._chunks[hits[0][0]].get("text", "")

        mean_rest = sum(rest_scores) / len(rest_scores)

        if mean_rest <= 0:
            rel_delta = 10.0  # top score spikes if rest is near-zero
        else:
            rel_delta = (top_score - mean_rest) / abs(mean_rest)

        if rel_delta < self.spike_threshold:
            return None  # flat profile → no context injection (Lesson A)

        top_chunks = [self._chunks[doc_id].get("text", "") for doc_id, _ in hits[:self.top_k]]

        if self.use_rerank and self.reranker is not None:
            top_chunks = self._rerank(query, top_chunks)

        return "\n\n".join(top_chunks)

    def batch_retrieve(
        self,
        queries: list[str],
        exclude_qids: list[str | None] | None = None,
    ) -> list[Optional[str]]:
        """Batch retrieval — one encode call for all queries."""
        self._lazy_load()
        if self._index is None or self._index.ntotal == 0:
            return [None] * len(queries)

        if exclude_qids is None:
            exclude_qids = [None] * len(queries)

        all_embs = self.embedder.encode(
            queries,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=len(queries) > 50,
        ).astype(np.float32)

        fetch_k = min(self.top_k * 3 + 10, self._index.ntotal)
        all_scores, all_indices = self._index.search(all_embs, fetch_k)

        results: list[Optional[str]] = []
        for i, (exc_qid, scores_row, indices_row) in enumerate(
            zip(exclude_qids, all_scores, all_indices)
        ):
            hits: list[tuple[int, float]] = []
            for doc_id, score in zip(indices_row.tolist(), scores_row.tolist()):
                if int(doc_id) < 0 or int(doc_id) >= len(self._chunks):
                    continue
                chunk = self._chunks[int(doc_id)]
                if (
                    exc_qid
                    and chunk.get("source") == "cot_chain"
                    and chunk.get("qid") == exc_qid
                ):
                    continue
                hits.append((int(doc_id), float(score)))
                if len(hits) >= self.top_k + 5:
                    break

            if not hits:
                results.append(None)
                continue

            top_score = hits[0][1]
            rest_scores = [s for _, s in hits[1:]]

            if not rest_scores:
                results.append(self._chunks[hits[0][0]].get("text", ""))
                continue

            mean_rest = sum(rest_scores) / len(rest_scores)
            rel_delta = (
                (top_score - mean_rest) / abs(mean_rest)
                if mean_rest != 0
                else 10.0
            )

            if rel_delta < self.spike_threshold:
                results.append(None)
                continue

            top_chunks = [
                self._chunks[doc_id].get("text", "")
                for doc_id, _ in hits[:self.top_k]
            ]
            if self.use_rerank and self.reranker is not None:
                top_chunks = self._rerank(queries[i], top_chunks)

            results.append("\n\n".join(top_chunks))

        return results

    # ── reranker (optional) ───────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[str]) -> list[str]:
        """Apply Qwen-Rerank to re-order chunks. Disabled by default."""
        try:
            pairs = [[query, c] for c in chunks]
            scores = self.reranker.compute_score(pairs)
            ranked = sorted(zip(scores, chunks), reverse=True)
            return [c for _, c in ranked]
        except Exception:
            return chunks


def load_retriever(cfg: dict, embedder=None) -> GatedRetriever:
    """Build a GatedRetriever from the pipeline config."""
    from pathlib import Path

    rag_cfg = cfg.get("rag", {})
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder_id = cfg["models"].get("embedder", "BAAI/bge-m3")
        embedder = SentenceTransformer(embedder_id)

    return GatedRetriever(
        embedder=embedder,
        index_path=data_dir / rag_cfg.get("index_file", "faiss_narrow.index"),
        chunks_path=data_dir / rag_cfg.get("chunks_file", "chunks_narrow.jsonl"),
        top_k=rag_cfg.get("top_k", 5),
        spike_threshold=rag_cfg.get("spike_threshold", 0.3),
        use_rerank=rag_cfg.get("use_rerank", False),
    )
