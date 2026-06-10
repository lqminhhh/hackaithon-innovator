"""Hybrid BM25 + dense retrieval agent with relevance gating.

Retrieves the top-k chunks from the FAISS index, optionally fused with BM25
keyword matches via Reciprocal Rank Fusion.  BM25 is auto-disabled on large
corpora (>50k chunks) because scoring every doc per query is too slow.

Embeddings are L2-normalised and stored in IndexFlatIP, so FAISS inner-product
scores equal cosine similarity — no per-chunk reconstruct needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"

_DEFAULT_BM25_MAX_CORPUS = 50_000


def _load_retrieval_config() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f).get("retrieval", {})


class RetrievalAgent:
    def __init__(
        self,
        embedder: SentenceTransformer,
        index_path: str | Path | None = None,
        chunks_path: str | Path | None = None,
        top_k: int = 5,
        relevance_threshold: float = 0.65,
        use_bm25: bool | None = None,
        bm25_max_corpus: int | None = None,
    ):
        retrieval_cfg = _load_retrieval_config()
        index_path = Path(index_path) if index_path else _DATA_DIR / "faiss.index"
        chunks_path = Path(chunks_path) if chunks_path else _DATA_DIR / "chunks.jsonl"
        bm25_max = bm25_max_corpus or retrieval_cfg.get(
            "bm25_max_corpus", _DEFAULT_BM25_MAX_CORPUS
        )

        self.embedder = embedder
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

        t = time.time()
        self.index = faiss.read_index(str(index_path))
        print(
            f"    FAISS index: {self.index.ntotal} vectors ({time.time() - t:.1f}s)",
            flush=True,
        )

        t = time.time()
        self.chunks: list[dict] = []
        self.chunk_texts: list[str] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(obj)
                self.chunk_texts.append(obj["text"])
        print(
            f"    Chunks loaded: {len(self.chunks)} ({time.time() - t:.1f}s)",
            flush=True,
        )

        if use_bm25 is None:
            use_bm25 = retrieval_cfg.get("use_bm25", True)
        if use_bm25 and self.index.ntotal > bm25_max:
            print(
                f"    BM25 disabled (corpus {self.index.ntotal:,} > {bm25_max:,}); "
                "using dense FAISS only",
                flush=True,
            )
            use_bm25 = False

        self.bm25: BM25Okapi | None = None
        if use_bm25 and self.chunk_texts:
            t = time.time()
            tokenized = [text.lower().split() for text in self.chunk_texts]
            self.bm25 = BM25Okapi(tokenized)
            print(f"    BM25 index built ({time.time() - t:.1f}s)", flush=True)
        elif use_bm25:
            print("    BM25 skipped (no chunks)", flush=True)
        else:
            print("    BM25 skipped", flush=True)

    # ── public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str, exclude_qid: str | None = None) -> list[str]:
        """Return top-k relevant chunks that pass the relevance gate."""
        if self.index.ntotal == 0:
            return []

        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        candidate_ids, candidate_scores = self._dense_search(q_emb, self._candidate_k())
        ranked_ids = self._maybe_fuse_bm25(query, candidate_ids)

        return self._select_hits_from_map(ranked_ids, score_map, exclude_qid)

    def batch_retrieve(
        self,
        queries: list[str],
        exclude_qids: list[str | None] | None = None,
    ) -> list[list[str]]:
        """Batch retrieval — one embed call + one batched FAISS search."""
        if self.index.ntotal == 0:
            return [[] for _ in queries]

        if exclude_qids is None:
            exclude_qids = [None] * len(queries)

        print(f"    Encoding {len(queries)} queries...", flush=True)
        all_query_embs = self.embedder.encode(
            queries,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        ).astype(np.float32)

        candidate_k = self._candidate_k()
        print(f"    FAISS batch search (k={candidate_k})...", flush=True)
        t = time.time()
        scores, indices = self.index.search(all_query_embs, candidate_k)
        print(f"    FAISS search done ({time.time() - t:.1f}s)", flush=True)

        results: list[list[str]] = []
        for i, exclude_qid in enumerate(exclude_qids):
            ranked_ids = self._maybe_fuse_bm25(
                queries[i],
                [int(doc_id) for doc_id in indices[i] if doc_id >= 0],
            )
            score_map = {
                int(doc_id): float(score)
                for doc_id, score in zip(indices[i], scores[i])
                if doc_id >= 0
            }
            hits = self._select_hits_from_map(
                ranked_ids, score_map, exclude_qid
            )
            results.append(hits)

        return results

    # ── internals ────────────────────────────────────────────────────────

    def _candidate_k(self) -> int:
        """Fetch extra candidates so decontamination + gating still fill top_k."""
        return min(max(self.top_k * 8, self.top_k + 20), self.index.ntotal)

    def _dense_search(
        self, query_emb: np.ndarray, k: int
    ) -> tuple[list[int], dict[int, float]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        scores, indices = self.index.search(query_emb, k)
        ids = [int(doc_id) for doc_id in indices[0] if doc_id >= 0]
        score_map = {
            int(doc_id): float(score)
            for doc_id, score in zip(indices[0], scores[0])
            if doc_id >= 0
        }
        return ids, score_map

    def _maybe_fuse_bm25(self, query: str, dense_ids: list[int]) -> list[int]:
        if self.bm25 is None:
            return dense_ids

        k_fetch = min(self.top_k * 4, len(self.chunk_texts))
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ids = [int(i) for i in np.argsort(bm25_scores)[::-1][:k_fetch]]

        fused: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank + 60)
        for rank, doc_id in enumerate(bm25_ids):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank + 60)

        return sorted(fused, key=fused.get, reverse=True)

    def _select_hits_from_map(
        self,
        ranked_ids: list[int],
        score_map: dict[int, float],
        exclude_qid: str | None,
    ) -> list[str]:
        hits: list[str] = []
        for doc_id in ranked_ids:
            chunk = self.chunks[doc_id]
            if (
                exclude_qid
                and chunk.get("source") == "cot_chain"
                and chunk.get("qid") == exclude_qid
            ):
                continue

            score = score_map.get(doc_id)
            if score is None:
                continue
            if score >= self.relevance_threshold:
                hits.append(self.chunk_texts[doc_id])
            if len(hits) >= self.top_k:
                break

        return hits
