"""S6 RAG engine -- BGE-m3 dense retrieval + Qwen3-Reranker-0.6B gating.

Architecture
------------
1. BGE-m3 encodes the query into a dense vector.
2. FAISS IndexFlatIP returns top-K candidates (cosine similarity).
3. Decontamination removes any chunk whose ``id`` matches the question qid.
4. Optional Qwen3-Reranker-0.6B re-scores candidates.
   Scoring: generative yes/no -- the model formats each (query, doc) pair
   with a structured prompt, then P("yes") at the last token position is
   used as the relevance score (see ``_QwenReranker``).
   When disabled, raw FAISS cosine scores are used as the gate signal.
5. If the top score >= RERANK_MIN (0.5), the top-N chunk texts are returned
   as context. Otherwise, ``None`` is returned (no context injected).

The full pipeline is time-boxed (RAG_TIMEOUT seconds). A timeout yields None
so the caller falls back to no-context generation -- never blocks the run.

Only the knowledge route calls this module. READING / STEM / SAFETY routes
bypass it entirely (enforced in solve.py).

VRAM (planning_v2.md invariant: total ≤ 20 GB)
------
  Qwen3-Reranker-0.6B  FP16   ~1.2 GB
  BGE-m3 embedder      FP16   ~1.1 GB
  Auxiliary total             ~2.3 GB
  Remaining on 20GB GPU after vLLM (GPU_MEM_UTIL=0.85): ~3.0 GB  → ✓
  Remaining on 24GB GPU after vLLM:                      ~3.6 GB  → ✓
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from src.config import (
    EMBED_MODEL,
    RAG_TIMEOUT,
    RAG_TOP_K,
    RAG_TOP_N,
    RERANK_BATCH_SIZE,
    RERANK_MIN,
    RERANK_MODEL,
    VMLU_CHUNKS_PATH,
    VMLU_INDEX_PATH,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class _QwenReranker:
    """Wrapper for Qwen3-Reranker-0.6B with a CrossEncoder-compatible predict() interface.

    The model scores relevance via generative yes/no logits:
    - Format each (query, document) pair into a structured chat prompt.
    - Append ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` to prime
      the model to output "yes" or "no" as its next token.
    - Read P("yes") from the softmax over ["yes", "no"] logits at position -1.

    The tokenizer must use left-padding so that the last real token is always
    at position -1 in batched inference.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default: ``Qwen/Qwen3-Reranker-0.6B``).
    max_length:
        Truncation length in tokens. 512 comfortably fits the MCQ chunks used
        in this project (query ≤ 200 tokens + chunk ≤ 300 tokens + overhead).
    batch_size:
        Pairs per forward pass. 8 balances throughput and memory.
    """

    _SYSTEM = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note only output 'yes' or 'no'."
    )
    _INSTRUCTION = (
        "Given a Vietnamese multiple-choice exam question, retrieve similar "
        "exam questions with correct answers that are relevant as reference."
    )
    # Empty <think> block primes the model to output "yes"/"no" immediately
    _ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        max_length: int = 512,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",  # keeps last real token at position -1
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Resolve token IDs for "yes" / "no" (single tokens in Qwen vocabulary)
        self._yes_id = self._resolve_token_id("yes")
        self._no_id = self._resolve_token_id("no")

    def _resolve_token_id(self, word: str) -> int:
        """Return the single-token ID for ``word``, trying multiple forms."""
        for form in (word, f" {word}", word.capitalize()):
            ids = self.tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        # Last resort: take the final subword
        return int(self.tokenizer.encode(word, add_special_tokens=False)[-1])

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {self._INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}<|im_end|>\n"
            f"{self._ASSISTANT_PREFIX}"
        )

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return P("yes") for each (query, document) pair; shape (len(pairs),)."""
        import torch

        device = next(self.model.parameters()).device
        all_scores: list[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            texts = [self._format_pair(q, d) for q, d in batch]

            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Keep ALL tensor ops inside no_grad to avoid autograd/memory issues
            with torch.no_grad():
                logits = self.model(**encoded).logits  # (B, seq_len, vocab)
                last_logits = logits[:, -1, :].float()  # (B, vocab), ensure fp32
                yes_no = last_logits[:, [self._yes_id, self._no_id]]  # (B, 2)
                probs = torch.softmax(yes_no, dim=-1)[:, 0]  # (B,) = P("yes")
                batch_scores = probs.cpu().tolist()

            all_scores.extend(batch_scores)

        return np.array(all_scores, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    text: str
    chunk_id: str
    score: float
    subject_id: str
    source: str


class RAGEngine:
    """Retrieval-Augmented Generation engine for the knowledge route.

    Parameters
    ----------
    index_path:
        Path to FAISS IndexFlatIP file produced by build_vmlu_index.py.
    chunks_path:
        Path to companion JSONL file with chunk text + metadata.
    embed_model:
        Sentence embedding model name (should match what was used at build time).
    rerank_model:
        Cross-encoder model name for reranking. Ignored when use_reranker=False.
    use_reranker:
        If True (default), load the cross-encoder for more accurate gating.
        If False, use FAISS cosine scores directly (saves ~1-2 GB GPU memory).
    rerank_min:
        Minimum reranker (or cosine) score to inject context. Below this,
        retrieval is treated as "no relevant result found" and None is returned.
    top_k:
        Number of FAISS candidates to retrieve before reranking.
    top_n:
        Number of chunks to include in the injected context string.
    timeout:
        Maximum seconds to spend on one retrieve_and_rerank call.
    """

    def __init__(
        self,
        index_path: str | Path = VMLU_INDEX_PATH,
        chunks_path: str | Path = VMLU_CHUNKS_PATH,
        embed_model: str = EMBED_MODEL,
        rerank_model: str = RERANK_MODEL,
        use_reranker: bool = True,
        rerank_min: float = RERANK_MIN,
        top_k: int = RAG_TOP_K,
        top_n: int = RAG_TOP_N,
        timeout: float = RAG_TIMEOUT,
    ) -> None:
        self.rerank_min = rerank_min
        self.top_k = top_k
        self.top_n = top_n
        self.timeout = timeout
        self.use_reranker = use_reranker

        t = time.time()
        print("  [RAG] Loading FAISS index...", flush=True)
        self._index: faiss.IndexFlatIP = faiss.read_index(str(index_path))
        print(
            f"  [RAG] Index loaded: {self._index.ntotal} vectors, "
            f"dim={self._index.d} ({time.time() - t:.1f}s)",
            flush=True,
        )

        t = time.time()
        self._chunks: list[dict] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._chunks.append(json.loads(line))
        print(f"  [RAG] Chunks loaded: {len(self._chunks)} ({time.time() - t:.1f}s)", flush=True)

        t = time.time()
        print(f"  [RAG] Loading embedder: {embed_model}", flush=True)
        from sentence_transformers import SentenceTransformer
        self._embedder: SentenceTransformer = SentenceTransformer(embed_model)
        print(f"  [RAG] Embedder loaded ({time.time() - t:.1f}s)", flush=True)

        self._reranker: _QwenReranker | None = None
        if use_reranker:
            t = time.time()
            print(f"  [RAG] Loading reranker: {rerank_model}", flush=True)
            try:
                self._reranker = _QwenReranker(model_name=rerank_model)
                print(f"  [RAG] Reranker loaded ({time.time() - t:.1f}s)", flush=True)
            except Exception as exc:
                print(
                    f"  [RAG] WARNING: reranker failed to load ({exc}), "
                    "falling back to cosine-only scoring",
                    flush=True,
                )
                self._reranker = None

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve_and_rerank(
        self,
        query: str,
        exclude_id: str | None = None,
    ) -> str | None:
        """Full RAG pipeline with time-box.

        Returns
        -------
        Formatted context string (top-N chunks joined by separators) if the
        best reranker score passes the gate, else None.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._pipeline, query, exclude_id)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                print(
                    f"  [RAG] Timeout after {self.timeout}s for query: "
                    f"{query[:60]!r}",
                    flush=True,
                )
                return None
            except Exception as exc:
                print(f"  [RAG] Error during retrieval: {exc}", flush=True)
                return None

    def retrieve(
        self,
        query: str,
        exclude_id: str | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Embed query, FAISS search, decontaminate. Returns cosine-scored chunks."""
        k = top_k if top_k is not None else self.top_k
        if self._index.ntotal == 0:
            return []

        q_emb = self._embed(query)
        k_fetch = min(k + 5, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k_fetch)

        candidates: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            chunk_id = chunk.get("id", "")
            if exclude_id and chunk_id == exclude_id:
                continue
            candidates.append(
                ScoredChunk(
                    text=chunk.get("text", ""),
                    chunk_id=chunk_id,
                    score=float(score),
                    subject_id=chunk.get("subject_id", ""),
                    source=chunk.get("source", ""),
                )
            )
            if len(candidates) >= k:
                break

        return candidates

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Re-score candidates with Qwen3-Reranker; fall back to cosine if unavailable.

        Qwen3-Reranker returns P("yes") ∈ [0, 1]. Chunks are sorted by this
        score descending. The gate in ``_pipeline`` then compares the top
        score against ``rerank_min`` (default 0.5 = model is ≥50% confident).
        When ``_reranker`` is None, chunks keep their FAISS cosine order.
        """
        if not chunks:
            return chunks

        if self._reranker is None:
            return chunks  # already sorted by cosine

        pairs = [(query, c.text) for c in chunks]
        try:
            raw_scores: list[float] = self._reranker.predict(pairs).tolist()
        except Exception as exc:
            print(f"  [RAG] Reranker predict failed ({exc}), using cosine", flush=True)
            return chunks

        rescored = [
            ScoredChunk(
                text=c.text,
                chunk_id=c.chunk_id,
                score=float(raw_scores[i]),
                subject_id=c.subject_id,
                source=c.source,
            )
            for i, c in enumerate(chunks)
        ]
        return sorted(rescored, key=lambda x: x.score, reverse=True)

    # ── internals ────────────────────────────────────────────────────────────

    def _pipeline(self, query: str, exclude_id: str | None) -> str | None:
        candidates = self.retrieve(query, exclude_id=exclude_id, top_k=self.top_k)
        if not candidates:
            return None

        ranked = self.rerank(query, candidates)
        top = ranked[: self.top_n]

        if not top or top[0].score < self.rerank_min:
            return None

        return _format_context(top)

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embedder.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.array(vec, dtype=np.float32)


# ── formatting ───────────────────────────────────────────────────────────────

def _format_context(chunks: list[ScoredChunk]) -> str:
    """Join top-N chunks into a single context string for prompt injection."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}]\n{chunk.text}")
    return "\n\n".join(parts)
