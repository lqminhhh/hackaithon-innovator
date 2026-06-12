#!/usr/bin/env python3
"""Build the narrow FAISS index for the Entropy-Gated Jury.

Sources (small and surgical — NO general Wikipedia, see Lesson A):
  1. data/corpus/legal/     — Vietnamese legal texts (Hiến pháp, core bộ luật,
                              key Nghị định/Thông tư likely in scope)
  2. data/corpus/curriculum/ — Chính trị/triết học/kinh tế curriculum summaries
  3. data/corpus/domain/     — Any other narrow curated domain .txt files

Embedder: BGE-M3 (whitelisted, FP16)

Outputs:
  data/faiss_narrow.index  — FAISS IndexFlatIP (cosine after L2-normalisation)
  data/chunks_narrow.jsonl — one JSON per line with text + metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 128

OUT_INDEX = DATA_DIR / "faiss_narrow.index"
OUT_CHUNKS = DATA_DIR / "chunks_narrow.jsonl"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_corpus_dir(subdir: str, source_tag: str) -> list[dict]:
    """Load all .txt files from a corpus subdirectory."""
    chunks: list[dict] = []
    dirpath = CORPUS_DIR / subdir
    if not dirpath.exists():
        print(f"  [{source_tag}] Not found: {dirpath} — skipping", flush=True)
        return chunks
    files = sorted(dirpath.glob("**/*.txt"))
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for idx, c in enumerate(chunk_text(text)):
                chunks.append({
                    "text": c,
                    "source": source_tag,
                    "file": fpath.name,
                    "chunk_idx": idx,
                })
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}", flush=True)
    return chunks


def build_index():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BGE-M3 embedder...", flush=True)
    encoder = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[dict] = []

    legal = load_corpus_dir("legal", "legal")
    print(f"  Legal corpus:      {len(legal)} chunks", flush=True)
    all_chunks.extend(legal)

    curriculum = load_corpus_dir("curriculum", "curriculum")
    print(f"  Curriculum corpus: {len(curriculum)} chunks", flush=True)
    all_chunks.extend(curriculum)

    domain = load_corpus_dir("domain", "domain")
    print(f"  Domain corpus:     {len(domain)} chunks", flush=True)
    all_chunks.extend(domain)

    if not all_chunks:
        print(
            "\nWARNING: No corpus found under data/corpus/. "
            "Creating an empty index. "
            "Populate data/corpus/legal/, data/corpus/curriculum/, "
            "data/corpus/domain/ with .txt files and rerun.",
            flush=True,
        )
        dim = encoder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, str(OUT_INDEX))
        OUT_CHUNKS.write_text("")
        print(f"Empty index written to {OUT_INDEX}", flush=True)
        return

    texts = [c["text"] for c in all_chunks]
    print(f"\nEmbedding {len(texts)} chunks (model={EMBED_MODEL}, batch={BATCH_SIZE})...", flush=True)
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUT_INDEX))
    print(f"FAISS index saved: {index.ntotal} vectors, dim={dim} → {OUT_INDEX}", flush=True)

    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Chunk metadata saved: {len(all_chunks)} entries → {OUT_CHUNKS}", flush=True)


if __name__ == "__main__":
    build_index()
