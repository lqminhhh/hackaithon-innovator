#!/usr/bin/env python3
"""Build the FAISS index from all knowledge sources.

Sources merged:
  1. Vietnamese Wikipedia articles (extracted JSON from wikiextractor)
  2. Domain-specific text files
  3. Pre-generated CoT reasoning chains

Outputs:
  data/faiss.index   — FAISS IndexFlatIP (cosine after L2-normalisation)
  data/chunks.jsonl  — one JSON object per line with chunk text + metadata
"""

from __future__ import annotations

import json
import glob
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBED_MODEL = "keepitreal/vietnamese-sbert"
BATCH_SIZE = 256


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-boundary chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_wiki_articles(wiki_dir: Path) -> list[dict]:
    """Load wikiextractor JSON output (one article per line)."""
    chunks = []
    for fpath in sorted(glob.glob(str(wiki_dir / "**" / "*"), recursive=True)):
        p = Path(fpath)
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    title = article.get("title", "")
                    text = article.get("text", "")
                    if not text:
                        continue
                    for idx, c in enumerate(chunk_text(text)):
                        chunks.append({
                            "text": c,
                            "source": "wikipedia",
                            "title": title,
                            "chunk_idx": idx,
                        })
        except (UnicodeDecodeError, IsADirectoryError):
            continue
    return chunks


def load_cot_chains(cot_path: Path) -> list[dict]:
    """Load pre-generated CoT reasoning chains."""
    chunks = []
    if not cot_path.exists():
        return chunks
    with open(cot_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = f"Câu hỏi: {obj.get('question', '')}\n{obj.get('cot_chain', '')}"
            chunks.append({
                "text": text,
                "source": "cot_chain",
                "correct_answer": obj.get("correct_answer", ""),
                "chunk_idx": 0,
            })
    return chunks


def load_domain_texts(domain_dir: Path) -> list[dict]:
    """Load plain .txt files from a domain texts directory."""
    chunks = []
    if not domain_dir.exists():
        return chunks
    for fpath in sorted(domain_dir.glob("**/*.txt")):
        text = fpath.read_text(encoding="utf-8")
        for idx, c in enumerate(chunk_text(text)):
            chunks.append({
                "text": c,
                "source": "domain",
                "file": fpath.name,
                "chunk_idx": idx,
            })
    return chunks


def build_index():
    wiki_dir = DATA_DIR / "wiki_text"
    cot_path = DATA_DIR / "cot_chains.jsonl"
    domain_dir = DATA_DIR / "domain_texts"

    print("Loading encoder...")
    encoder = SentenceTransformer(EMBED_MODEL)

    print("Loading sources...")
    all_chunks: list[dict] = []
    wiki = load_wiki_articles(wiki_dir)
    print(f"  Wikipedia: {len(wiki)} chunks")
    all_chunks.extend(wiki)

    cot = load_cot_chains(cot_path)
    print(f"  CoT chains: {len(cot)} chunks")
    all_chunks.extend(cot)

    domain = load_domain_texts(domain_dir)
    print(f"  Domain texts: {len(domain)} chunks")
    all_chunks.extend(domain)

    if not all_chunks:
        print("WARNING: No source data found. Creating empty index.")
        dim = encoder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, str(DATA_DIR / "faiss.index"))
        with open(DATA_DIR / "chunks.jsonl", "w") as f:
            pass
        return

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
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

    faiss.write_index(index, str(DATA_DIR / "faiss.index"))
    print(f"FAISS index saved: {index.ntotal} vectors, dim={dim}")

    with open(DATA_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Chunk metadata saved: {len(all_chunks)} entries")


if __name__ == "__main__":
    build_index()
