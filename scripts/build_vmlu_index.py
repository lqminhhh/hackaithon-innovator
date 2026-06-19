#!/usr/bin/env python3
"""Build a FAISS dense-retrieval index from the VMLU v1.5 Q&A corpus.

Only entries with ground-truth answers (dev.jsonl + valid.jsonl) are indexed,
and only the 34 subjects where Qwen3.5 has knowledge gaps (RAG_INCLUDE_SUBJECTS).
STEM subjects and universal-theory subjects are excluded because the stem route
already handles them via self-consistency.

Outputs
-------
data/vmlu_faiss.index   -- FAISS IndexFlatIP (cosine after L2 normalization)
data/vmlu_chunks.jsonl  -- one JSON object per line with chunk text + metadata

Extensibility
-------------
New knowledge sources can be added by:
  1. Writing a loader function that returns list[dict] with ``text`` + metadata.
  2. Registering it in SOURCE_LOADERS below.
  3. Running with ``--sources vmlu,<new_source>`` or ``--append``.

Usage
-----
  python scripts/build_vmlu_index.py
  python scripts/build_vmlu_index.py --append --sources wikipedia
  python scripts/build_vmlu_index.py --index-path data/vmlu_faiss.index
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBED_MODEL,
    RAG_INCLUDE_SUBJECTS,
    SUBJECT_META,
    VMLU_CHUNKS_PATH,
    VMLU_DATA_DIR,
    VMLU_INDEX_PATH,
)

EMBED_BATCH_SIZE = 256


# ── chunk formatters ─────────────────────────────────────────────────────────

def _parse_answer_text(choices: list[str], answer_label: str) -> str:
    """Return the full choice string for the correct answer label."""
    prefix = f"{answer_label}."
    for choice in choices:
        stripped = choice.strip()
        if stripped.startswith(prefix) or stripped.startswith(f"{answer_label} "):
            return stripped
    return f"{answer_label}. (text not found)"


def format_vmlu_chunk(obj: dict) -> str:
    """Format a VMLU Q&A entry as a retrievable knowledge chunk.

    Full format:
        Câu hỏi: <question>
        Các lựa chọn:
        A. ...
        B. ...
        Đáp án đúng: B. <answer text>
    """
    choices_block = "\n".join(c.strip() for c in obj["choices"])
    answer_text = _parse_answer_text(obj["choices"], obj["answer"])
    return (
        f"Câu hỏi: {obj['question'].strip()}\n"
        f"Các lựa chọn:\n{choices_block}\n"
        f"Đáp án đúng: {answer_text}"
    )


# ── source loaders ───────────────────────────────────────────────────────────

def load_vmlu_chunks(
    data_dir: Path = VMLU_DATA_DIR,
    include_subjects: frozenset[str] = RAG_INCLUDE_SUBJECTS,
) -> list[dict]:
    """Load dev.jsonl + valid.jsonl, filter to the 34 knowledge-gap subjects.

    Returns a list of chunk dicts, each containing:
        text, id, subject_id, subject_name, category, source
    """
    chunks: list[dict] = []
    for split in ("dev", "valid"):
        fpath = data_dir / f"{split}.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping", flush=True)
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                subject_id = obj["id"].split("-")[0]
                if subject_id not in include_subjects:
                    continue
                meta = SUBJECT_META.get(subject_id, {"name": "Unknown", "category": "Unknown"})
                chunks.append({
                    "text": format_vmlu_chunk(obj),
                    "id": obj["id"],
                    "subject_id": subject_id,
                    "subject_name": meta["name"],
                    "category": meta["category"],
                    "source": f"vmlu_{split}",
                })
    return chunks


# Registry for future sources. Each value is a zero-arg callable returning chunks.
# To add Wikipedia: SOURCE_LOADERS["wikipedia"] = load_wikipedia_chunks
SOURCE_LOADERS: dict[str, Callable[[], list[dict]]] = {
    "vmlu": load_vmlu_chunks,
}


# ── embedding + index ─────────────────────────────────────────────────────────

def embed_chunks(
    texts: list[str],
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    print(f"  Loading embedder: {model_name}", flush=True)
    embedder = SentenceTransformer(model_name)
    t = time.time()
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  Embedded {len(texts)} chunks in {time.time() - t:.1f}s", flush=True)
    return np.array(embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ── append mode ──────────────────────────────────────────────────────────────

def load_existing(
    index_path: Path,
    chunks_path: Path,
) -> tuple[faiss.IndexFlatIP | None, list[dict]]:
    """Load existing index + chunks for append mode."""
    existing_chunks: list[dict] = []
    existing_index = None

    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_chunks.append(json.loads(line))
        print(f"  Loaded {len(existing_chunks)} existing chunks from {chunks_path}", flush=True)

    if index_path.exists():
        existing_index = faiss.read_index(str(index_path))
        print(f"  Loaded existing index: {existing_index.ntotal} vectors", flush=True)

    return existing_index, existing_chunks


def existing_ids(chunks: list[dict]) -> set[str]:
    return {c.get("id", "") for c in chunks}


# ── save ─────────────────────────────────────────────────────────────────────

def save_artifacts(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    index_path: Path,
    chunks_path: Path,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"  Saved FAISS index ({index.ntotal} vectors, dim={index.d}) -> {index_path}", flush=True)

    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved {len(chunks)} chunks -> {chunks_path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def build(
    sources: list[str],
    index_path: Path,
    chunks_path: Path,
    append: bool = False,
    embed_model: str = EMBED_MODEL,
) -> None:
    t0 = time.time()

    old_index, old_chunks = (None, [])
    if append:
        old_index, old_chunks = load_existing(index_path, chunks_path)

    seen_ids = existing_ids(old_chunks)

    new_chunks: list[dict] = []
    for source_name in sources:
        loader = SOURCE_LOADERS.get(source_name)
        if loader is None:
            print(f"  WARNING: unknown source '{source_name}', skipping", flush=True)
            continue
        loaded = loader()
        # Deduplicate by id when appending
        before = len(loaded)
        loaded = [c for c in loaded if c.get("id", "") not in seen_ids]
        print(
            f"  Source '{source_name}': {before} entries, "
            f"{len(loaded)} new after dedup",
            flush=True,
        )
        new_chunks.extend(loaded)

    if not new_chunks:
        print("No new chunks to embed. Nothing to do.", flush=True)
        return

    print(f"\nEmbedding {len(new_chunks)} new chunks...", flush=True)
    texts = [c["text"] for c in new_chunks]
    new_embeddings = embed_chunks(texts, model_name=embed_model)

    if append and old_index is not None:
        old_index.add(new_embeddings)
        final_index = old_index
        final_chunks = old_chunks + new_chunks
        print(
            f"  Appended {len(new_chunks)} vectors -> total {final_index.ntotal}",
            flush=True,
        )
    else:
        final_index = build_index(new_embeddings)
        final_chunks = new_chunks

    save_artifacts(final_index, final_chunks, index_path, chunks_path)

    from collections import Counter
    subject_counts = Counter(c.get("subject_id", "?") for c in final_chunks)
    source_counts = Counter(c.get("source", "?") for c in final_chunks)
    print(f"\n  Sources: {dict(source_counts)}", flush=True)
    print(f"  Subjects ({len(subject_counts)}): total {sum(subject_counts.values())} chunks", flush=True)
    print(f"  Total time: {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS RAG index from VMLU Q&A corpus (and optional future sources)"
    )
    parser.add_argument(
        "--sources",
        default="vmlu",
        help="Comma-separated list of source names to include (default: vmlu). "
             "Available: vmlu. Future: wikipedia, law, ...",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new chunks to an existing index instead of rebuilding from scratch",
    )
    parser.add_argument(
        "--index-path",
        default=str(VMLU_INDEX_PATH),
        help=f"Output FAISS index path (default: {VMLU_INDEX_PATH})",
    )
    parser.add_argument(
        "--chunks-path",
        default=str(VMLU_CHUNKS_PATH),
        help=f"Output chunks JSONL path (default: {VMLU_CHUNKS_PATH})",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help=f"Sentence embedding model (default: {EMBED_MODEL})",
    )
    args = parser.parse_args()

    source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
    print(f"Building RAG index", flush=True)
    print(f"  sources  : {source_list}", flush=True)
    print(f"  append   : {args.append}", flush=True)
    print(f"  index    : {args.index_path}", flush=True)
    print(f"  chunks   : {args.chunks_path}", flush=True)
    print(f"  embedder : {args.embed_model}", flush=True)
    print()

    build(
        sources=source_list,
        index_path=Path(args.index_path),
        chunks_path=Path(args.chunks_path),
        append=args.append,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
