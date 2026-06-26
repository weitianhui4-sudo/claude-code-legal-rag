"""FAISS-backed vector store for document chunks.

Supports:
  - Building an index from DocumentChunk objects
  - Persisting to / loading from disk
  - k-NN similarity search with metadata return
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.models import DocumentChunk
from rag.embeddings import embed_texts, fit_backend


class VectorStore:
    def __init__(self) -> None:
        self._index = None          # faiss.IndexFlatIP
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk], verbose: bool = True) -> None:
        import faiss

        if verbose:
            print(f"[VectorStore] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        fit_backend(texts)  # no-op for transformer backends
        embeddings = embed_texts(texts, show_progress=verbose)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim (normalized vecs)
        index.add(embeddings)

        self._index = index
        self._chunks = chunks
        self._embeddings = embeddings

        if verbose:
            print(f"[VectorStore] Index built: {index.ntotal} vectors, dim={dim}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        import faiss
        import pickle
        from rag.embeddings import _get_backend

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        # Persist the fitted backend so TF-IDF vocab is available after reload
        with open(directory / "backend.pkl", "wb") as f:
            pickle.dump(_get_backend(), f)

        chunks_data = []
        for c in self._chunks:
            chunks_data.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "chunk_type": c.chunk_type.value,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "metadata": c.metadata,
            })
        (directory / "chunks.json").write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False)
        )
        print(f"[VectorStore] Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import faiss
        import pickle
        import rag.embeddings as emb_module
        from pipeline.models import ChunkType

        directory = Path(directory)
        store = cls()
        store._index = faiss.read_index(str(directory / "index.faiss"))

        backend_path = directory / "backend.pkl"
        if backend_path.exists():
            with open(backend_path, "rb") as f:
                emb_module._backend = pickle.load(f)

        raw = json.loads((directory / "chunks.json").read_text())
        store._chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                text=c["text"],
                chunk_type=ChunkType(c["chunk_type"]),
                section_title=c["section_title"],
                chunk_index=c["chunk_index"],
                token_count=c["token_count"],
                metadata=c["metadata"],
            )
            for c in raw
        ]
        print(f"[VectorStore] Loaded {store._index.ntotal} vectors from {directory}")
        return store

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return up to k (chunk, score) pairs ordered by descending similarity."""
        if self._index is None:
            raise RuntimeError("VectorStore is empty. Call build() or load() first.")

        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            if score >= score_threshold:
                results.append((self._chunks[idx], float(score)))
        return results
