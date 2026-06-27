"""Numpy-backed vector store for document chunks.

Uses brute-force cosine similarity (dot product on normalized vectors).
No native extensions required — works on any platform and Python version.

For production scale (millions of chunks) swap the _search method for a
FAISS or Chroma backend; the public interface stays identical.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from pipeline.models import DocumentChunk
from rag.embeddings import embed_texts, fit_backend


class VectorStore:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._chunks: list[DocumentChunk] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk], verbose: bool = True) -> None:
        if verbose:
            print(f"[VectorStore] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        fit_backend(texts)  # no-op for transformer backends
        self._embeddings = embed_texts(texts, show_progress=verbose)
        self._chunks = chunks
        if verbose:
            print(f"[VectorStore] Index built: {len(chunks)} vectors, dim={self._embeddings.shape[1]}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        import rag.embeddings as emb_module

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(str(directory / "embeddings.npy"), self._embeddings)

        with open(directory / "backend.pkl", "wb") as f:
            pickle.dump(emb_module._get_backend(), f)

        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "chunk_type": c.chunk_type.value,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        (directory / "chunks.json").write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False)
        )
        print(f"[VectorStore] Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        import rag.embeddings as emb_module
        from pipeline.models import ChunkType

        directory = Path(directory)
        store = cls()
        store._embeddings = np.load(str(directory / "embeddings.npy"))

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
        print(f"[VectorStore] Loaded {len(store._chunks)} vectors from {directory}")
        return store

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return up to k (chunk, score) pairs ordered by descending cosine similarity."""
        if self._embeddings is None:
            raise RuntimeError("VectorStore is empty. Call build() or load() first.")

        # Cosine similarity = dot product when vectors are L2-normalised
        scores = (self._embeddings @ query_embedding.astype(np.float32)).tolist()

        ranked = sorted(
            ((score, i) for i, score in enumerate(scores) if score >= score_threshold),
            reverse=True,
        )
        return [(self._chunks[i], score) for score, i in ranked[:k]]
