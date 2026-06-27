"""Top-level RAG pipeline: build index and answer queries."""

from __future__ import annotations

from pathlib import Path

from pipeline.models import DocumentChunk
from rag.generator import RAGResponse, generate_answer
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class LegalRAG:
    """End-to-end legal RAG system."""

    def __init__(self) -> None:
        self._store: VectorStore | None = None
        self._retriever: Retriever | None = None

    def build_index(self, chunks: list[DocumentChunk], index_dir: str | Path) -> None:
        self._store = VectorStore()
        self._store.build(chunks)
        self._store.save(index_dir)
        self._retriever = Retriever(self._store)

    def load_index(self, index_dir: str | Path) -> None:
        self._store = VectorStore.load(index_dir)
        self._retriever = Retriever(self._store)

    def query(
        self,
        question: str,
        k: int = 5,
        model: str = "claude-sonnet-4-6",
    ) -> RAGResponse:
        if self._retriever is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        chunks = self._retriever.retrieve(question, k=k)
        return generate_answer(question, chunks, model=model)
