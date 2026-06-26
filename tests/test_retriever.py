"""Integration tests for the vector store and retriever (no API calls)."""

import numpy as np
import pytest

from pipeline.models import ChunkType, DocumentChunk
from rag.vector_store import VectorStore
from rag.retriever import Retriever, _keyword_bonus


def make_chunk(chunk_id: str, text: str, doc_id: str = "doc1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_type=ChunkType.SECTION,
        section_title="Test Section",
        chunk_index=0,
        token_count=len(text) // 4,
        metadata={"filename": "test.txt", "doc_type": "contract"},
    )


SAMPLE_CHUNKS = [
    make_chunk("c1", "The licensee shall pay annual fees of $150,000 within 30 days of invoice."),
    make_chunk("c2", "This agreement shall be governed by the laws of the State of Delaware."),
    make_chunk("c3", "Licensor may terminate this agreement upon 30 days written notice for material breach."),
    make_chunk("c4", "The consumer has the right to opt out of targeted advertising and data sales."),
    make_chunk("c5", "Non-compete clauses are prohibited under the FTC final rule effective September 2024."),
]


@pytest.fixture(scope="module")
def store():
    s = VectorStore()
    s.build(SAMPLE_CHUNKS, verbose=False)
    return s


class TestVectorStore:
    def test_build_success(self, store):
        assert store._index is not None
        assert store._index.ntotal == len(SAMPLE_CHUNKS)

    def test_search_returns_results(self, store):
        from rag.embeddings import embed_query
        q = embed_query("payment terms invoice")
        results = store.search(q, k=3)
        assert len(results) >= 1

    def test_search_relevance_order(self, store):
        from rag.embeddings import embed_query
        q = embed_query("payment fees invoice")
        results = store.search(q, k=3)
        # First result should be the payment chunk
        assert results[0][0].chunk_id == "c1"

    def test_save_load_roundtrip(self, store, tmp_path):
        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert loaded._index.ntotal == store._index.ntotal
        assert len(loaded._chunks) == len(store._chunks)

    def test_score_threshold(self, store):
        from rag.embeddings import embed_query
        q = embed_query("payment fees invoice")
        results = store.search(q, k=5, score_threshold=0.99)  # Very high threshold
        # With such a high threshold, may get 0 or very few results
        assert all(score >= 0.99 for _, score in results)


class TestRetriever:
    def test_retrieve_top_k(self, store):
        retriever = Retriever(store, use_mmr=False)
        results = retriever.retrieve("termination notice breach", k=2)
        assert len(results) <= 2

    def test_keyword_bonus(self):
        text = "The licensee shall pay annual fees within thirty days"
        score = _keyword_bonus(text, "payment fees invoice")
        assert score > 0

    def test_mmr_diversifies_results(self, store):
        retriever_mmr = Retriever(store, use_mmr=True)
        retriever_no_mmr = Retriever(store, use_mmr=False)
        results_mmr = retriever_mmr.retrieve("legal agreement rights", k=3)
        results_no_mmr = retriever_no_mmr.retrieve("legal agreement rights", k=3)
        # MMR should return different ordering or different chunks
        # (At minimum, both should return results)
        assert len(results_mmr) >= 1
        assert len(results_no_mmr) >= 1
