"""Hybrid retriever combining dense (semantic) search with keyword boosting.

Retrieval strategy:
  1. Dense retrieval: embed query, search FAISS index for top-K candidates.
  2. Keyword boost: re-rank by adding a small bonus for exact keyword matches
     (important for legal queries with specific case names, statute numbers, etc.).
  3. MMR (Maximal Marginal Relevance): optionally diversify results to avoid
     redundant chunks from the same document section.
"""

from __future__ import annotations

import re

import numpy as np

from pipeline.models import DocumentChunk
from rag.embeddings import embed_query
from rag.vector_store import VectorStore


def _keyword_bonus(text: str, query: str, weight: float = 0.05) -> float:
    """Small additive score for each query token found in the chunk text."""
    tokens = re.findall(r"\w+", query.lower())
    text_lower = text.lower()
    hits = sum(1 for t in tokens if t in text_lower and len(t) > 3)
    return weight * hits / max(len(tokens), 1)


def _mmr(
    candidates: list[tuple[DocumentChunk, float]],
    query_emb: np.ndarray,
    embeddings: dict[str, np.ndarray],
    k: int,
    lambda_: float = 0.6,
) -> list[tuple[DocumentChunk, float]]:
    """Maximal Marginal Relevance re-ranking for diversity."""
    if not candidates:
        return []

    selected: list[tuple[DocumentChunk, float]] = []
    remaining = list(candidates)

    while remaining and len(selected) < k:
        best_score = -np.inf
        best_idx = 0
        for i, (chunk, rel_score) in enumerate(remaining):
            if not selected:
                mmr_score = rel_score
            else:
                emb = embeddings.get(chunk.chunk_id)
                if emb is None:
                    mmr_score = rel_score
                else:
                    # Max similarity to already-selected chunks
                    sel_embs = np.array([embeddings[s.chunk_id] for s, _ in selected if s.chunk_id in embeddings])
                    if len(sel_embs) == 0:
                        redundancy = 0.0
                    else:
                        sims = sel_embs @ emb
                        redundancy = float(sims.max())
                    mmr_score = lambda_ * rel_score - (1 - lambda_) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


class Retriever:
    def __init__(self, store: VectorStore, use_mmr: bool = True) -> None:
        self._store = store
        self._use_mmr = use_mmr

    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        score_threshold: float = 0.2,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return top-k (chunk, score) pairs for a given query."""
        query_emb = embed_query(query)

        # Dense retrieval: fetch more candidates than needed for re-ranking
        candidates = self._store.search(query_emb, k=fetch_k, score_threshold=score_threshold)

        # Keyword boost re-ranking
        boosted = [
            (chunk, score + _keyword_bonus(chunk.text, query))
            for chunk, score in candidates
        ]
        boosted.sort(key=lambda x: x[1], reverse=True)

        if self._use_mmr and self._store._embeddings is not None:
            # Build chunk_id -> embedding map for MMR
            emb_map = {
                chunk.chunk_id: self._store._embeddings[i]
                for i, chunk in enumerate(self._store._chunks)
            }
            return _mmr(boosted[:fetch_k], query_emb, emb_map, k=k)

        return boosted[:k]
