"""Embedding model wrapper with pluggable backends.

Backends (in preference order):
  1. sentence-transformers (all-MiniLM-L6-v2) — dense semantic embeddings, best quality
  2. TF-IDF + SVD (sklearn) — sparse-to-dense fallback, no network required

The backend is selected at first call and cached for the session.
Set EMBEDDING_BACKEND=tfidf to force the fallback (useful in offline environments).
"""

from __future__ import annotations

import os
import numpy as np
from typing import Protocol


class EmbeddingBackend(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


# ── Backend implementations ────────────────────────────────────────────────────

class SentenceTransformerBackend:
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._dim = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> None:
        pass  # pre-trained; no fitting needed

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


class TFIDFBackend:
    """TF-IDF + truncated SVD (LSA) — fully offline, no GPU/network needed."""
    _N_COMPONENTS = 256

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=self._N_COMPONENTS, random_state=42)),
            ("norm", Normalizer(norm="l2")),
        ])
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        # Determine vocab size first to avoid SVD dimension error on small corpora
        tfidf_probe = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
        X_probe = tfidf_probe.fit_transform(texts)
        n_components = min(self._N_COMPONENTS, X_probe.shape[1] - 1, X_probe.shape[0] - 1)

        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm", Normalizer(norm="l2")),
        ])
        self._pipe.fit(texts)
        self._fitted = True
        self._dim_actual = n_components

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFBackend.fit() must be called before encode().")
        return self._pipe.transform(texts).astype(np.float32)

    @property
    def dim(self) -> int:
        return getattr(self, "_dim_actual", self._N_COMPONENTS)


# ── Module-level cache ─────────────────────────────────────────────────────────

_backend: EmbeddingBackend | None = None


def _get_backend() -> EmbeddingBackend:
    global _backend
    if _backend is not None:
        return _backend

    force = os.environ.get("EMBEDDING_BACKEND", "").lower()

    if force != "tfidf":
        try:
            _backend = SentenceTransformerBackend()
            print("[Embeddings] Using sentence-transformers backend (all-MiniLM-L6-v2)")
            return _backend
        except Exception as e:
            print(f"[Embeddings] sentence-transformers unavailable ({e}); falling back to TF-IDF+SVD")

    _backend = TFIDFBackend()
    print("[Embeddings] Using TF-IDF + SVD backend (offline mode)")
    return _backend


def fit_backend(corpus: list[str]) -> None:
    """Fit the backend on a corpus (required for TF-IDF; no-op for transformers)."""
    _get_backend().fit(corpus)


def embed_texts(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Return float32 array of shape (N, D) for N texts."""
    return _get_backend().encode(texts)


def embed_query(query: str) -> np.ndarray:
    """Return float32 array of shape (D,) for a single query."""
    return embed_texts([query])[0]
