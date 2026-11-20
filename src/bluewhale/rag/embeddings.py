"""Embedding backends.

* `FastEmbedEmbedder` - BAAI/bge-small-en-v1.5 (384-d) through ONNX Runtime.
  Runs on CPU, no GPU or API key, and is what production uses.
* `HashingEmbedder` - deterministic feature-hashing of unigrams and bigrams.
  No model download, so unit tests and CI stay fast and reproducible. It is
  a lexical embedding, not a semantic one - fine for tests, not for quality.

Both return L2-normalized float32 vectors so cosine == dot product.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

from ..text import tokens


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384,
                 batch_size: int = 64) -> None:
        self.model_name = model
        self.dim = dim
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # optional: pip install .[embeddings]

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._load().embed(texts, batch_size=self.batch_size))
        return _l2(np.asarray(vecs, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models expect this instruction prefix on the query side only
        vec = next(iter(self._load().query_embed([text])))
        return _l2(np.asarray([vec], dtype=np.float32))[0]


def build_embedder(backend: str, model: str, dim: int) -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model, dim)
    return HashingEmbedder(dim)
