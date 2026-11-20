"""Second-stage reranking of the fused candidate list.

`CrossEncoderReranker` scores (query, passage) pairs jointly with a small
cross-encoder (fastembed, ONNX, CPU). `LexicalReranker` is the offline
fallback: query-term coverage plus a bonus for terms appearing as a phrase.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..text import tokens
from .documents import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]: ...


class LexicalReranker:
    def rerank(self, query, chunks, top_k):
        q = tokens(query)
        q_set = set(q)
        bigrams = {f"{a} {b}" for a, b in zip(q, q[1:], strict=False)}
        scored = []
        for c in chunks:
            body = " ".join(tokens(c.text))
            words = set(body.split())
            coverage = len(q_set & words) / len(q_set) if q_set else 0.0
            phrase = sum(1 for bg in bigrams if bg in body) / len(bigrams) if bigrams else 0.0
            scored.append((c, round(0.8 * coverage + 0.2 * phrase, 4)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:  # pragma: no cover - needs model weights
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model: Any = None

    def rerank(self, query, chunks, top_k):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.model_name)
        scores = list(self._model.rerank(query, [c.text for c in chunks]))
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda t: t[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]


def build_reranker(backend: str) -> Reranker:
    return CrossEncoderReranker() if backend == "cross-encoder" else LexicalReranker()
