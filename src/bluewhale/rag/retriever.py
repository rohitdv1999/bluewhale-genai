"""Hybrid retrieval: dense + keyword, fused with Reciprocal Rank Fusion, then reranked.

Dense search catches paraphrases ("led a team" vs "people management");
keyword search catches exact tokens embeddings blur (acronyms, versions like
"Java 21", tool names). RRF combines the two rankings without having to
calibrate their raw scores against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..observability import metrics, trace
from .documents import Chunk
from .embeddings import Embedder
from .rerank import Reranker
from .store import Filters, VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None
    keyword_rank: int | None


def rrf(rankings: list[list[str]], weights: list[float] | None = None,
        k: int = 60) -> dict[str, float]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + w / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker: Reranker | None = None,
                 candidates: int = 30, dense_weight: float = 1.0,
                 keyword_weight: float = 1.0) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, k: int = 6, filters: Filters | None = None,
                 mode: str = "hybrid") -> list[RetrievedChunk]:
        with trace("rag.retrieve", mode=mode, k=k) as span:
            dense: list[tuple[Chunk, float]] = []
            keyword: list[tuple[Chunk, float]] = []
            if mode in ("hybrid", "dense"):
                qvec = self.embedder.embed_query(query)
                dense = self.store.vector_search(qvec, self.candidates, filters)
            if mode in ("hybrid", "keyword"):
                keyword = self.store.keyword_search(query, self.candidates, filters)

            by_id = {c.chunk_id: c for c, _ in dense + keyword}
            d_rank = {c.chunk_id: i for i, (c, _) in enumerate(dense)}
            k_rank = {c.chunk_id: i for i, (c, _) in enumerate(keyword)}
            fused = rrf([[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in keyword]],
                        [self.dense_weight, self.keyword_weight])
            ordered = sorted(fused.items(), key=lambda t: t[1], reverse=True)

            if self.reranker and ordered:
                pool = [by_id[cid] for cid, _ in ordered[: self.candidates]]
                ranked = self.reranker.rerank(query, pool, k)
                # break reranker ties with the fused order
                fused_pos = {cid: i for i, (cid, _) in enumerate(ordered)}
                ranked.sort(key=lambda t: (-t[1], fused_pos[t[0].chunk_id]))
                results = [(c, s) for c, s in ranked]
            else:
                results = [(by_id[cid], s) for cid, s in ordered[:k]]

            out = [RetrievedChunk(c, round(s, 5), d_rank.get(c.chunk_id), k_rank.get(c.chunk_id))
                   for c, s in results]
            span.attrs.update(dense_hits=len(dense), keyword_hits=len(keyword), returned=len(out))
            metrics.inc("retrievals_total", mode=mode)
            if not out:
                metrics.inc("retrievals_empty_total", mode=mode)
            return out
