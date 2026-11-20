"""Okapi BM25 for the in-memory store (Postgres uses ts_rank instead)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..text import tokens


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len: dict[str, int] = {}
        self.tf: dict[str, Counter[str]] = {}
        self.df: defaultdict[str, int] = defaultdict(int)

    def add(self, key: str, text: str) -> None:
        if key in self.tf:
            self.remove(key)
        counts = Counter(tokens(text))
        self.tf[key] = counts
        self.doc_len[key] = sum(counts.values())
        for term in counts:
            self.df[term] += 1

    def remove(self, key: str) -> None:
        counts = self.tf.pop(key, None)
        self.doc_len.pop(key, None)
        if counts:
            for term in counts:
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]

    def search(self, query: str, keys: set[str] | None = None, k: int = 20) -> list[tuple[str, float]]:
        n = len(self.tf)
        if n == 0:
            return []
        avg = sum(self.doc_len.values()) / n
        q = set(tokens(query))
        scores: list[tuple[str, float]] = []
        for key, counts in self.tf.items():
            if keys is not None and key not in keys:
                continue
            s = 0.0
            for term in q:
                f = counts.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[key] / avg)
                s += idf * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((key, s))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:k]
