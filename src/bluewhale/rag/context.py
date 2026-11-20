"""Builds the context block that goes into the prompt.

Steps: drop chunks that look like prompt injection, drop near-duplicates
(overlapping chunks from the same document), then pack in rank order until
the token budget is used. Each block gets a stable citation number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..guardrails import scan_injection, wrap_untrusted
from ..observability import metrics
from ..text import estimate_tokens, tokens
from .retriever import RetrievedChunk


@dataclass
class ContextBundle:
    text: str
    blocks: dict[int, str]  # citation number -> raw chunk text
    sources: dict[int, dict] = field(default_factory=dict)
    dropped_injection: int = 0
    dropped_duplicate: int = 0
    tokens: int = 0


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def build_context(results: list[RetrievedChunk], token_budget: int = 1800,
                  dedupe_threshold: float = 0.8, injection_threshold: float = 0.7) -> ContextBundle:
    blocks: dict[int, str] = {}
    sources: dict[int, dict] = {}
    parts: list[str] = []
    seen: list[set[str]] = []
    used = 0
    dropped_inj = dropped_dup = 0

    for r in results:
        text = r.chunk.text
        if scan_injection(text).blocked(injection_threshold):
            dropped_inj += 1
            metrics.inc("context_injection_dropped_total")
            continue
        terms = set(tokens(text))
        if any(_jaccard(terms, s) >= dedupe_threshold for s in seen):
            dropped_dup += 1
            continue
        cost = estimate_tokens(text) + 12
        if used + cost > token_budget:
            break
        n = len(blocks) + 1
        blocks[n] = text
        sources[n] = {"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                      "title": r.chunk.metadata.get("title", ""), "section": r.chunk.section,
                      "score": r.score}
        parts.append(f"[{n}] {wrap_untrusted(text)}")
        seen.append(terms)
        used += cost

    return ContextBundle("\n\n".join(parts), blocks, sources, dropped_inj, dropped_dup, used)
