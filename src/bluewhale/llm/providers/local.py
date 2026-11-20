"""Offline extractive provider.

Used for local development, CI and the eval suite so the whole pipeline runs
without network access or spend. It answers from the numbered context blocks
in the prompt by picking the sentences that overlap the question most, and
cites the blocks it used. It never invents text that is not in the context,
which makes it a useful floor when comparing hosted models in evals.
"""

from __future__ import annotations

import json
import re
import time

from ...text import estimate_tokens, sentences, tokens
from ..types import LLMRequest, LLMResponse, Usage

_BLOCK = re.compile(r"^\[(\d+)\]\s*(.*?)(?=^\[\d+\]|\Z|^Question:)", re.S | re.M)
_QUESTION = re.compile(r"^Question:\s*(.+)$", re.M)
NO_ANSWER = "I could not find this in the provided documents."


class LocalExtractiveProvider:
    name = "local"
    model = "extractive-v1"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 2,
                 min_coverage: float = 0.3) -> None:
        self.max_sentences = max_sentences
        # below these the context does not really answer the question -> say so
        self.min_overlap = min_overlap
        self.min_coverage = min_coverage

    def complete(self, request: LLMRequest, *, timeout_s: float) -> LLMResponse:
        started = time.perf_counter()
        prompt = "\n".join(m.content for m in request.messages if m.role == "user")
        q_match = _QUESTION.search(prompt)
        question = q_match.group(1) if q_match else prompt[-400:]
        q_terms = set(tokens(question)) or {"_"}

        scored: list[tuple[float, int, str]] = []
        for block in _BLOCK.finditer(prompt):
            cid, body = int(block.group(1)), re.sub(r"</?document>", "", block.group(2)).strip()
            # first line of a chunk is its "doc title / section" label: use it to pick the
            # right block, never as answer text
            header, _, rest = body.partition("\n")
            header_hits = len(q_terms & set(tokens(header)))
            for sent in sentences(rest):
                s_terms = set(tokens(sent))
                if not s_terms or sent in header:
                    continue
                overlap = len(q_terms & s_terms)
                if overlap >= self.min_overlap and overlap / len(q_terms) >= self.min_coverage:
                    norm = len(q_terms) ** 0.5 + len(s_terms) ** 0.5
                    scored.append(((overlap + 0.5 * header_hits) / norm, cid, sent))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked = scored[: self.max_sentences]

        if picked:
            answer = " ".join(f"{s} [{cid}]" for _, cid, s in picked)
            cites = sorted({cid for _, cid, _ in picked})
        else:
            answer, cites = NO_ANSWER, []

        content = json.dumps({"answer": answer, "citations": cites}) if request.json_mode else answer
        return LLMResponse(
            content=content,
            usage=Usage(estimate_tokens(prompt), estimate_tokens(content)),
            model=self.model,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
