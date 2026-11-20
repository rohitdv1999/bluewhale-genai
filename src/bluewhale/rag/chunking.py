"""Section-aware chunking.

Resumes and job posts have strong structure (Experience, Requirements,
Benefits...). Splitting on those headings first keeps each chunk about one
thing, which matters more for retrieval quality than the exact chunk size.
Long sections are then packed sentence by sentence up to a token budget with
a small overlap so a fact split across a boundary is still retrievable.
"""

from __future__ import annotations

import re

from ..text import estimate_tokens, sentences
from .documents import Chunk, Document

_HEADING = re.compile(
    r"^(?:#+\s*)?(professional summary|summary|about (?:the role|us|you)|experience|"
    r"work experience|professional experience|education|projects|technical skills|skills|"
    r"responsibilities|what you.ll do|requirements|qualifications|minimum qualifications|"
    r"preferred qualifications|nice to have|benefits|compensation|certifications)\s*:?\s*$",
    re.I | re.M,
)


def split_sections(text: str) -> list[tuple[str, str]]:
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [("", text)]
    out: list[tuple[str, str]] = []
    if marks[0].start() > 0:
        out.append(("header", text[: marks[0].start()].strip()))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end(): end].strip()
        if body:
            out.append((m.group(1).lower(), body))
    return [(s, b) for s, b in out if b]


def _pack(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        t = estimate_tokens(unit)
        if current and size + t > max_tokens:
            chunks.append(" ".join(current))
            # carry the tail of the previous chunk forward as overlap
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                carried += estimate_tokens(prev)
                if carried > overlap_tokens:
                    break
                carry.insert(0, prev)
            current, size = carry, sum(estimate_tokens(c) for c in carry)
        current.append(unit)
        size += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, max_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    title = doc.metadata.get("title", "")
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        units = sentences(body) or [body]
        for piece in _pack(units, max_tokens, overlap_tokens):
            pos = len(chunks)
            # prefix the doc title and section so a chunk is meaningful on its own
            prefix = " / ".join(p for p in (title, section) if p)
            text = f"{prefix}\n{piece}" if prefix else piece
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{pos}",
                doc_id=doc.doc_id,
                text=text,
                position=pos,
                section=section,
                metadata={k: v for k, v in doc.metadata.items() if k != "chars"},
            ))
    return chunks
