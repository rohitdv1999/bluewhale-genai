"""Document loading, cleaning and metadata extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..matching.skills import extract_seniority, extract_skills, extract_years
from ..text import normalize

# Page chrome that job boards wrap around the actual description.
_BOILERPLATE = [
    re.compile(r"^(apply now|share this job|save job|back to (search|jobs)|sign in)\b.*$", re.I | re.M),
    re.compile(r"^(cookie|we use cookies).*$", re.I | re.M),
    re.compile(r"^equal opportunity employer.*$", re.I | re.M),
]

_RESUME_HINTS = re.compile(r"\b(professional summary|work experience|education|"
                           r"technical skills|certifications)\b", re.I)
_JD_HINTS = re.compile(r"\b(responsibilities|requirements|qualifications|what you.ll do|"
                       r"about the role|we are looking for)\b", re.I)


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(raw: str) -> str:
    text = normalize(raw)
    for pattern in _BOILERPLATE:
        text = pattern.sub("", text)
    # collapse a body that a scraper captured twice back to back
    half = len(text) // 2
    if len(text) > 400 and text[:half].strip() == text[half:].strip():
        text = text[:half]
    return normalize(text)


def detect_doc_type(text: str) -> str:
    r, j = len(_RESUME_HINTS.findall(text)), len(_JD_HINTS.findall(text))
    if r == j == 0:
        return "other"
    return "resume" if r > j else "job_description"


def extract_metadata(text: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(base or {})
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    meta.setdefault("title", first_line[:120])
    meta.setdefault("doc_type", detect_doc_type(text))
    meta["skills"] = extract_skills(text)
    meta["seniority"] = extract_seniority(meta["title"])
    years = extract_years(text)
    if years is not None:
        meta["years"] = years
    meta["chars"] = len(text)
    return meta


def load_file(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md"}:
        return p.read_text("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader  # optional: pip install .[docs]

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    if suffix == ".docx":
        import docx  # optional: pip install .[docs]

        return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)
    raise ValueError(f"unsupported file type: {suffix}")


def make_document(text: str, *, doc_id: str | None = None,
                  metadata: dict[str, Any] | None = None) -> Document:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("document is empty after cleaning")
    did = doc_id or hashlib.sha1(cleaned.encode()).hexdigest()[:16]
    return Document(did, cleaned, extract_metadata(cleaned, metadata))
