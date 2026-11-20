"""Small text helpers shared by ingestion, retrieval and guardrails."""

from __future__ import annotations

import html
import re
import unicodedata

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]")
_STOP = frozenset(
    """a an and are as at be but by for from has have i in into is it its of on or our
    that the their this to was we were will with you your they them he she his her who
    what when where which how all any can may must should would could not no do does""".split()
)


def normalize(text: str) -> str:
    """Unicode-normalize, decode HTML entities (twice, for double-escaped feeds),
    drop control characters and collapse runs of blank space."""
    if not text:
        return ""
    out = html.unescape(html.unescape(text))
    out = unicodedata.normalize("NFKC", out)
    out = "".join(ch for ch in out if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    out = out.replace("\t", " ")
    out = re.sub(r"[  ]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    words = _WORD.findall(text.lower())
    if keep_stopwords:
        return words
    return [w for w in words if w not in _STOP]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token for English). Good enough for
    budgeting context windows; the gateway records real usage from providers."""
    return max(1, (len(text) + 3) // 4)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*•") for p in parts if len(p.strip()) > 2]
