"""Input and output guardrails.

Resumes and job posts are user-supplied text that ends up inside prompts, so
they are treated as untrusted:

* `scan_injection` scores text for instruction-override patterns. Retrieved
  chunks above the threshold are dropped from the context; user questions
  above it are refused before any model call.
* `redact_pii` masks emails, phone numbers and SSN-shaped strings before text
  is logged or sent to a hosted model that is not approved for PII.
* `wrap_untrusted` fences document text so the system prompt can tell the
  model where data ends and instructions begin.
* `check_grounding` verifies an answer's citations exist and that its content
  words are actually supported by the cited context.

None of this replaces provider-side controls (Bedrock Guardrails, Azure
content filters); it is the layer we own and can test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import tokens

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bignore (all |any )?(the )?(previous|prior|above|earlier) "
                r"(instructions|prompts?|rules)", re.I), 0.9),
    (re.compile(r"\bdisregard (the |all |your )?(system|previous|above)", re.I), 0.8),
    (re.compile(r"\byou are now\b|\bact as (an? )?(unfiltered|jailbroken|dan)\b", re.I), 0.6),
    (re.compile(r"\b(reveal|print|show|repeat) (the |your )?(system prompt|hidden instructions)",
                re.I), 0.8),
    (re.compile(r"\bnew instructions?:", re.I), 0.5),
    (re.compile(r"</?(system|assistant|instructions?)>", re.I), 0.5),
    (re.compile(r"\b(rate|rank|score) (this|me|the candidate) (as )?(10/10|highest|top)", re.I),
     0.6),
    (re.compile(r"\bdo not (tell|inform) the (user|recruiter)", re.I), 0.5),
    (re.compile(r"\b(call|invoke|use) the [a-z_]+ tool\b", re.I), 0.4),
]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


@dataclass
class InjectionVerdict:
    score: float
    matches: list[str] = field(default_factory=list)

    def blocked(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold


def scan_injection(text: str) -> InjectionVerdict:
    score, matches = 0.0, []
    for pattern, weight in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            matches.append(m.group(0))
            # combine like independent probabilities so several weak signals add up
            score = 1 - (1 - score) * (1 - weight)
    return InjectionVerdict(round(score, 3), matches)


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    return _PHONE.sub("[PHONE]", text)


def wrap_untrusted(text: str) -> str:
    # strip anything that looks like our own fence so a document cannot close it early
    cleaned = re.sub(r"</?document>", "", text, flags=re.I)
    return f"<document>\n{cleaned}\n</document>"


@dataclass
class GroundingReport:
    supported_ratio: float
    missing_citations: list[int]
    unsupported_terms: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_citations


_CITE = re.compile(r"\[(\d+)\]")


def check_grounding(answer: str, cited: list[int], context: dict[int, str]) -> GroundingReport:
    """Share of the answer's content words that appear in the cited blocks."""
    referenced = set(cited) | {int(n) for n in _CITE.findall(answer)}
    missing = sorted(n for n in referenced if n not in context)
    support_text = " ".join(context[n] for n in referenced if n in context)
    support = set(tokens(support_text))
    terms = [t for t in tokens(_CITE.sub(" ", answer)) if len(t) > 2 and not t.isdigit()]
    if not terms:
        return GroundingReport(1.0, missing, [])
    unsupported = [t for t in terms if t not in support]
    return GroundingReport(round(1 - len(unsupported) / len(terms), 3), missing,
                           sorted(set(unsupported))[:20])
