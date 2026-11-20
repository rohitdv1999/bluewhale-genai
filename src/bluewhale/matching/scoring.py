"""Candidate-job match scoring.

A transparent weighted model rather than a black box, because the score is
shown to candidates and has to be explainable feature by feature. Features
are in [0, 1]. A feature we cannot compute (no salary posted, no years
stated) is *missing*, not zero: its weight is dropped and the remaining
weights are renormalized, so sparse job posts are not unfairly pushed down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .skills import SENIORITY_RANK

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.35,
    "skills": 0.35,
    "seniority": 0.15,
    "experience": 0.15,
}


@dataclass
class CandidateProfile:
    skills: list[str]
    seniority: str = "mid"
    years: float | None = None
    embedding: list[float] | None = None


@dataclass
class JobProfile:
    job_id: str
    title: str
    skills: list[str]
    seniority: str = "mid"
    min_years: float | None = None
    embedding: list[float] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    job_id: str
    title: str
    score: float
    features: dict[str, float | None]
    matched_skills: list[str]
    missing_skills: list[str]
    reasons: list[str]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def compute_features(c: CandidateProfile, j: JobProfile) -> dict[str, float | None]:
    feats: dict[str, float | None] = {}
    if c.embedding and j.embedding:
        # cosine of normalized text embeddings sits roughly in [0, 1] for real text
        feats["semantic"] = max(0.0, min(1.0, _cosine(c.embedding, j.embedding)))
    else:
        feats["semantic"] = None

    if j.skills:
        feats["skills"] = len(set(c.skills) & set(j.skills)) / len(set(j.skills))
    else:
        feats["skills"] = None

    gap = SENIORITY_RANK.get(c.seniority, 2) - SENIORITY_RANK.get(j.seniority, 2)
    feats["seniority"] = 1.0 if gap in (0, 1) else 0.6 if gap == -1 or gap == 2 else 0.2

    if j.min_years is None or c.years is None:
        feats["experience"] = None
    elif c.years >= j.min_years:
        feats["experience"] = 1.0
    else:
        feats["experience"] = max(0.0, c.years / j.min_years)
    return feats


def score_features(feats: dict[str, float | None], weights: dict[str, float]) -> float:
    present = [(weights[k], v) for k, v in feats.items() if v is not None and k in weights]
    total_w = sum(w for w, _ in present)
    if total_w <= 0:
        return 0.0
    return max(0.0, min(1.0, sum(w * v for w, v in present) / total_w))


def explain(feats: dict[str, float | None], matched: list[str], missing: list[str]) -> list[str]:
    reasons: list[str] = []
    if matched:
        reasons.append(f"Matches {len(matched)} required skills: {', '.join(matched[:6])}")
    if missing:
        reasons.append(f"Gaps: {', '.join(missing[:5])}")
    if feats.get("experience") == 1.0:
        reasons.append("Meets the stated years of experience")
    elif feats.get("experience") is not None:
        reasons.append("Below the stated years of experience")
    if feats.get("seniority", 0) and feats["seniority"] < 0.5:  # type: ignore[operator]
        reasons.append("Seniority level is a stretch")
    return reasons


def rank_jobs(c: CandidateProfile, jobs: list[JobProfile],
              weights: dict[str, float] | None = None, top_n: int = 10) -> list[MatchResult]:
    w = weights or DEFAULT_WEIGHTS
    results: list[MatchResult] = []
    for j in jobs:
        feats = compute_features(c, j)
        matched = sorted(set(c.skills) & set(j.skills))
        missing = sorted(set(j.skills) - set(c.skills))
        results.append(MatchResult(j.job_id, j.title, round(score_features(feats, w), 4),
                                   feats, matched, missing, explain(feats, matched, missing)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
