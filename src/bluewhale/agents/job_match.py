"""Job-matching tools and a graph-style matching workflow.

`build_job_tools` exposes the platform to the tool-calling agent (and to the
MCP server). `JobMatchWorkflow` is the fixed pipeline used by the API when no
open-ended reasoning is needed: a small state graph with explicit nodes and a
conditional retry edge, which is cheaper and easier to test than letting a
model plan every step.

    parse_resume -> retrieve_jobs -> score --(weak matches?)--> widen_search -> score
                                          \\-> explain -> done
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..matching.scoring import CandidateProfile, JobProfile, MatchResult, rank_jobs
from ..matching.skills import extract_seniority, extract_skills, extract_years
from ..observability import trace
from ..rag.documents import clean_text
from ..rag.service import RAGService
from .tools import Permission, Tool, ToolRegistry

JOB = "job_description"


# --------------------------------------------------------------------- tools
class SearchJobsArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    skill: str | None = Field(None, description="only jobs that list this skill")
    limit: int = Field(5, ge=1, le=20)


class GetJobArgs(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=64)


class MatchResumeArgs(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=40000)
    limit: int = Field(5, ge=1, le=20)


class ShortlistArgs(BaseModel):
    job_ids: list[str] = Field(..., min_length=1, max_length=20)
    note: str = Field("", max_length=500)


def build_job_tools(rag: RAGService, shortlist_sink: Callable[[list[str], str], Any] | None = None
                    ) -> ToolRegistry:
    registry = ToolRegistry()
    saved: list[dict[str, Any]] = []

    def search_jobs(query: str, skill: str | None, limit: int) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"doc_type": JOB}
        if skill:
            filters["skills"] = skill.lower()
        hits = rag.retriever.retrieve(query, k=limit * 3, filters=filters)
        out, seen = [], set()
        for h in hits:
            if h.chunk.doc_id in seen:
                continue
            seen.add(h.chunk.doc_id)
            out.append({"job_id": h.chunk.doc_id, "title": h.chunk.metadata.get("title"),
                        "snippet": h.chunk.text[:240], "score": h.score})
            if len(out) == limit:
                break
        return out

    def get_job(job_id: str) -> dict[str, Any]:
        for doc_id, meta in rag.store.documents({"doc_type": JOB}):
            if doc_id == job_id:
                return {"job_id": doc_id, **meta}
        raise LookupError(f"no job with id {job_id}")

    def match_resume(resume_text: str, limit: int) -> list[dict[str, Any]]:
        state = JobMatchWorkflow(rag).run(resume_text, top_n=limit)
        return [m.__dict__ for m in state.matches]

    def save_shortlist(job_ids: list[str], note: str) -> dict[str, Any]:
        entry = {"job_ids": job_ids, "note": note}
        if shortlist_sink:
            shortlist_sink(job_ids, note)
        saved.append(entry)
        return {"saved": len(job_ids)}

    registry.register(Tool("search_jobs", "Hybrid search over indexed job descriptions.",
                           SearchJobsArgs, search_jobs))
    registry.register(Tool("get_job", "Fetch one job's metadata by id.", GetJobArgs, get_job))
    registry.register(Tool("match_resume", "Rank indexed jobs against a resume.",
                           MatchResumeArgs, match_resume))
    registry.register(Tool("save_shortlist", "Save a shortlist of jobs for the candidate.",
                           ShortlistArgs, save_shortlist, permission=Permission.WRITE,
                           requires_approval=True))
    return registry


# ------------------------------------------------------------------ workflow
@dataclass
class MatchState:
    resume_text: str
    candidate: CandidateProfile | None = None
    jobs: list[JobProfile] = field(default_factory=list)
    matches: list[MatchResult] = field(default_factory=list)
    widened: bool = False
    trail: list[str] = field(default_factory=list)


class JobMatchWorkflow:
    def __init__(self, rag: RAGService, weak_threshold: float = 0.45,
                 retrieve_k: int = 40) -> None:
        self.rag = rag
        self.weak_threshold = weak_threshold
        self.retrieve_k = retrieve_k

    # nodes ----------------------------------------------------------------
    def parse_resume(self, s: MatchState) -> str:
        text = clean_text(s.resume_text)
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        vec = self.rag.embedder.embed_query(text[:4000])
        s.candidate = CandidateProfile(extract_skills(text), extract_seniority(first),
                                       extract_years(text), vec.tolist())
        return "retrieve_jobs"

    def _job_profiles(self, doc_ids: list[str]) -> list[JobProfile]:
        meta = dict(self.rag.store.documents({"doc_type": JOB}))
        ids = [d for d in doc_ids if d in meta]
        if not ids:
            return []
        # a compact "profile text" per job: title plus normalized skills
        texts = [f"{meta[d].get('title', '')} {' '.join(meta[d].get('skills', []))}" for d in ids]
        vecs = self.rag.embedder.embed(texts)
        return [JobProfile(d, meta[d].get("title", ""), meta[d].get("skills", []),
                           meta[d].get("seniority", "mid"), meta[d].get("years"),
                           vecs[i].tolist(), meta[d])
                for i, d in enumerate(ids)]

    def retrieve_jobs(self, s: MatchState) -> str:
        assert s.candidate is not None
        query = " ".join(s.candidate.skills) or s.resume_text[:500]
        hits = self.rag.retriever.retrieve(query, k=self.retrieve_k,
                                           filters={"doc_type": JOB})
        ids = list(dict.fromkeys(h.chunk.doc_id for h in hits))
        s.jobs = self._job_profiles(ids)
        return "score"

    def widen_search(self, s: MatchState) -> str:
        # fall back to every indexed job: retrieval may have been too narrow
        s.widened = True
        s.jobs = self._job_profiles([d for d, _ in self.rag.store.documents({"doc_type": JOB})])
        return "score"

    def score(self, s: MatchState, top_n: int) -> str:
        assert s.candidate is not None
        s.matches = rank_jobs(s.candidate, s.jobs, top_n=top_n)
        weak = not s.matches or s.matches[0].score < self.weak_threshold
        return "widen_search" if weak and not s.widened else "done"

    def run(self, resume_text: str, top_n: int = 5) -> MatchState:
        s = MatchState(resume_text)
        node = "parse_resume"
        with trace("workflow.job_match") as span:
            for _ in range(8):  # hard bound on graph steps
                s.trail.append(node)
                if node == "done":
                    break
                if node == "parse_resume":
                    node = self.parse_resume(s)
                elif node == "retrieve_jobs":
                    node = self.retrieve_jobs(s)
                elif node == "widen_search":
                    node = self.widen_search(s)
                elif node == "score":
                    node = self.score(s, top_n)
            span.attrs.update(trail="->".join(s.trail), matches=len(s.matches))
        return s
