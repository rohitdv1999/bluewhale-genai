"""HTTP API (FastAPI).

    POST /v1/documents               ingest a resume or job description
    POST /v1/documents/upload        ingest a .txt/.md/.pdf/.docx file
    POST /v1/ask                     grounded Q&A with citations
    POST /v1/match                   rank indexed jobs for a resume
    POST /v1/agent/runs              start an agent run
    GET  /v1/agent/runs/{id}         run status, events, pending approval
    POST /v1/agent/runs/{id}/review  approve or reject the pending action
    POST /v1/feedback                thumbs up/down on an answer
    GET  /healthz /readyz /metrics
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..container import Services, build_services, load_samples
from ..guardrails import redact_pii
from ..observability import configure_logging, log, metrics, trace
from ..rag.documents import load_file
from .security import RateLimiter, require_api_key


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=200_000)
    doc_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    doc_type: str | None = Field(None, pattern=r"^(resume|job_description|other)$")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    doc_type: str | None = Field(None, pattern=r"^(resume|job_description)$")
    skill: str | None = Field(None, max_length=40)
    k: int = Field(6, ge=1, le=20)


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=40_000)
    top_n: int = Field(5, ge=1, le=20)


class AgentRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=2000)


class ReviewRequest(BaseModel):
    approved: bool
    reviewer: str = Field(..., min_length=1, max_length=80)
    note: str = Field("", max_length=500)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., max_length=64)
    helpful: bool
    comment: str = Field("", max_length=1000)


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(os.environ.get("BW_LOG_LEVEL", "INFO"))
        if app.state.services.settings.is_local and os.environ.get("BW_LOAD_SAMPLES", "1") == "1":
            n = load_samples(app.state.services.rag)
            log.info("loaded sample documents", extra={"fields": {"count": n}})
        yield

    app = FastAPI(title="BlueWhale GenAI", version=__version__, lifespan=lifespan)
    app.state.services = services or build_services()
    app.state.limiter = RateLimiter(app.state.services.settings.rate_limit_per_minute)
    app.state.feedback = []

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.inc("http_requests_total", path=path, status=str(response.status_code))
        metrics.observe("http_latency_ms", (time.perf_counter() - started) * 1000, path=path)
        response.headers["x-request-id"] = request_id
        return response

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz(s: Services = Depends(svc)) -> dict[str, Any]:
        return {"status": "ready", "store": s.rag.store.stats(),
                "providers": list(s.gateway.providers)}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        return metrics.prometheus()

    @app.post("/v1/documents", dependencies=[Depends(require_api_key)])
    def ingest(body: IngestRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        meta = {"doc_type": body.doc_type} if body.doc_type else None
        try:
            r = s.rag.ingest_text(body.text, doc_id=body.doc_id, metadata=meta)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"doc_id": r.doc_id, "chunks": r.chunks, "skipped": r.skipped,
                "doc_type": r.metadata.get("doc_type"), "skills": r.metadata.get("skills", [])}

    @app.post("/v1/documents/upload", dependencies=[Depends(require_api_key)])
    async def upload(file: UploadFile = File(...), s: Services = Depends(svc)) -> dict[str, Any]:
        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in {".txt", ".md", ".pdf", ".docx"}:
            raise HTTPException(415, "supported types: .txt .md .pdf .docx")
        data = await file.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "file larger than 5 MB")
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            text = load_file(tmp.name)
        r = s.rag.ingest_text(text)
        return {"doc_id": r.doc_id, "chunks": r.chunks, "skipped": r.skipped,
                "doc_type": r.metadata.get("doc_type")}

    @app.post("/v1/ask", dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if body.doc_type:
            filters["doc_type"] = body.doc_type
        if body.skill:
            filters["skills"] = body.skill.lower()
        with trace("api.ask") as span:
            a = s.rag.answer(body.question, filters or None, body.k)
            log.info("ask", extra={"fields": {"question": redact_pii(body.question)[:200],
                                              "refused": a.refused, "reason": a.reason}})
            return {"trace_id": span.trace_id, **a.__dict__}

    @app.post("/v1/match", dependencies=[Depends(require_api_key)])
    def match(body: MatchRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        state = s.matcher.run(body.resume_text, top_n=body.top_n)
        cand = state.candidate
        return {
            "candidate": {"skills": cand.skills, "seniority": cand.seniority,
                          "years": cand.years} if cand else None,
            "matches": [m.__dict__ for m in state.matches],
            "trail": state.trail,
        }

    @app.post("/v1/agent/runs", dependencies=[Depends(require_api_key)])
    def start_run(body: AgentRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        return _run_view(s.agent.start(body.goal))

    @app.get("/v1/agent/runs/{run_id}", dependencies=[Depends(require_api_key)])
    def get_run(run_id: str, s: Services = Depends(svc)) -> dict[str, Any]:
        run = s.agent.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return _run_view(run)

    @app.post("/v1/agent/runs/{run_id}/review", dependencies=[Depends(require_api_key)])
    def review(run_id: str, body: ReviewRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        try:
            run = s.agent.resume(run_id, approved=body.approved, reviewer=body.reviewer,
                                 note=body.note)
        except KeyError as exc:
            raise HTTPException(404, "run not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _run_view(run)

    @app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
    def feedback(body: FeedbackRequest, request: Request) -> dict[str, str]:
        request.app.state.feedback.append(body.model_dump())
        metrics.inc("feedback_total", helpful=str(body.helpful).lower())
        return {"status": "recorded"}

    return app


def _run_view(run) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "steps": run.steps,
        "final_answer": run.final_answer,
        "pending_action": {"tool": run.pending.name, "args": run.pending.arguments}
        if run.pending else None,
        "events": run.events,
        "cost_usd": round(run.cost_usd, 6),
    }
