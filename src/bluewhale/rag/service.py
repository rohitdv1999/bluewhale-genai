"""Ingestion and grounded question answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..guardrails import check_grounding, scan_injection
from ..llm.gateway import LLMGateway
from ..llm.prompts import get_prompt
from ..llm.providers.local import NO_ANSWER
from ..llm.structured import StructuredOutputError, complete_structured
from ..llm.types import Message
from ..observability import metrics, trace
from .chunking import chunk_document
from .context import build_context
from .documents import Document, make_document
from .embeddings import Embedder
from .retriever import HybridRetriever
from .store import Filters, VectorStore


class AnswerPayload(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    chunks: int
    skipped: bool
    metadata: dict[str, Any]


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    grounding_score: float
    refused: bool = False
    reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class RAGService:
    def __init__(self, store: VectorStore, embedder: Embedder, retriever: HybridRetriever,
                 gateway: LLMGateway, settings: Settings | None = None,
                 prompt_version: int | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.gateway = gateway
        self.settings = settings or get_settings()
        self.prompt_version = prompt_version

    # ---- ingestion -------------------------------------------------------
    def ingest_document(self, doc: Document) -> IngestResult:
        with trace("rag.ingest", doc_id=doc.doc_id):
            if self.store.document_hash(doc.doc_id) == doc.content_hash:
                metrics.inc("ingest_skipped_total")
                return IngestResult(doc.doc_id, 0, True, doc.metadata)
            chunks = chunk_document(doc, self.settings.chunk_tokens, self.settings.chunk_overlap)
            vectors = self.embedder.embed([c.text for c in chunks])
            self.store.upsert(doc.doc_id, doc.content_hash, doc.metadata, chunks, vectors)
            metrics.inc("ingest_documents_total", doc_type=doc.metadata.get("doc_type", "other"))
            metrics.inc("ingest_chunks_total", len(chunks))
            return IngestResult(doc.doc_id, len(chunks), False, doc.metadata)

    def ingest_text(self, text: str, *, doc_id: str | None = None,
                    metadata: dict[str, Any] | None = None) -> IngestResult:
        return self.ingest_document(make_document(text, doc_id=doc_id, metadata=metadata))

    # ---- answering -------------------------------------------------------
    def _refuse(self, question: str, reason: str, message: str) -> Answer:
        metrics.inc("answers_refused_total", reason=reason)
        return Answer(question, message, [], False, 0.0, refused=True, reason=reason)

    def answer(self, question: str, filters: Filters | None = None,
               k: int | None = None) -> Answer:
        with trace("rag.answer") as span:
            verdict = scan_injection(question)
            if verdict.blocked():
                return self._refuse(question, "prompt_injection",
                                    "This request looks like an attempt to override instructions.")

            results = self.retriever.retrieve(question, k or self.settings.top_k, filters)
            ctx = build_context(results, self.settings.context_token_budget)
            if not ctx.blocks:
                return self._refuse(question, "no_context", NO_ANSWER)

            prompt = get_prompt("grounded_answer", self.prompt_version)
            system, user = prompt.render(context=ctx.text, question=question)
            span.attrs.update(prompt=f"{prompt.name}@v{prompt.version}",
                              prompt_fp=prompt.fingerprint, context_blocks=len(ctx.blocks))
            msgs = [Message("system", system), Message("user", user)]
            try:
                payload, resp = complete_structured(self.gateway, msgs, AnswerPayload,
                                                    task="answer")
            except StructuredOutputError:
                return self._refuse(question, "invalid_output",
                                    "The model returned an invalid response.")

            report = check_grounding(payload.answer, payload.citations, ctx.blocks)
            not_found = payload.answer.strip() == NO_ANSWER
            grounded = not_found or (report.ok and
                                     report.supported_ratio >= self.settings.min_grounding)
            if not grounded:
                metrics.inc("answers_ungrounded_total")
                return self._refuse(question, "ungrounded",
                                    "I could not produce an answer supported by the documents.")

            cited = sorted(set(payload.citations) & set(ctx.blocks))
            metrics.inc("answers_total")
            return Answer(
                question=question,
                answer=payload.answer,
                citations=[{"n": n, **ctx.sources[n]} for n in cited],
                grounded=True,
                grounding_score=report.supported_ratio,
                usage={"provider": resp.provider, "model": resp.model,
                       "input_tokens": resp.usage.input_tokens,
                       "output_tokens": resp.usage.output_tokens,
                       "cost_usd": resp.cost_usd, "latency_ms": round(resp.latency_ms, 1),
                       "prompt": f"{prompt.name}@v{prompt.version}"},
            )
