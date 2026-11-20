"""Wires the components together from settings (one place to swap backends)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agents.job_match import JobMatchWorkflow, build_job_tools
from .agents.runner import AgentRunner, MemoryCheckpoints
from .config import Settings, get_settings
from .llm.gateway import LLMGateway, build_gateway
from .rag.embeddings import build_embedder
from .rag.rerank import build_reranker
from .rag.retriever import HybridRetriever
from .rag.service import RAGService
from .rag.store import InMemoryStore, VectorStore


@dataclass
class Services:
    settings: Settings
    rag: RAGService
    gateway: LLMGateway
    agent: AgentRunner
    matcher: JobMatchWorkflow


def build_services(settings: Settings | None = None, gateway: LLMGateway | None = None,
                   store: VectorStore | None = None) -> Services:
    s = settings or get_settings()
    embedder = build_embedder(s.embedding_backend, s.embedding_model, s.embedding_dim)
    if store is None:
        if s.vector_backend == "pgvector":
            from .rag.store import PgVectorStore

            store = PgVectorStore(s.database_url, dim=s.embedding_dim)
        else:
            store = InMemoryStore()
    retriever = HybridRetriever(store, embedder, build_reranker(s.reranker_backend))
    gw = gateway or build_gateway(s)
    rag = RAGService(store, embedder, retriever, gw, s)
    agent = AgentRunner(gw, build_job_tools(rag), MemoryCheckpoints(), max_steps=s.agent_max_steps)
    return Services(s, rag, gw, agent, JobMatchWorkflow(rag))


def load_samples(rag: RAGService, root: str | Path | None = None) -> int:
    """Index the synthetic sample resumes and job descriptions under data/samples."""
    if root:
        base = Path(root)
    else:
        candidates = [Path.cwd() / "data" / "samples",
                      Path(__file__).resolve().parents[2] / "data" / "samples"]
        base = next((p for p in candidates if p.is_dir()), candidates[0])
    count = 0
    for path in sorted(base.glob("*/*.md")):
        doc_type = "resume" if path.parent.name == "resumes" else "job_description"
        rag.ingest_text(path.read_text("utf-8"), doc_id=path.stem,
                        metadata={"doc_type": doc_type, "source": path.parent.name})
        count += 1
    return count
