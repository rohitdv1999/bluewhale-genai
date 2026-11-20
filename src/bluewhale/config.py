"""Runtime settings, read once from the environment.

Everything has a default that works offline, so `pytest` and the local demo
run without any cloud account. Production deployments override through env
vars (see .env.example) - secrets come from the platform's secret store, never
from this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_list(name: str, default: str) -> list[str]:
    return [p.strip() for p in _env(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    env: str = "local"

    # Storage
    vector_backend: str = "memory"  # memory | pgvector
    database_url: str = ""

    # Embeddings / reranking
    embedding_backend: str = "hashing"  # hashing | fastembed
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_backend: str = "lexical"  # lexical | cross-encoder

    # LLM gateway
    llm_providers: list[str] = field(default_factory=lambda: ["local"])
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.nova-pro-v1:0"
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-06-01"

    # Retrieval
    chunk_tokens: int = 220
    chunk_overlap: int = 40
    top_k: int = 6
    context_token_budget: int = 1800
    min_grounding: float = 0.35

    # API
    api_keys: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60

    # Agents
    agent_max_steps: int = 8

    @property
    def is_local(self) -> bool:
        return self.env == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        env=_env("BW_ENV", "local"),
        vector_backend=_env("BW_VECTOR_BACKEND", "memory"),
        database_url=_env("BW_DATABASE_URL"),
        embedding_backend=_env("BW_EMBEDDING_BACKEND", "hashing"),
        embedding_model=_env("BW_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=_env_int("BW_EMBEDDING_DIM", 384),
        reranker_backend=_env("BW_RERANKER_BACKEND", "lexical"),
        llm_providers=_env_list("BW_LLM_PROVIDERS", "local"),
        llm_timeout_s=_env_float("BW_LLM_TIMEOUT_S", 30.0),
        llm_max_retries=_env_int("BW_LLM_MAX_RETRIES", 2),
        aws_region=_env("AWS_REGION", "us-east-1"),
        bedrock_model_id=_env(
            "BW_BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0"
        ),
        azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        chunk_tokens=_env_int("BW_CHUNK_TOKENS", 220),
        chunk_overlap=_env_int("BW_CHUNK_OVERLAP", 40),
        top_k=_env_int("BW_TOP_K", 6),
        context_token_budget=_env_int("BW_CONTEXT_TOKENS", 1800),
        min_grounding=_env_float("BW_MIN_GROUNDING", 0.35),
        api_keys=_env_list("BW_API_KEYS", ""),
        rate_limit_per_minute=_env_int("BW_RATE_LIMIT_PER_MINUTE", 60),
        agent_max_steps=_env_int("BW_AGENT_MAX_STEPS", 8),
    )
