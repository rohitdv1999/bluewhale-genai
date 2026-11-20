"""Skill, seniority and experience extraction.

A curated alias table maps the many ways people write a skill ("k8s",
"Kubernetes", "EKS") onto one canonical name, so a resume and a job post can
be compared on the same vocabulary. Matching is on word boundaries, longest
alias first, so "java" does not fire inside "javascript".
"""

from __future__ import annotations

import re
from functools import lru_cache

SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"],
    "java": ["java", "java 17", "java 21"],
    "typescript": ["typescript"],
    "javascript": ["javascript", "js", "node.js", "nodejs"],
    "sql": ["sql", "t-sql", "pl/sql"],
    "go": ["golang"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "spring boot": ["spring boot", "springboot"],
    "graphql": ["graphql", "apollo"],
    "rest apis": ["rest api", "rest apis", "restful"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "pgvector": ["pgvector"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "opensearch", "elastic"],
    "kafka": ["kafka"],
    "kinesis": ["kinesis"],
    "dynamodb": ["dynamodb"],
    "neo4j": ["neo4j"],
    "aws": ["aws", "amazon web services"],
    "aws bedrock": ["bedrock", "aws bedrock", "amazon bedrock"],
    "azure": ["azure"],
    "azure openai": ["azure openai"],
    "gcp": ["gcp", "google cloud"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s", "eks", "aks", "gke"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "cicd", "github actions", "jenkins", "gitlab ci"],
    "llm": ["llm", "llms", "large language model", "large language models", "foundation model",
            "foundation models"],
    "rag": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "embeddings": ["embedding", "embeddings", "vector embeddings"],
    "vector search": ["vector search", "vector database", "semantic search", "hybrid search"],
    "prompt engineering": ["prompt engineering", "prompt templates", "prompting"],
    "agents": ["agentic", "agents", "ai agents", "agentic workflows", "tool calling",
               "function calling"],
    "langchain": ["langchain"],
    "langgraph": ["langgraph"],
    "mcp": ["mcp", "model context protocol"],
    "llm evaluation": ["llm evaluation", "evals", "evaluation datasets", "ragas"],
    "pytorch": ["pytorch", "torch"],
    "hugging face": ["hugging face", "huggingface", "transformers"],
    "mlflow": ["mlflow"],
    "observability": ["observability", "opentelemetry", "tracing", "langfuse", "langsmith",
                      "prometheus", "grafana"],
    "streamlit": ["streamlit"],
    "react": ["react", "react.js"],
    "pytest": ["pytest"],
    "security": ["prompt injection", "rbac", "oauth", "jwt", "guardrails"],
}

_SENIORITY = [
    ("principal", re.compile(r"\b(principal|distinguished)\b", re.I)),
    ("staff", re.compile(r"\bstaff\b", re.I)),
    ("lead", re.compile(r"\b(lead|tech lead|manager)\b", re.I)),
    ("senior", re.compile(r"\b(senior|sr\.?|iii)\b", re.I)),
    ("junior", re.compile(r"\b(junior|jr\.?|entry[- ]level|new grad|associate)\b", re.I)),
]
SENIORITY_RANK = {"junior": 1, "mid": 2, "senior": 3, "lead": 4, "staff": 4, "principal": 5}

# "5+ years", "3-5 years of experience", "minimum of 4 years". The look-behind
# keeps phrases like "for over 100 years" (company history) out.
_YEARS = re.compile(
    r"(?<!over )(?<!than )\b(\d{1,2})(?:\s*(?:-|to|–)\s*(\d{1,2}))?\+?\s*(?:years?|yrs?)"
    r"(?:\s+of)?(?:\s+(?:professional|relevant|hands-on|industry|work))?\s+"
    r"(?:experience|exp\b)", re.I)


@lru_cache(maxsize=1)
def _alias_patterns() -> list[tuple[str, re.Pattern[str]]]:
    pairs = [(canon, alias) for canon, aliases in SKILL_ALIASES.items() for alias in aliases]
    pairs.sort(key=lambda p: len(p[1]), reverse=True)
    return [(canon, re.compile(rf"(?<![\w.+#-]){re.escape(alias)}(?![\w+#]|\.\w)", re.I))
            for canon, alias in pairs]


def extract_skills(text: str) -> list[str]:
    found: set[str] = set()
    # Longest aliases run first and their spans are blanked out, so "azure openai"
    # does not also count as "azure" unless "azure" appears on its own elsewhere.
    masked = text
    for canon, pattern in _alias_patterns():
        if pattern.search(masked):
            found.add(canon)
            masked = pattern.sub(" ", masked)
    return sorted(found)


def extract_seniority(title: str) -> str:
    for level, pattern in _SENIORITY:
        if pattern.search(title):
            return level
    return "mid"


def extract_years(text: str) -> float | None:
    """Minimum years of experience stated in the text (the lower bound of a range)."""
    values = [int(m.group(1)) for m in _YEARS.finditer(text)]
    values = [v for v in values if 0 < v <= 40]
    return float(min(values)) if values else None
