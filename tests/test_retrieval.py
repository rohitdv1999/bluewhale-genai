import numpy as np

from bluewhale.rag.embeddings import HashingEmbedder
from bluewhale.rag.retriever import rrf


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(64)
    a, b = e.embed(["python fastapi"]), e.embed(["python fastapi"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert max(fused, key=fused.get) in {"a", "b"}
    assert fused["d"] < fused["a"]


def test_hybrid_finds_exact_terms(services):
    hits = services.rag.retriever.retrieve("MLflow experiment tracking", k=3)
    assert hits[0].chunk.doc_id == "job_ml_platform"


def test_metadata_filters_on_type_and_skill(services):
    hits = services.rag.retriever.retrieve("python experience", k=10,
                                           filters={"doc_type": "resume"})
    assert hits and all(h.chunk.metadata["doc_type"] == "resume" for h in hits)
    hits = services.rag.retriever.retrieve("experience", k=10,
                                           filters={"doc_type": "job_description",
                                                    "skills": "kafka"})
    assert {h.chunk.doc_id for h in hits} <= {"job_backend_java", "job_data_engineer"}


def test_modes_return_results(services):
    for mode in ("dense", "keyword", "hybrid"):
        assert services.rag.retriever.retrieve("Spring Boot Kafka", k=3, mode=mode)


def test_empty_store_returns_nothing(settings):
    from bluewhale.container import build_services

    s = build_services(settings)
    assert s.rag.retriever.retrieve("anything", k=3) == []
