import dataclasses

import pytest
from fastapi.testclient import TestClient

from bluewhale.api.app import create_app
from bluewhale.container import build_services, load_samples


@pytest.fixture
def client(services):
    with TestClient(create_app(services)) as c:
        yield c


def test_health_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["store"]["documents"] >= 9
    client.post("/v1/ask", json={"question": "Which job needs MLflow?"})
    body = client.get("/metrics").text
    assert "bw_http_requests_total" in body and "bw_llm_tokens_total" in body


def test_ask_returns_citations_and_trace(client):
    r = client.post("/v1/ask", json={"question": "Which job needs MLflow experiment tracking?"})
    data = r.json()
    assert r.status_code == 200 and data["grounded"] and data["citations"]
    assert len(data["trace_id"]) == 32


def test_ingest_validates_input(client):
    assert client.post("/v1/documents", json={"text": "short"}).status_code == 422
    r = client.post("/v1/documents", json={
        "text": "Platform Engineer\nRequirements\n- 3+ years of experience with Terraform and AWS.",
        "doc_id": "new_job"})
    assert r.status_code == 200 and "terraform" in r.json()["skills"]


def test_match_endpoint(client):
    text = open("data/samples/resumes/resume_java_backend.md").read()
    data = client.post("/v1/match", json={"resume_text": text, "top_n": 3}).json()
    assert data["matches"][0]["job_id"] == "job_backend_java"
    assert "java" in data["candidate"]["skills"]


def test_agent_run_lifecycle(client):
    r = client.post("/v1/agent/runs", json={"goal": "Find a role with MLflow"}).json()
    assert r["status"] == "completed"
    assert client.get(f"/v1/agent/runs/{r['run_id']}").status_code == 200
    assert client.get("/v1/agent/runs/missing").status_code == 404
    assert client.post(f"/v1/agent/runs/{r['run_id']}/review",
                       json={"approved": True, "reviewer": "x"}).status_code == 409


def test_api_keys_and_rate_limit(settings):
    s = build_services(dataclasses.replace(settings, api_keys=["k-123"], rate_limit_per_minute=2))
    load_samples(s.rag)
    with TestClient(create_app(s)) as c:
        q = {"question": "Which job needs MLflow?"}
        assert c.post("/v1/ask", json=q).status_code == 401
        assert c.post("/v1/ask", json=q, headers={"x-api-key": "wrong"}).status_code == 401
        h = {"x-api-key": "k-123"}
        assert c.post("/v1/ask", json=q, headers=h).status_code == 200
        assert c.post("/v1/ask", json=q, headers=h).status_code == 200
        assert c.post("/v1/ask", json=q, headers=h).status_code == 429
