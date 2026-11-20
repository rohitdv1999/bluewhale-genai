import pytest
from pydantic import BaseModel

from bluewhale.llm.gateway import AllProvidersFailed, LLMGateway
from bluewhale.llm.providers.base import ProviderError
from bluewhale.llm.structured import StructuredOutputError, complete_structured, extract_json
from bluewhale.llm.types import LLMRequest, Message
from bluewhale.observability import metrics
from tests.conftest import ScriptedProvider

REQ = LLMRequest(messages=[Message("user", "hi")])


def gw(*providers, **kw):
    return LLMGateway(list(providers), sleep=lambda s: None, **kw)


def test_retries_retryable_errors_then_succeeds():
    p = ScriptedProvider([ProviderError("throttled", retryable=True), "ok"])
    resp = gw(p, max_retries=2).complete(REQ)
    assert resp.content == "ok" and len(p.requests) == 2


def test_hard_error_fails_over_without_retry():
    a = ScriptedProvider([ProviderError("auth", retryable=False)], name="a")
    b = ScriptedProvider(["from b"], name="b")
    resp = gw(a, b, max_retries=3).complete(REQ)
    assert resp.provider == "b" and len(a.requests) == 1


def test_all_failed_raises_with_every_error():
    a = ScriptedProvider([ProviderError("x", retryable=False)], name="a")
    b = ScriptedProvider([ProviderError("y", retryable=False)], name="b")
    with pytest.raises(AllProvidersFailed) as exc:
        gw(a, b).complete(REQ)
    assert len(exc.value.errors) == 2


def test_circuit_breaker_skips_failing_provider_until_cooldown():
    now = [0.0]
    a = ScriptedProvider([ProviderError("down", retryable=False)] * 3 + ["a back"], name="a")
    b = ScriptedProvider(["b"] * 5, name="b")
    g = gw(a, b, max_retries=0, clock=lambda: now[0])
    for _ in range(3):
        g.complete(REQ)
    calls = len(a.requests)
    g.complete(REQ)  # breaker open: a is skipped
    assert len(a.requests) == calls
    now[0] += 31
    assert g.complete(REQ).content == "a back"  # half-open probe succeeds


def test_routing_table_orders_providers_per_task():
    a = ScriptedProvider(["a"], name="a")
    b = ScriptedProvider(["b"], name="b")
    g = gw(a, b, routes={"extract": ["b", "a"]})
    assert g.complete(LLMRequest(messages=[Message("user", "x")], task="extract")).provider == "b"


def test_cost_and_token_accounting():
    p = ScriptedProvider(["ok"], name="bedrock")
    resp = gw(p).complete(REQ)
    assert resp.cost_usd == pytest.approx(10 / 1000 * 0.003 + 5 / 1000 * 0.015)
    assert metrics.counter("llm_tokens_total", provider="bedrock", kind="input") == 10


class Person(BaseModel):
    name: str
    years: int


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": "}"}\n```') == {"a": "}"}
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_structured_output_repairs_once():
    p = ScriptedProvider(['{"name": "Ana"}', '{"name": "Ana", "years": 4}'])
    out, _ = complete_structured(gw(p), [Message("user", "extract")], Person)
    assert out == Person(name="Ana", years=4)
    assert "did not match the schema" in p.requests[1].messages[-1].content


def test_structured_output_gives_up_after_repairs():
    p = ScriptedProvider(["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        complete_structured(gw(p), [Message("user", "extract")], Person)
