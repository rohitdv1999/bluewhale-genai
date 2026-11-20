from __future__ import annotations

import pytest

from bluewhale.config import Settings
from bluewhale.container import build_services, load_samples
from bluewhale.llm.types import LLMRequest, LLMResponse, Usage
from bluewhale.observability import metrics


class ScriptedProvider:
    """Returns queued responses in order; records every request it receives."""

    def __init__(self, responses, name: str = "scripted") -> None:
        self.name = name
        self.model = f"{name}-model"
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request, *, timeout_s):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            item.provider = self.name
            return item
        return LLMResponse(content=item, usage=Usage(10, 5), model=self.model, provider=self.name)


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    yield


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def services(settings):
    s = build_services(settings)
    load_samples(s.rag)
    return s
