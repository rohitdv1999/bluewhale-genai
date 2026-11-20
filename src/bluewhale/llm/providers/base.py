from __future__ import annotations

from typing import Protocol

from ..types import LLMRequest, LLMResponse


class ProviderError(Exception):
    """Raised by providers. `retryable` tells the gateway whether to retry
    the same provider (throttling, timeouts, 5xx) or fail over immediately
    (auth errors, bad requests, content filtered)."""

    def __init__(self, message: str, *, retryable: bool, provider: str = "") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider


class Provider(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest, *, timeout_s: float) -> LLMResponse: ...
