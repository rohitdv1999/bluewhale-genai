"""Multi-provider LLM gateway.

One entry point (`LLMGateway.complete`) in front of Bedrock, Azure OpenAI and
the offline provider. It handles:

* routing   - a task -> ordered provider chain table, so e.g. "extract" can
              prefer a small cheap model while "answer" prefers a larger one
* retries   - exponential backoff with full jitter, only for retryable errors
* failover  - next provider in the chain on hard errors or an open breaker
* breakers  - per-provider circuit breaker so a failing region is skipped
              quickly instead of eating the latency budget on every call
* accounting- tokens, latency and estimated cost per call, as metrics + spans
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..observability import log, metrics, trace
from .providers.base import Provider, ProviderError
from .types import LLMRequest, LLMResponse

# USD per 1K tokens (input, output). Defaults for budgeting dashboards only;
# set real numbers per deployment with `LLMGateway.set_price`.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "local": (0.0, 0.0),
    "bedrock": (0.003, 0.015),
    "azure_openai": (0.0025, 0.01),
}


class AllProvidersFailed(RuntimeError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("all providers failed: " + " | ".join(errors))
        self.errors = errors


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_after_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        # half-open: let one probe through after the cool-down
        return now - self.opened_at >= self.reset_after_s

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = now


class LLMGateway:
    def __init__(
        self,
        providers: list[Provider],
        *,
        routes: dict[str, list[str]] | None = None,
        max_retries: int = 2,
        timeout_s: float = 30.0,
        base_backoff_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("gateway needs at least one provider")
        self.providers = {p.name: p for p in providers}
        self.default_chain = [p.name for p in providers]
        self.routes = routes or {}
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.base_backoff_s = base_backoff_s
        self._sleep = sleep
        self._clock = clock
        self._breakers = {name: CircuitBreaker() for name in self.providers}
        self._prices = dict(DEFAULT_PRICES)
        self._lock = threading.Lock()

    def set_price(self, provider: str, input_per_1k: float, output_per_1k: float) -> None:
        self._prices[provider] = (input_per_1k, output_per_1k)

    def chain_for(self, task: str) -> list[str]:
        chain = self.routes.get(task, self.default_chain)
        return [name for name in chain if name in self.providers]

    def _cost(self, resp: LLMResponse) -> float:
        p_in, p_out = self._prices.get(resp.provider, (0.0, 0.0))
        return round(resp.usage.input_tokens / 1000 * p_in
                     + resp.usage.output_tokens / 1000 * p_out, 6)

    def _backoff(self, attempt: int) -> float:
        return random.uniform(0, self.base_backoff_s * (2 ** attempt))

    def complete(self, request: LLMRequest) -> LLMResponse:
        errors: list[str] = []
        with trace("llm.complete", task=request.task) as span:
            for name in self.chain_for(request.task):
                provider = self.providers[name]
                breaker = self._breakers[name]
                with self._lock:
                    allowed = breaker.allow(self._clock())
                if not allowed:
                    errors.append(f"{name}: circuit open")
                    metrics.inc("llm_skipped_total", provider=name, reason="circuit_open")
                    continue

                for attempt in range(self.max_retries + 1):
                    try:
                        resp = provider.complete(request, timeout_s=self.timeout_s)
                    except ProviderError as exc:
                        metrics.inc("llm_errors_total", provider=name,
                                    retryable=str(exc.retryable).lower())
                        errors.append(f"{name}#{attempt}: {exc}")
                        with self._lock:
                            breaker.record_failure(self._clock())
                        if exc.retryable and attempt < self.max_retries:
                            self._sleep(self._backoff(attempt))
                            continue
                        break  # hard error or retries exhausted -> next provider
                    with self._lock:
                        breaker.record_success()
                    resp.cost_usd = self._cost(resp)
                    span.attrs.update(provider=name, model=resp.model,
                                      input_tokens=resp.usage.input_tokens,
                                      output_tokens=resp.usage.output_tokens,
                                      cost_usd=resp.cost_usd, attempts=attempt + 1)
                    metrics.inc("llm_requests_total", provider=name, task=request.task)
                    metrics.inc("llm_tokens_total", resp.usage.input_tokens,
                                provider=name, kind="input")
                    metrics.inc("llm_tokens_total", resp.usage.output_tokens,
                                provider=name, kind="output")
                    metrics.inc("llm_cost_usd_total", resp.cost_usd, provider=name)
                    metrics.observe("llm_latency_ms", resp.latency_ms, provider=name)
                    return resp
            log.warning("llm gateway exhausted", extra={"fields": {"errors": errors}})
            raise AllProvidersFailed(errors)


def build_gateway(settings: Settings | None = None) -> LLMGateway:
    """Build the gateway from settings. Unknown or unconfigured providers are
    skipped; the offline provider is always appended as the last resort in
    local mode so the demo never hard-fails."""
    s = settings or get_settings()
    providers: list[Provider] = []
    for name in s.llm_providers:
        if name == "bedrock":
            from .providers.bedrock import BedrockProvider

            providers.append(BedrockProvider(s.bedrock_model_id, s.aws_region))
        elif name == "azure_openai" and s.azure_openai_endpoint:
            from .providers.azure_openai import AzureOpenAIProvider

            providers.append(AzureOpenAIProvider(s.azure_openai_endpoint,
                                                 s.azure_openai_deployment,
                                                 s.azure_openai_api_version))
        elif name == "local":
            from .providers.local import LocalExtractiveProvider

            providers.append(LocalExtractiveProvider())
    if s.is_local and not any(p.name == "local" for p in providers):
        from .providers.local import LocalExtractiveProvider

        providers.append(LocalExtractiveProvider())
    return LLMGateway(providers, max_retries=s.llm_max_retries, timeout_s=s.llm_timeout_s)
