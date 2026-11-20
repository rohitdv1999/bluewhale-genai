"""Azure OpenAI chat completions with tool calling.

Auth uses Entra ID (DefaultAzureCredential) when available, otherwise the
AZURE_OPENAI_API_KEY env var injected by the platform's secret store.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..types import LLMRequest, LLMResponse, Message, ToolCall, Usage
from .base import ProviderError


class AzureOpenAIProvider:
    name = "azure_openai"

    def __init__(self, endpoint: str, deployment: str, api_version: str,
                 client: Any | None = None) -> None:
        self.model = deployment
        self._endpoint = endpoint
        self._api_version = api_version
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai not installed (pip install .[azure])",
                                retryable=False, provider=self.name) from exc
        kwargs: dict[str, Any] = {"azure_endpoint": self._endpoint,
                                  "api_version": self._api_version, "max_retries": 0}
        if os.environ.get("AZURE_OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        else:  # pragma: no cover - depends on azure-identity
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        self._client = AzureOpenAI(**kwargs)
        return self._client

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append({"role": "assistant", "content": m.content or None, "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in m.tool_calls]})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def complete(self, request: LLMRequest, *, timeout_s: float) -> LLMResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout": timeout_s,
        }
        if request.tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools]
        if request.json_mode and not request.tools:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 409, 429) or (status is not None and status >= 500) \
                or type(exc).__name__ in ("APITimeoutError", "APIConnectionError")
            raise ProviderError(f"azure_openai {status}: {exc}", retryable=retryable,
                                provider=self.name) from exc

        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise ProviderError("content filtered", retryable=False, provider=self.name)
        calls = [ToolCall(id=tc.id, name=tc.function.name,
                          arguments=json.loads(tc.function.arguments or "{}"))
                 for tc in (choice.message.tool_calls or [])]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=calls,
            usage=Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens),
            model=self.model,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            stop_reason=choice.finish_reason or "end",
        )
