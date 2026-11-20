"""AWS Bedrock via the Converse API (one request shape for every Bedrock model).

Credentials come from the standard AWS chain (IAM role on ECS, SSO profile
locally). Optional Guardrails are attached with BW_BEDROCK_GUARDRAIL_ID.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..types import LLMRequest, LLMResponse, Message, ToolCall, Usage
from .base import ProviderError

_RETRYABLE = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
    "ModelNotReadyException",
}


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str, region: str, client: Any | None = None) -> None:
        self.model = model
        self.region = region
        self._client = client
        self._guardrail_id = os.environ.get("BW_BEDROCK_GUARDRAIL_ID", "")
        self._guardrail_version = os.environ.get("BW_BEDROCK_GUARDRAIL_VERSION", "DRAFT")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("boto3 not installed (pip install .[aws])",
                                    retryable=False, provider=self.name) from exc
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(retries={"max_attempts": 1}, read_timeout=60),
            )
        return self._client

    @staticmethod
    def _to_converse(messages: list[Message]) -> tuple[list[dict], list[dict]]:
        system: list[dict] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                system.append({"text": m.content})
            elif m.role == "tool":
                block = {"toolResult": {"toolUseId": m.tool_call_id,
                                        "content": [{"text": m.content}]}}
                # Bedrock expects tool results inside a user turn.
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant" and m.tool_calls:
                content: list[dict] = [{"text": m.content}] if m.content else []
                content += [{"toolUse": {"toolUseId": c.id, "name": c.name, "input": c.arguments}}
                            for c in m.tool_calls]
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m.role, "content": [{"text": m.content}]})
        return system, out

    def complete(self, request: LLMRequest, *, timeout_s: float) -> LLMResponse:
        client = self._get_client()
        system, messages = self._to_converse(request.messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {"maxTokens": request.max_tokens,
                                "temperature": request.temperature},
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["toolConfig"] = {"tools": [
                {"toolSpec": {"name": t.name, "description": t.description,
                              "inputSchema": {"json": t.parameters}}}
                for t in request.tools
            ]}
        if self._guardrail_id:
            kwargs["guardrailConfig"] = {"guardrailIdentifier": self._guardrail_id,
                                         "guardrailVersion": self._guardrail_version}
        started = time.perf_counter()
        try:
            resp = client.converse(**kwargs)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
            raise ProviderError(f"bedrock {code}: {exc}", retryable=code in _RETRYABLE,
                                provider=self.name) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp["output"]["message"]["content"]:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                calls.append(ToolCall(id=tu["toolUseId"], name=tu["name"], arguments=tu["input"]))
        usage = resp.get("usage", {})
        if resp.get("stopReason") == "guardrail_intervened":
            raise ProviderError("blocked by guardrail", retryable=False, provider=self.name)
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=calls,
            usage=Usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0)),
            model=self.model,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000,
            stop_reason=resp.get("stopReason", "end"),
        )
