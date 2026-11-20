from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSpec:
    """Provider-neutral tool definition. `parameters` is a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 800
    json_mode: bool = False
    task: str = "general"  # used by the gateway's routing table


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str = "end"
