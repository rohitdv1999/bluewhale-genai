"""Tool registry for agents.

Each tool declares a Pydantic argument model (which doubles as the JSON
schema sent to the model), a permission level and whether a human must
approve it. The registry validates arguments before running anything, so a
model that hallucinates a field or a type gets a structured error back
instead of a stack trace - and nothing executes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from ..llm.types import ToolSpec
from ..observability import metrics, trace


class Permission(StrEnum):
    READ = "read"          # reads our own data
    WRITE = "write"        # changes our own data
    EXTERNAL = "external"  # sends something outside the system (email, ATS, ...)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    permission: Permission = Permission.READ
    requires_approval: bool = False
    timeout_s: float = 10.0

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(self.name, self.description, schema)


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0


class ToolRegistry:
    def __init__(self, allowed: set[Permission] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        # which permission levels this registry instance may run at all
        self.allowed = allowed or {Permission.READ, Permission.WRITE, Permission.EXTERNAL}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values() if t.permission in self.allowed]

    def names(self) -> list[str]:
        return list(self._tools)

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, error=f"unknown tool '{name}'. available: {self.names()}")
        if tool.permission not in self.allowed:
            return ToolResult(False, error=f"tool '{name}' is not permitted in this context")
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(False, error=f"invalid arguments: {exc.errors(include_url=False)}")
        started = time.perf_counter()
        with trace("agent.tool", tool=name):
            try:
                out = tool.fn(**args.model_dump())
            except Exception as exc:  # tool failures are reported to the model, not raised
                metrics.inc("tool_errors_total", tool=name)
                return ToolResult(False, error=f"{type(exc).__name__}: {exc}",
                                  duration_ms=(time.perf_counter() - started) * 1000)
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > tool.timeout_s * 1000:
            metrics.inc("tool_slow_total", tool=name)
        metrics.inc("tool_calls_total", tool=name)
        return ToolResult(True, out, duration_ms=elapsed)
