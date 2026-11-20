"""Bounded tool-calling agent with human approval and checkpoints.

The loop is deliberately small and explicit:

    model -> (tool calls?) -> validate -> [approval gate] -> run -> model -> ...

Limits that keep it predictable in production:
* `max_steps` model turns per run, then the run stops with status `max_steps`
* the same tool call (name + args) twice in a row ends the run (loop guard)
* tool errors go back to the model as data so it can correct itself, up to
  `max_tool_errors` per run
* any tool marked `requires_approval` pauses the run (`awaiting_approval`);
  state is checkpointed and the run resumes from exactly that point once a
  reviewer approves or rejects
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..llm.gateway import AllProvidersFailed, LLMGateway
from ..llm.prompts import get_prompt
from ..llm.types import LLMRequest, Message, ToolCall
from ..observability import metrics, trace
from .tools import ToolRegistry

RUNNING, AWAITING, DONE, FAILED, MAX_STEPS = (
    "running", "awaiting_approval", "completed", "failed", "max_steps")


@dataclass
class AgentRun:
    run_id: str
    goal: str
    messages: list[Message] = field(default_factory=list)
    status: str = RUNNING
    steps: int = 0
    tool_errors: int = 0
    pending: ToolCall | None = None
    final_answer: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentRun:
        msgs = [Message(m["role"], m["content"], [ToolCall(**c) for c in m["tool_calls"]],
                        m.get("tool_call_id"), m.get("name")) for m in d["messages"]]
        pending = ToolCall(**d["pending"]) if d.get("pending") else None
        return cls(d["run_id"], d["goal"], msgs, d["status"], d["steps"], d["tool_errors"],
                   pending, d["final_answer"], d["events"], d.get("cost_usd", 0.0))


class CheckpointStore(Protocol):
    def save(self, run: AgentRun) -> None: ...

    def load(self, run_id: str) -> AgentRun | None: ...


class MemoryCheckpoints:
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, run: AgentRun) -> None:
        with self._lock:
            self._runs[run.run_id] = json.loads(json.dumps(run.to_dict(), default=str))

    def load(self, run_id: str) -> AgentRun | None:
        with self._lock:
            d = self._runs.get(run_id)
        return AgentRun.from_dict(d) if d else None


class FileCheckpoints:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: AgentRun) -> None:
        tmp = self.root / f"{run.run_id}.json.tmp"
        tmp.write_text(json.dumps(run.to_dict(), default=str))
        tmp.replace(self.root / f"{run.run_id}.json")  # atomic on POSIX

    def load(self, run_id: str) -> AgentRun | None:
        p = self.root / f"{run_id}.json"
        return AgentRun.from_dict(json.loads(p.read_text())) if p.exists() else None


class AgentRunner:
    def __init__(self, gateway: LLMGateway, tools: ToolRegistry,
                 checkpoints: CheckpointStore | None = None, *, max_steps: int = 8,
                 max_tool_errors: int = 3) -> None:
        self.gateway = gateway
        self.tools = tools
        self.checkpoints = checkpoints or MemoryCheckpoints()
        self.max_steps = max_steps
        self.max_tool_errors = max_tool_errors

    def start(self, goal: str) -> AgentRun:
        system, user = get_prompt("agent_system").render(goal=goal)
        run = AgentRun(uuid.uuid4().hex[:12], goal,
                       [Message("system", system), Message("user", user)])
        metrics.inc("agent_runs_total")
        return self._loop(run)

    def resume(self, run_id: str, *, approved: bool, reviewer: str, note: str = "") -> AgentRun:
        run = self.checkpoints.load(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.status != AWAITING or run.pending is None:
            raise ValueError(f"run {run_id} is not waiting for approval (status={run.status})")
        call, run.pending, run.status = run.pending, None, RUNNING
        run.events.append({"type": "approval", "tool": call.name, "approved": approved,
                           "reviewer": reviewer, "note": note})
        metrics.inc("agent_approvals_total", approved=str(approved).lower())
        if approved:
            self._execute(run, call)
        else:
            run.messages.append(Message("tool", json.dumps(
                {"error": "rejected by human reviewer", "note": note}),
                tool_call_id=call.id, name=call.name))
        return self._loop(run)

    def get(self, run_id: str) -> AgentRun | None:
        return self.checkpoints.load(run_id)

    # ------------------------------------------------------------------
    def _execute(self, run: AgentRun, call: ToolCall) -> None:
        result = self.tools.run(call.name, call.arguments)
        if not result.ok:
            run.tool_errors += 1
        payload = {"ok": result.ok, "output": result.output} if result.ok \
            else {"ok": False, "error": result.error}
        run.events.append({"type": "tool", "tool": call.name, "args": call.arguments,
                           "ok": result.ok, "ms": round(result.duration_ms, 1)})
        run.messages.append(Message("tool", json.dumps(payload, default=str)[:6000],
                                    tool_call_id=call.id, name=call.name))

    def _finish(self, run: AgentRun, status: str, answer: str = "") -> AgentRun:
        run.status = status
        run.final_answer = answer or run.final_answer
        metrics.inc("agent_runs_finished_total", status=status)
        self.checkpoints.save(run)
        return run

    def _loop(self, run: AgentRun) -> AgentRun:
        last_call: tuple[str, str] | None = None
        with trace("agent.run", run_id=run.run_id):
            while run.steps < self.max_steps:
                run.steps += 1
                try:
                    resp = self.gateway.complete(LLMRequest(
                        messages=run.messages, tools=self.tools.specs(), task="agent"))
                except AllProvidersFailed as exc:
                    run.events.append({"type": "error", "error": str(exc)[:500]})
                    return self._finish(run, FAILED)
                run.cost_usd += resp.cost_usd
                run.messages.append(Message("assistant", resp.content, resp.tool_calls))

                if not resp.tool_calls:
                    return self._finish(run, DONE, resp.content)

                for i, call in enumerate(resp.tool_calls):
                    signature = (call.name, json.dumps(call.arguments, sort_keys=True))
                    if signature == last_call:
                        run.events.append({"type": "loop_guard", "tool": call.name})
                        return self._finish(run, FAILED, "Stopped: repeated identical tool call.")
                    last_call = signature

                    tool = self.tools.get(call.name)
                    if tool is not None and tool.requires_approval:
                        run.pending = call
                        run.events.append({"type": "approval_requested", "tool": call.name,
                                           "args": call.arguments})
                        # every tool call needs a result; later calls in this turn are deferred
                        for later in resp.tool_calls[i + 1:]:
                            run.messages.append(Message("tool", json.dumps(
                                {"ok": False, "error": "deferred: waiting for approval"}),
                                tool_call_id=later.id, name=later.name))
                        return self._finish(run, AWAITING)
                    self._execute(run, call)

                if run.tool_errors > self.max_tool_errors:
                    return self._finish(run, FAILED, "Stopped: too many tool errors.")
            return self._finish(run, MAX_STEPS, "Stopped: step limit reached.")
