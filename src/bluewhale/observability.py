"""Tracing, metrics and structured logs.

Spans are recorded in-process (so tests and the /metrics endpoint can read
them) and, when the OpenTelemetry SDK is installed and BW_OTEL=1, mirrored to
an OTLP exporter. Every LLM call records model, tokens, latency and cost.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bluewhale")

_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bw_trace_id", default=None
)

_otel_tracer: Any = None
if os.environ.get("BW_OTEL") == "1":  # pragma: no cover - optional dependency
    try:
        from opentelemetry import trace as _otel_trace

        _otel_tracer = _otel_trace.get_tracer("bluewhale")
    except ImportError:
        _otel_tracer = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "msg": record.getMessage(),
            "trace_id": _current_trace.get(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("bluewhale")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000


class Metrics:
    """Thread-safe counters and latency histograms with Prometheus text output."""

    _BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self.hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
        self.spans: deque[Span] = deque(maxlen=2000)

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[self._key(name, labels)] += value

    def observe(self, name: str, value_ms: float, **labels: str) -> None:
        with self._lock:
            series = self.hist[self._key(name, labels)]
            series.append(value_ms)
            if len(series) > 5000:
                del series[:1000]

    def counter(self, name: str, **labels: str) -> float:
        return self.counters.get(self._key(name, labels), 0.0)

    def percentile(self, name: str, pct: float, **labels: str) -> float | None:
        values = sorted(self.hist.get(self._key(name, labels), []))
        if not values:
            return None
        idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
        return values[idx]

    def prometheus(self) -> str:
        lines: list[str] = []

        def fmt(labels: tuple[tuple[str, str], ...]) -> str:
            if not labels:
                return ""
            return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"

        with self._lock:
            for (name, labels), value in sorted(self.counters.items()):
                lines.append(f"bw_{name}{fmt(labels)} {value:g}")
            for (name, labels), values in sorted(self.hist.items()):
                for b in self._BUCKETS:
                    le = labels + (("le", str(b)),)
                    lines.append(f"bw_{name}_bucket{fmt(le)} {sum(1 for v in values if v <= b)}")
                lines.append(f"bw_{name}_count{fmt(labels)} {len(values)}")
                lines.append(f"bw_{name}_sum{fmt(labels)} {sum(values):.3f}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.hist.clear()
            self.spans.clear()


metrics = Metrics()


def new_trace_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def trace(name: str, **attrs: Any) -> Iterator[Span]:
    """Open a span. Nested spans share the caller's trace id."""
    trace_id = _current_trace.get()
    token = None
    if trace_id is None:
        trace_id = new_trace_id()
        token = _current_trace.set(trace_id)
    span = Span(name=name, trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                start=time.perf_counter(), attrs=dict(attrs))
    otel_cm = _otel_tracer.start_as_current_span(name) if _otel_tracer else None
    otel_span = otel_cm.__enter__() if otel_cm else None
    try:
        yield span
    except Exception as exc:
        span.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        span.end = time.perf_counter()
        metrics.spans.append(span)
        metrics.observe("span_latency_ms", span.duration_ms, span=name)
        if span.error:
            metrics.inc("span_errors_total", span=name)
        if otel_span is not None:
            for k, v in span.attrs.items():
                if isinstance(v, str | int | float | bool):
                    otel_span.set_attribute(f"bw.{k}", v)
            otel_cm.__exit__(None, None, None)
        log.debug("span", extra={"fields": {"span": name, "ms": round(span.duration_ms, 2),
                                             "error": span.error, **span.attrs}})
        if token is not None:
            _current_trace.reset(token)


def current_trace_id() -> str | None:
    return _current_trace.get()
