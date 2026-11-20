"""Structured outputs: ask for JSON, validate with Pydantic, repair once.

Models occasionally wrap JSON in prose or code fences, or miss a field. We
pull the first JSON object out of the reply, validate it, and on failure send
one repair turn that includes the validation error. Anything still invalid
raises - callers decide whether to degrade or surface the error.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..observability import metrics
from .gateway import LLMGateway
from .types import LLMRequest, LLMResponse, Message

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class StructuredOutputError(ValueError):
    pass


def extract_json(text: str) -> dict:
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise StructuredOutputError("no JSON object in model output")
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(candidate[start:], start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start: i + 1])
    raise StructuredOutputError("unterminated JSON object in model output")


def complete_structured(
    gateway: LLMGateway,
    messages: list[Message],
    schema: type[T],
    *,
    task: str = "extract",
    max_repairs: int = 1,
) -> tuple[T, LLMResponse]:
    convo = list(messages)
    last_error = ""
    for attempt in range(max_repairs + 1):
        resp = gateway.complete(LLMRequest(messages=convo, json_mode=True, task=task))
        try:
            return schema.model_validate(extract_json(resp.content)), resp
        except (StructuredOutputError, ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            metrics.inc("structured_output_failures_total", schema=schema.__name__)
            if attempt == max_repairs:
                break
            convo = convo + [
                Message("assistant", resp.content),
                Message("user", "That response did not match the schema. Error:\n"
                        f"{last_error[:800]}\nReturn only the corrected JSON."),
            ]
    raise StructuredOutputError(f"{schema.__name__}: {last_error}")
