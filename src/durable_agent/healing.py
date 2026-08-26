"""Repairing invalid model output without resending the whole prompt.

When a model returns something a schema rejects, the reflexive fix is to send
the original prompt again and hope. That pays full price for the context a
second time, and it gives the model no information about what was wrong.

This module does the opposite: it extracts the exact fields that failed and
sends only those back. The saving is measured in ``scripts/benchmark.py`` rather
than asserted here — see docs/BENCHMARK.md for the numbers and how to reproduce
them.

There is a hard limit on correction rounds. A model that cannot satisfy a
schema in two attempts is not going to satisfy it in ten, and the honest
response at that point is to escalate to a human, not to keep spending.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

MAX_CORRECTION_ROUNDS = 2

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class Unparseable(Exception):
    """The response contained nothing that could be read as JSON."""


@dataclass(frozen=True)
class FieldFault:
    """One thing the model got wrong, in the smallest form that describes it."""

    field: str
    problem: str
    received: Any

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "problem": self.problem, "received": self.received}


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a response that may be wrapped in prose.

    Models fence their JSON, apologise before it, and add commentary after it.
    Refusing to handle that is not strictness, it is just a fragile parser.
    """
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise Unparseable(text[:200])


def faults_from(error: ValidationError) -> list[FieldFault]:
    """Reduce a ValidationError to the minimum a model needs to fix it."""
    faults: list[FieldFault] = []
    for err in error.errors():
        faults.append(
            FieldFault(
                field=".".join(str(part) for part in err["loc"]) or "<root>",
                problem=err["msg"],
                received=err.get("input"),
            )
        )
    return faults


def correction_prompt(faults: list[FieldFault], schema_name: str) -> str:
    """The delta prompt: what was wrong, nothing else.

    It carries no task description and no original context, because the model
    is not being asked to redo the task — only to fix named fields.
    """
    payload = json.dumps([f.as_dict() for f in faults], ensure_ascii=False)
    return (
        f"CORRECTION for {schema_name}. Your previous answer failed validation.\n"
        f"Fix ONLY these fields and return the full corrected JSON object:\n{payload}"
    )


def parse_or_faults(text: str, schema: type[T]) -> tuple[T | None, list[FieldFault]]:
    """Validate a response. Returns the model, or the faults preventing it."""
    try:
        data = extract_json(text)
    except Unparseable:
        return None, [
            FieldFault(
                field="<response>",
                problem="response was not valid JSON",
                received=text[:120],
            )
        ]
    try:
        return schema.model_validate(data), []
    except ValidationError as exc:
        return None, faults_from(exc)
