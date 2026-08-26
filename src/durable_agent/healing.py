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


def json_candidates(text: str) -> list[Any]:
    """Every JSON value that can be read out of a response, in the order found.

    Models fence their JSON, apologise before it, and add commentary after it.
    Refusing to handle that is not strictness, just a fragile parser.

    Returning every candidate rather than the first one matters: a response can
    contain more than one JSON object, and the first is often an example of the
    format rather than the answer. Deciding between them needs the schema, which
    lives one function up.
    """
    seen: list[str] = []
    for fenced in _FENCE.finditer(text):
        seen.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        seen.append(text[start : end + 1])
        # The widest span fails when a response holds two objects, so also try
        # each brace-balanced region on its own.
        seen.extend(_balanced_objects(text))
    seen.append(text.strip())

    values: list[Any] = []
    for candidate in seen:
        try:
            values.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    if not values:
        raise Unparseable(text[:200])
    return values


def _balanced_objects(text: str) -> list[str]:
    """Each top-level {...} region, found by counting braces."""
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start != -1:
                out.append(text[start : i + 1])
    return out


def extract_json(text: str) -> Any:
    """The first readable JSON value. Prefer ``json_candidates`` with a schema."""
    return json_candidates(text)[0]


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


def cheaper_retry(original: str, correction: str) -> str:
    """Pick the cheaper retry, but never one carrying no information.

    Delta correction is not universally cheaper. On a short prompt the fault
    description is longer than the prompt it replaces — measured at -105% in
    scripts/benchmark.py, which is what this function exists for.

    The fallback is not the bare original, though. Resending the exact prompt
    that just failed tells the model nothing about what was wrong, so a
    deterministic model returns the same invalid answer and the correction round
    is spent for nothing. The original plus a short fault note is still cheaper
    than a full delta at that size, and unlike a bare resend it can actually
    succeed.
    """
    if len(correction) < len(original):
        return correction
    return f"{original}\n\n{correction}"


def parse_or_faults(text: str, schema: type[T]) -> tuple[T | None, list[FieldFault]]:
    """Validate a response against a schema, or report what stopped it.

    When a response holds several JSON values — a worked example followed by the
    real answer is the common shape — the schema decides between them, and the
    *last* one that validates wins. Models put their answer after their
    explanation, and taking the first match would return the example: a wrong
    value wearing the right shape, which is worse than no value at all.
    """
    try:
        values = json_candidates(text)
    except Unparseable:
        return None, [
            FieldFault(
                field="<response>",
                problem="response was not valid JSON",
                received=text[:120],
            )
        ]

    last_error: ValidationError | None = None
    for data in reversed(values):
        try:
            return schema.model_validate(data), []
        except ValidationError as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover - json_candidates never returns empty
        raise Unparseable(text[:200])
    return None, faults_from(last_error)
