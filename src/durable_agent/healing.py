"""Repairing invalid model output without resending the whole prompt.

When a model returns something a schema rejects, the reflexive fix is to send
the original prompt again and hope. That pays full price for the context a
second time, and it gives the model no information about what was wrong.

This module sends back the answer and its faults instead of the task. That is
self-contained — the model repairs a draft it can see — and it costs a fraction
of the context it replaces. The saving is measured in ``scripts/benchmark.py`` rather
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

# The fault raised when a response held no JSON at all. Named because the
# retry strategy branches on it: there is nothing to repair from.
UNPARSEABLE_FIELD = "<response>"

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


def correction_prompt(previous: str, faults: list[FieldFault], schema_name: str) -> str:
    """The delta prompt: the previous answer, and exactly what was wrong with it.

    Carrying the previous answer is what makes this self-contained, and leaving
    it out was a silent data-loss bug rather than an economy.

    A fault list only names the fields that *failed*. A field that validated is
    correct and therefore absent — so a prompt built from faults alone asks the
    model to "return the full corrected object" while withholding the parts of
    that object it got right. The model cannot recover them and has to invent
    them, and an invented value in a valid shape passes validation and reaches
    the caller as fact. On a decision schema that is the difference between
    approve and deny, decided by a guess.

    The previous answer costs a few dozen tokens and removes the guess. What it
    still does not carry is the task, which is deliberate: schema validation
    catches shape, and shape can be repaired from the answer alone.
    """
    payload = json.dumps([f.as_dict() for f in faults], ensure_ascii=False)
    return (
        f"CORRECTION for {schema_name}.\n"
        f"Your previous response was:\n{previous.strip()}\n\n"
        f"It failed validation on these fields:\n{payload}\n\n"
        "Fix ONLY the invalid fields, keep every valid field exactly as it was, "
        "and return the complete corrected JSON object."
    )


def is_unparseable(faults: list[FieldFault]) -> bool:
    """Whether the response held no JSON at all, rather than the wrong JSON."""
    return any(f.field == UNPARSEABLE_FIELD for f in faults)


def build_retry_prompt(
    original: str, previous: str, faults: list[FieldFault], schema_name: str
) -> str:
    """The retry to send, chosen by what can be repaired rather than by size.

    A response that failed validation is still a draft: it holds the fields the
    model got right, so the cheap self-contained correction can repair it and
    the task never has to be resent.

    A response that held no JSON is not a draft. There is nothing in it to
    preserve and nothing to correct, so the only thing that can succeed is the
    original task with an explicit instruction about the format. That is the
    expensive path, and it is the right one exactly when the cheap path cannot
    work — which is the distinction an earlier version of this got wrong by
    choosing on token count instead.
    """
    if is_unparseable(faults):
        return (
            f"{original}\n\n"
            f"ERROR: your previous response contained no JSON. Return only a JSON "
            f"object matching {schema_name}, with no surrounding text."
        )
    return correction_prompt(previous, faults, schema_name)


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
                field=UNPARSEABLE_FIELD,
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
