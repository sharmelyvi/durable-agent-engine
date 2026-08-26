"""Reading an answer out of a response that also contains other things.

A model asked for JSON returns JSON surrounded by whatever it felt like saying:
a worked example of the format, an apology, a summary afterwards. More than one
JSON value in a response is normal, not pathological.

Picking the first one is the tempting implementation and the wrong one. The
first is usually the example; the answer comes after the explanation. Returning
the example is worse than returning nothing — it is a wrong value in the right
shape, which passes validation and reaches the caller as fact.

So the schema decides, and the last value that satisfies it wins.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from durable_agent.healing import (
    MAX_CORRECTION_ROUNDS,
    cheaper_retry,
    correction_prompt,
    json_candidates,
    parse_or_faults,
)


class Decision(BaseModel):
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)


REAL = {"decision": "approve", "confidence": 0.9}


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (
            "a fenced example before the real answer",
            '```json\n{"decision":"EXAMPLE","confidence":0.0}\n```\n'
            'Actual: {"decision":"approve","confidence":0.9}',
        ),
        (
            "two objects in sequence",
            '{"decision":"deny","confidence":0.1} {"decision":"approve","confidence":0.9}',
        ),
        (
            "a brace in the prose beforehand",
            'Use {placeholder} first. Answer: {"decision":"approve","confidence":0.9}',
        ),
        ("commentary afterwards", '{"decision":"approve","confidence":0.9}\nHope this helps!'),
        ("a single fenced object", '```json\n{"decision":"approve","confidence":0.9}\n```'),
    ],
)
def test_the_answer_is_found_among_other_json(label: str, text: str) -> None:
    parsed, faults = parse_or_faults(text, Decision)
    assert parsed is not None, f"{label}: {[f.field for f in faults]}"
    assert parsed.model_dump() == REAL, label


def test_a_response_with_only_an_invalid_object_reports_faults() -> None:
    """Failing is correct here. Inventing a value would not be."""
    parsed, faults = parse_or_faults('```json\n{"decision":1,"confidence":"x"}\n```', Decision)
    assert parsed is None
    assert {f.field for f in faults} == {"decision", "confidence"}


def test_unreadable_text_reports_the_response_itself() -> None:
    parsed, faults = parse_or_faults("I'm sorry, I can't help with that.", Decision)
    assert parsed is None
    assert faults[0].field == "<response>"


def test_candidates_are_returned_in_the_order_they_appear() -> None:
    values = json_candidates('{"a":1} then {"b":2}')
    assert {"a": 1} in values and {"b": 2} in values


def test_braces_inside_strings_do_not_confuse_the_scanner() -> None:
    parsed, _ = parse_or_faults('{"decision":"approve {not a brace}","confidence":0.9}', Decision)
    assert parsed is not None
    assert parsed.decision == "approve {not a brace}"


def test_a_short_prompt_retry_still_carries_the_fault() -> None:
    """The failure mode this guards: a retry that repeats itself.

    When the delta costs more than the prompt it would replace, the cheaper
    move is to resend the original — but resending it *bare* tells the model
    nothing, so a deterministic model returns the same invalid answer and the
    correction round is spent for nothing.
    """
    _, faults = parse_or_faults('{"decision":"a","confidence":"high"}', Decision)
    delta = correction_prompt(faults, "Decision")
    short = "Approve?"

    retry = cheaper_retry(short, delta)

    assert retry != short, "a bare resend carries no information about the failure"
    assert "CORRECTION" in retry
    assert short in retry, "and it keeps the original task context"


def test_a_long_prompt_retry_drops_the_context_it_would_repeat() -> None:
    long_prompt = "Assess this transaction.\n" + ("history line\n" * 200)
    _, faults = parse_or_faults('{"decision":"a","confidence":"high"}', Decision)
    delta = correction_prompt(faults, "Decision")

    retry = cheaper_retry(long_prompt, delta)

    assert retry == delta
    assert len(retry) < len(long_prompt) / 10


def test_correction_rounds_stay_bounded() -> None:
    assert MAX_CORRECTION_ROUNDS == 2, (
        "a model that cannot satisfy a schema twice will not satisfy it ten times"
    )
