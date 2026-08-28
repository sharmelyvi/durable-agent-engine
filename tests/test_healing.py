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
    build_retry_prompt,
    correction_prompt,
    is_unparseable,
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


def test_a_correction_carries_the_fields_the_model_got_right() -> None:
    """The bug this replaced: a fault list names only what failed.

    ``decision`` validated, so it is absent from the faults. A correction built
    from faults alone asks for the full object while withholding the part that
    was already correct, and the model has to invent it — an invented value in a
    valid shape passes validation and is returned as fact.
    """
    answer = '{"decision":"approve","confidence":"very high"}'
    _, faults = parse_or_faults(answer, Decision)
    assert {f.field for f in faults} == {"confidence"}, "decision was valid, so it has no fault"

    retry = correction_prompt(answer, faults, "Decision")

    assert "approve" in retry, "the value that validated must survive the round"
    assert "confidence" in retry


def test_a_correction_does_not_resend_the_task() -> None:
    """Self-contained means the answer, not the question."""
    long_prompt = "Assess this transaction.\n" + ("history line\n" * 200)
    answer = '{"decision":"approve","confidence":"very high"}'
    _, faults = parse_or_faults(answer, Decision)

    retry = build_retry_prompt(long_prompt, answer, faults, "Decision")

    assert "Assess this transaction" not in retry
    assert len(retry) < len(long_prompt) / 5


def test_a_response_with_no_json_gets_the_task_back() -> None:
    """The one case a self-contained correction cannot serve.

    There is no draft to repair, so the only retry that can succeed is the
    original task with an explicit instruction about the format. The branch is
    about what is repairable, not about which prompt is shorter.
    """
    long_prompt = "Assess this transaction.\n" + ("history line\n" * 200)
    refusal = "I'm sorry, I can't help with that."
    _, faults = parse_or_faults(refusal, Decision)
    assert is_unparseable(faults)

    retry = build_retry_prompt(long_prompt, refusal, faults, "Decision")

    assert "Assess this transaction" in retry, "nothing else can carry the task"
    assert "no JSON" in retry


def test_the_expensive_retry_is_only_used_when_the_cheap_one_cannot_work() -> None:
    """Both directions, so the branch cannot quietly invert."""
    prompt = "Assess this.\n" + ("line\n" * 200)
    repairable = '{"decision":"approve","confidence":"very high"}'

    _, faults = parse_or_faults(repairable, Decision)
    cheap = build_retry_prompt(prompt, repairable, faults, "Decision")

    _, junk_faults = parse_or_faults("no json here at all", Decision)
    expensive = build_retry_prompt(prompt, "no json here at all", junk_faults, "Decision")

    assert len(cheap) < len(expensive)
    assert "Assess this" not in cheap and "Assess this" in expensive


def test_correction_rounds_stay_bounded() -> None:
    assert MAX_CORRECTION_ROUNDS == 2, (
        "a model that cannot satisfy a schema twice will not satisfy it ten times"
    )
