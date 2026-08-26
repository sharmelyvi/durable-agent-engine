"""Test doubles shared across modules.

Kept out of conftest.py on purpose: that file is pytest's, and importing from
it couples test modules to a collection mechanism rather than to a stated
contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """The schema a model answer has to satisfy to be accepted."""

    decision: str
    confidence: float = Field(ge=0.0, le=1.0)


class EffectLedger:
    """Records every external effect, so a test can prove one happened once.

    A counter would show how many times a step ran. Recording the token shows
    whether the outside world would have deduplicated them, which is the
    property that actually matters.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def charge(self, token: str) -> str:
        self.calls.append(token)
        return token

    @property
    def distinct(self) -> set[str]:
        return set(self.calls)
