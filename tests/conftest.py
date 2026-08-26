"""Shared fixtures.

Every test gets its own database file. Tests that share state pass when run
together and fail when run alone, which is worse than no test at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from durable_agent.engine import Engine, StepContext
from durable_agent.models import Plan, RunSpec, Step
from durable_agent.providers import MockProvider
from durable_agent.store import SQLiteStore


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


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "runs.db")
    s.setup()
    return s


@pytest.fixture
def ledger() -> EffectLedger:
    return EffectLedger()


@pytest.fixture
def plan() -> Plan:
    return Plan(
        name="checkout",
        steps=(
            Step(name="parse"),
            Step(name="assess"),
            Step(name="charge", effect=True),
            Step(name="notify"),
        ),
    )


@pytest.fixture
def handlers(ledger: EffectLedger) -> dict:
    def parse(ctx: StepContext) -> dict:
        return {"amount": ctx.payload.get("amount", 0), "currency": "PEN"}

    def assess(ctx: StepContext) -> dict:
        answer = ctx.ask("Assess this transaction for risk.", Decision)
        return answer.model_dump()

    def charge(ctx: StepContext) -> dict:
        # The token is what makes this safe to replay.
        return {"receipt": ledger.charge(ctx.effect_token)}

    def notify(ctx: StepContext) -> dict:
        return {"sent": True}

    return {"parse": parse, "assess": assess, "charge": charge, "notify": notify}


@pytest.fixture
def engine(store: SQLiteStore, handlers: dict) -> Engine:
    return Engine(store=store, provider=MockProvider(), handlers=handlers)


@pytest.fixture
def spec(plan: Plan) -> RunSpec:
    return RunSpec(
        plan=plan,
        payload={"amount": 4200},
        idempotency_key="order-42-renewal",
    )


@pytest.fixture
def submitted(engine: Engine, spec: RunSpec) -> Iterator[str]:
    yield engine.submit(spec).run_id
