"""Shared fixtures.

The `store` fixture is parametrised over both backends, so every invariant in
this suite is asserted twice: once against SQLite and once against PostgreSQL.
Testing each backend with its own bespoke tests would prove that each one works
and leave the interesting question — whether they behave the same where it
matters — unasked.

Postgres tests skip when no server is reachable, so `pytest` works on a clean
checkout and `make test-postgres` exercises the full matrix.

Every test gets isolated state: a fresh file for SQLite, a truncated schema for
Postgres. Tests that share state pass together and fail alone, which is worse
than no test.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from durable_agent.engine import Engine, StepContext
from durable_agent.models import Plan, RunSpec, Step
from durable_agent.providers import MockProvider
from durable_agent.store import SQLiteStore

from .support import Decision, EffectLedger


def postgres_available() -> bool:
    """True only if psycopg is installed AND a server answers.

    Imported lazily: psycopg is an optional extra, and a clean install without
    it must still be able to run the SQLite half of this suite.
    """
    try:
        import psycopg

        from durable_agent.postgres import dsn_from_env

        with psycopg.connect(dsn_from_env(), connect_timeout=2):
            return True
    except Exception:
        return False


HAVE_POSTGRES = postgres_available()

_BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param(
        "postgres",
        id="postgres",
        marks=pytest.mark.skipif(
            not HAVE_POSTGRES,
            reason="no PostgreSQL reachable; run `docker compose up -d` first",
        ),
    ),
]


@pytest.fixture(params=_BACKENDS)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator:
    if request.param == "sqlite":
        sqlite_store = SQLiteStore(tmp_path / "runs.db")
        sqlite_store.setup()
        yield sqlite_store
        return

    from durable_agent.postgres import PostgresStore

    # A schema per test rather than TRUNCATE on a shared one. Truncation is
    # fragile isolation: one lost commit or one leaked connection and a test
    # inherits another's rows, which shows up as a foreign-key violation far
    # from the cause. A private schema cannot be contaminated.
    pg = PostgresStore(schema=f"t_{uuid.uuid4().hex[:16]}")
    pg.setup()
    yield pg
    pg.drop()


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
