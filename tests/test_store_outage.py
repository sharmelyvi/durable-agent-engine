"""A database outage must not be mistaken for a broken run.

Two failures look similar from inside `run()` and mean opposite things.

A handler raising is a defect in this run's code. The run is marked failed,
because retrying it unchanged will fail again, and leaving it non-terminal would
block its idempotency key forever.

The store raising is infrastructure. Nothing is wrong with the run; the database
is simply unreachable. It must stay non-terminal so that it resumes and finishes
once the database returns. Marking it failed would convert a transient outage
into permanent loss for every run in flight at that moment — a far larger
failure than the outage.

The distinction currently rests on where a `try` block ends, which is easy to
undo while tidying. These tests are what makes the boundary load-bearing.
"""

from __future__ import annotations

import pytest

from durable_agent.engine import Engine, StepContext
from durable_agent.models import Plan, RunSpec, RunStatus, Step, StepOutcome
from durable_agent.providers import MockProvider
from durable_agent.store import SQLiteStore

PLAN = Plan(name="p", steps=(Step(name="charge", effect=True),))


class _Outage:
    """Wraps a store and refuses to write successful checkpoints.

    Failing *every* write would make this test unable to tell the two cases
    apart: with the whole store down, the error path cannot record anything
    either, so a run stays non-terminal whether that was intended or accidental.

    Refusing only the success-path write leaves the error path able to complete.
    If the engine ever treats a store outage as a run defect, it will manage to
    write FAILED — and these tests will see it.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.down = True
        self.refused = 0

    def append(self, record) -> None:
        if self.down and record.outcome is not StepOutcome.FAILED:
            self.refused += 1
            raise OSError("database is unreachable")
        self._inner.append(record)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


@pytest.fixture
def sqlite_only(store):
    if not isinstance(store, SQLiteStore):
        pytest.skip("the outage double wraps the SQLite store")
    return store


def test_a_store_outage_leaves_the_run_resumable(sqlite_only) -> None:
    charges: list[str] = []

    def charge(ctx: StepContext) -> dict:
        charges.append(ctx.effect_token)
        return {"ok": True}

    outage = _Outage(sqlite_only)
    engine = Engine(store=outage, provider=MockProvider(), handlers={"charge": charge})
    run = engine.submit(RunSpec(plan=PLAN, payload={}, idempotency_key="outage-run-00001"))

    with pytest.raises(OSError, match="unreachable"):
        engine.run(run.run_id, PLAN)

    state = sqlite_only.load(run.run_id)
    assert state.status is RunStatus.RUNNING, (
        "an unreachable database is not a broken run; marking it failed would "
        "make a transient outage permanent"
    )
    assert not state.status.is_terminal
    assert len(charges) == 1, "the effect did happen, it just was not recorded"
    assert outage.refused == 1
    assert len(state.steps) == 0, "no checkpoint was written, so resume repeats the step"


def test_the_run_finishes_once_the_store_returns(sqlite_only) -> None:
    """And the effect is not paid for twice, because the token is the same."""
    charges: list[str] = []

    def charge(ctx: StepContext) -> dict:
        charges.append(ctx.effect_token)
        return {"ok": True}

    outage = _Outage(sqlite_only)
    engine = Engine(store=outage, provider=MockProvider(), handlers={"charge": charge})
    run = engine.submit(RunSpec(plan=PLAN, payload={}, idempotency_key="recovers-run-001"))

    with pytest.raises(OSError):
        engine.run(run.run_id, PLAN)

    outage.down = False  # the database comes back
    final = engine.run(run.run_id, PLAN)

    assert final.status is RunStatus.COMPLETED
    assert len(charges) == 2, "the step genuinely ran again"
    assert len(set(charges)) == 1, "under one token, so the gateway charged once"


def test_a_handler_defect_is_still_terminal_during_normal_operation(sqlite_only) -> None:
    """The other side of the boundary, so the two cannot drift together."""

    def broken(ctx: StepContext) -> dict:
        raise ValueError("defect")

    engine = Engine(store=sqlite_only, provider=MockProvider(), handlers={"charge": broken})
    run = engine.submit(RunSpec(plan=PLAN, payload={}, idempotency_key="defect-vs-outage1"))

    with pytest.raises(ValueError):
        engine.run(run.run_id, PLAN)

    assert sqlite_only.load(run.run_id).status is RunStatus.FAILED
