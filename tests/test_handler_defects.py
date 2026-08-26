"""A defect in a handler must not poison an idempotency key.

`Escalation` is the engine's word for "I gave up in a way a human should look
at". Anything else a handler raises is a defect: a typo, a bad cast, a library
throwing something undocumented.

If such a run were left in `running`, the partial unique index — which permits
exactly one non-terminal run per idempotency key — would refuse every future
submission for that key. One bad handler would make that customer's order
permanently unprocessable, with no recovery short of editing the database by
hand. That is a far worse failure than the original bug.

So an unexpected exception marks the run FAILED and re-raises. The invariant
holds, the key is freed, and the traceback still reaches whoever has to fix it.
"""

from __future__ import annotations

import pytest

from durable_agent.engine import Engine, SimulatedCrash, StepContext
from durable_agent.models import Plan, RunSpec, RunStatus, Step, StepOutcome
from durable_agent.providers import MockProvider

PLAN = Plan(name="p", steps=(Step(name="boom"),))


def _engine(store, handler) -> Engine:
    return Engine(store=store, provider=MockProvider(), handlers={"boom": handler})


def test_an_unexpected_exception_reaches_a_terminal_state(store) -> None:
    def boom(ctx: StepContext) -> dict:
        raise ValueError("a typo, not a business decision")

    engine = _engine(store, boom)
    run = engine.submit(RunSpec(plan=PLAN, payload={}, idempotency_key="defect-run-0001"))

    with pytest.raises(ValueError, match="a typo"):
        engine.run(run.run_id, PLAN)

    final = store.load(run.run_id)
    assert final.status is RunStatus.FAILED, "I-4: every run reaches a terminal state"
    assert final.status.is_terminal
    assert "ValueError" in (final.error or ""), "the record names what broke"


def test_the_idempotency_key_survives_a_handler_defect(store) -> None:
    """The consequence that makes this more than a tidiness issue."""

    def boom(ctx: StepContext) -> dict:
        raise RuntimeError("kaboom")

    engine = _engine(store, boom)
    spec = RunSpec(plan=PLAN, payload={}, idempotency_key="reusable-key-001")

    first = engine.submit(spec)
    with pytest.raises(RuntimeError):
        engine.run(first.run_id, PLAN)

    # Once the defect is fixed, the same order must be submittable again.
    second = engine.submit(spec)
    assert second.run_id != first.run_id


def test_the_failed_step_is_recorded_with_its_cause(store) -> None:
    def boom(ctx: StepContext) -> dict:
        raise KeyError("missing_field")

    engine = _engine(store, boom)
    run = engine.submit(RunSpec(plan=PLAN, payload={}, idempotency_key="recorded-run-01"))
    with pytest.raises(KeyError):
        engine.run(run.run_id, PLAN)

    final = store.load(run.run_id)
    assert len(final.steps) == 1
    assert final.steps[0].outcome is StepOutcome.FAILED
    assert "KeyError" in final.steps[0].output["error"]


def test_a_simulated_crash_still_leaves_the_run_resumable(store) -> None:
    """The one exception that must not be turned into a terminal state.

    SimulatedCrash models the process dying. A dead process writes nothing, so
    the run has to stay resumable — otherwise the resume tests would be
    measuring the engine's error handling rather than its recovery.
    """
    calls: list[int] = []

    def once(ctx: StepContext) -> dict:
        calls.append(ctx.index)
        return {}

    plan = Plan(name="p", steps=(Step(name="boom"), Step(name="boom2")))
    engine = Engine(store=store, provider=MockProvider(), handlers={"boom": once, "boom2": once})
    run = engine.submit(RunSpec(plan=plan, payload={}, idempotency_key="crash-run-00001"))

    with pytest.raises(SimulatedCrash):
        engine.run(run.run_id, plan, crash_at=1)

    mid = store.load(run.run_id)
    assert not mid.status.is_terminal, "a crashed run must remain resumable"

    final = engine.run(run.run_id, plan)
    assert final.status is RunStatus.COMPLETED
    assert calls == [0, 1], "step 0 ran once, before the crash"
