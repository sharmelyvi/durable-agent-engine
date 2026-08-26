"""Resuming a run under a changed plan must fail loudly, not quietly.

Resume position is an index. The engine derives it from checkpoints and then
walks a plan handed in by the caller — and nothing forces those two to describe
the same work.

A rolling deploy is exactly the case that separates them. Worker v1 dies
mid-run; worker v2 is already on new code and picks the run up. If a step was
removed, index 2 no longer means what it meant when index 1 was written, and
the engine would execute the wrong step and mark the run complete.

The failure this prevents: an order that finishes without ever being charged.
"""

from __future__ import annotations

import pytest

from durable_agent.engine import Engine, PlanChanged, SimulatedCrash, StepContext
from durable_agent.models import Plan, RunSpec, RunStatus, Step
from durable_agent.providers import MockProvider

V1 = Plan(
    name="checkout",
    steps=(
        Step(name="validate"),
        Step(name="screen"),
        Step(name="charge", effect=True),
        Step(name="notify"),
    ),
)


def _build(store, charges: list[str]) -> Engine:
    def make(name: str):
        def handler(ctx: StepContext) -> dict:
            if name == "charge":
                charges.append(ctx.effect_token)
            return {"step": name}

        return handler

    names = ("validate", "screen", "charge", "notify", "extra")
    return Engine(
        store=store,
        provider=MockProvider(),
        handlers={n: make(n) for n in names},
    )


def _crash_after_two(engine: Engine, key: str) -> str:
    run = engine.submit(RunSpec(plan=V1, payload={}, idempotency_key=key))
    with pytest.raises(SimulatedCrash):
        engine.run(run.run_id, V1, crash_at=2)
    return run.run_id


def test_a_removed_step_is_refused_rather_than_skipped(store) -> None:
    """The one that costs money: without this, the charge never happens."""
    charges: list[str] = []
    engine = _build(store, charges)
    run_id = _crash_after_two(engine, "drift-removed-0001")

    without_screen = Plan(
        name="checkout",
        steps=(Step(name="validate"), Step(name="charge", effect=True), Step(name="notify")),
    )

    with pytest.raises(PlanChanged, match="'screen'"):
        engine.run(run_id, without_screen)

    assert charges == [], "nothing ran under the wrong plan"
    assert store.load(run_id).status is RunStatus.RUNNING, "and the run is not stranded"


def test_a_reordered_plan_is_refused(store) -> None:
    charges: list[str] = []
    engine = _build(store, charges)
    run_id = _crash_after_two(engine, "drift-reorder-001")

    swapped = Plan(
        name="checkout",
        steps=(
            Step(name="screen"),
            Step(name="validate"),
            Step(name="charge", effect=True),
            Step(name="notify"),
        ),
    )
    with pytest.raises(PlanChanged):
        engine.run(run_id, swapped)


def test_a_truncated_plan_is_refused(store) -> None:
    engine = _build(store, [])
    run_id = _crash_after_two(engine, "drift-short-00001")

    with pytest.raises(PlanChanged, match="only 1 steps"):
        engine.run(run_id, Plan(name="checkout", steps=(Step(name="validate"),)))


def test_appending_steps_after_the_resume_point_is_allowed(store) -> None:
    """Not every change invalidates a run.

    Steps beyond the resume point have not happened yet. A plan that grows a
    tail between deploys is still a plan this run can finish under, and
    refusing it would make every deploy break every in-flight run.
    """
    charges: list[str] = []
    engine = _build(store, charges)
    run_id = _crash_after_two(engine, "drift-append-0001")

    with_extra = Plan(name="checkout", steps=(*V1.steps, Step(name="extra")))
    final = engine.run(run_id, with_extra)

    assert final.status is RunStatus.COMPLETED
    assert len(charges) == 1
    assert [s.name for s in final.steps] == [
        "validate",
        "screen",
        "charge",
        "notify",
        "extra",
    ]


def test_the_run_still_finishes_under_the_plan_it_started_with(store) -> None:
    """Refusing the wrong plan must not strand the run under the right one."""
    charges: list[str] = []
    engine = _build(store, charges)
    run_id = _crash_after_two(engine, "drift-recover-001")

    with pytest.raises(PlanChanged):
        engine.run(run_id, Plan(name="checkout", steps=(Step(name="validate"),)))

    final = engine.run(run_id, V1)
    assert final.status is RunStatus.COMPLETED
    assert len(charges) == 1
