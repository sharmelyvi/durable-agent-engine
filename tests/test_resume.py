"""Crash a run, resume it, and prove nothing ran twice.

This is the property the whole project exists for. An agent that restarts a
five-step plan from step one after a crash pays for every earlier step again and
may repeat an effect that was already committed — a second charge, a second
email. These tests kill the run at each position and check both halves: work is
not repeated, and the effect is deduplicable.
"""

from __future__ import annotations

import pytest

from durable_agent.engine import Engine, SimulatedCrash
from durable_agent.models import Plan, RunStatus, StepOutcome

from .support import delete_checkpoint


def test_crash_before_effect_then_resume_completes(
    engine: Engine, plan: Plan, submitted: str, ledger
) -> None:
    with pytest.raises(SimulatedCrash):
        engine.run(submitted, plan, crash_at=2)

    partial = engine.store.load(submitted)
    assert partial is not None
    assert [s.name for s in partial.steps] == ["parse", "assess"]
    assert ledger.calls == [], "the effect must not have run before the crash"

    final = engine.run(submitted, plan)
    assert final.status is RunStatus.COMPLETED
    assert [s.name for s in final.steps] == ["parse", "assess", "charge", "notify"]
    assert len(ledger.calls) == 1


def test_crash_after_effect_does_not_charge_twice(
    engine: Engine, plan: Plan, submitted: str, ledger
) -> None:
    """The dangerous window: the effect committed, the checkpoint did not."""
    with pytest.raises(SimulatedCrash):
        engine.run(submitted, plan, crash_at=3)

    assert len(ledger.calls) == 1, "charge ran once before the crash"

    final = engine.run(submitted, plan)
    assert final.status is RunStatus.COMPLETED
    assert len(ledger.calls) == 1, "resume must not re-run a checkpointed effect"
    assert len(ledger.distinct) == 1


@pytest.mark.parametrize("crash_at", [0, 1, 2, 3])
def test_resume_from_every_position(
    engine: Engine, plan: Plan, submitted: str, ledger, crash_at: int
) -> None:
    """Whatever step the process dies on, the run finishes without repeating an effect."""
    with pytest.raises(SimulatedCrash):
        engine.run(submitted, plan, crash_at=crash_at)

    final = engine.run(submitted, plan)

    assert final.status is RunStatus.COMPLETED
    assert len(final.steps) == len(plan.steps)
    assert len(ledger.distinct) == 1, "exactly one distinct effect token"


def test_effect_token_is_identical_across_restarts(
    engine: Engine, plan: Plan, submitted: str, ledger
) -> None:
    """An external API deduplicating on this token would reject the replay."""
    with pytest.raises(SimulatedCrash):
        engine.run(submitted, plan, crash_at=3)
    first = ledger.calls[0]

    # Force a replay of the effect step by deleting its checkpoint, standing in
    # for a crash between performing the effect and committing the checkpoint.
    delete_checkpoint(engine.store, submitted, step_index=2)

    engine.run(submitted, plan)

    assert len(ledger.calls) == 2, "the step genuinely ran again"
    assert ledger.calls[0] == ledger.calls[1] == first
    assert len(ledger.distinct) == 1, "same token, so the effect is deduplicable"


def test_completed_run_is_not_re_executed(
    engine: Engine, plan: Plan, submitted: str, ledger
) -> None:
    engine.run(submitted, plan)
    assert len(ledger.calls) == 1

    again = engine.run(submitted, plan)
    assert again.status is RunStatus.COMPLETED
    assert len(ledger.calls) == 1


def test_checkpoints_record_token_cost_per_step(engine: Engine, plan: Plan, submitted: str) -> None:
    """Cost is attributed to the step that spent it, not to the run as a lump."""
    final = engine.run(submitted, plan)
    by_name = {s.name: s for s in final.steps}

    assert by_name["assess"].tokens_used > 0, "the model step spent tokens"
    assert by_name["parse"].tokens_used == 0, "a deterministic step spends none"
    assert final.tokens_used == sum(s.tokens_used for s in final.steps)
    assert all(s.outcome is StepOutcome.OK for s in final.steps)


def test_a_step_records_the_attempts_it_actually_made(store) -> None:
    """The attempts column has to mean something.

    Recording a literal 1 for every step would make the column decorative: a
    step that burned three provider calls before succeeding would be
    indistinguishable in the record from one that succeeded immediately, and
    cost attribution after an incident would be guesswork.
    """
    from durable_agent.engine import Engine, StepContext
    from durable_agent.models import Plan, RunSpec, Step
    from durable_agent.providers import ChaosProvider

    from .support import Decision

    def assess(ctx: StepContext) -> dict:
        return ctx.ask("Assess this.", Decision).model_dump()

    # Fails twice, then answers: three provider calls for one step. The draws
    # are scripted rather than seeded, because the point is the exact count.
    class ScriptedProvider(ChaosProvider):
        def __init__(self) -> None:
            super().__init__(failure_rate=1.0, seed=3, modes=("timeout",), heal_after=0)
            self._draws = iter([0.0, 0.0, 1.0])  # fail, fail, succeed

        def complete(self, prompt: str):  # type: ignore[override]
            self.rng.random = lambda: next(self._draws)  # type: ignore[method-assign]
            return super().complete(prompt)

    provider = ScriptedProvider()

    plan = Plan(name="p", steps=(Step(name="assess", max_attempts=5),))
    engine = Engine(store=store, provider=provider, handlers={"assess": assess})
    run = engine.submit(RunSpec(plan=plan, payload={}, idempotency_key="attempts-run-01"))
    final = engine.run(run.run_id, plan)

    assert final.steps[0].attempts == 3, (
        f"expected the record to show three provider calls, got {final.steps[0].attempts}"
    )


def test_a_step_can_declare_its_own_retry_budget(store) -> None:
    """max_attempts on a Step must reach the engine.

    A configuration field the engine never reads is worse than no field: it
    looks like a control and silently is not one.
    """
    from durable_agent.engine import Engine, Escalation, StepContext
    from durable_agent.models import Plan, RunSpec, RunStatus, Step
    from durable_agent.providers import ChaosProvider

    from .support import Decision

    calls: list[int] = []

    def assess(ctx: StepContext) -> dict:
        try:
            return ctx.ask("Assess this.", Decision).model_dump()
        except Escalation:
            calls.append(ctx.provider.calls)  # type: ignore[attr-defined]
            raise

    plan = Plan(name="p", steps=(Step(name="assess", max_attempts=7),))
    provider = ChaosProvider(failure_rate=1.0, seed=1, modes=("unavailable",))
    engine = Engine(store=store, provider=provider, handlers={"assess": assess})
    run = engine.submit(RunSpec(plan=plan, payload={}, idempotency_key="budget-run-0001"))
    final = engine.run(run.run_id, plan)

    assert final.status is RunStatus.ESCALATED
    assert calls == [7], f"the step asked for 7 attempts; the engine made {calls}"
