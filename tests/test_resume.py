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
    import sqlite3

    conn = sqlite3.connect(engine.store.path)
    conn.execute("DELETE FROM step_records WHERE run_id = ? AND step_index = 2", (submitted,))
    conn.commit()
    conn.close()

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


def test_checkpoints_record_token_cost_per_step(
    engine: Engine, plan: Plan, submitted: str
) -> None:
    """Cost is attributed to the step that spent it, not to the run as a lump."""
    final = engine.run(submitted, plan)
    by_name = {s.name: s for s in final.steps}

    assert by_name["assess"].tokens_used > 0, "the model step spent tokens"
    assert by_name["parse"].tokens_used == 0, "a deterministic step spends none"
    assert final.tokens_used == sum(s.tokens_used for s in final.steps)
    assert all(s.outcome is StepOutcome.OK for s in final.steps)
