"""Run the engine under deliberate provider failure and check the invariants hold.

The claim under test is not "nothing fails" — things fail constantly here, by
design. The claim is that no failure leaves the system in a state a human cannot
reason about:

* every run reaches a terminal status; none hangs half-done;
* an escalated run says why, in the record, not just in a log line;
* no effect is ever performed under two different tokens;
* checkpoints are contiguous, so resume position is always well defined.

Every provider is seeded. A chaos suite that cannot replay its own failure is
reporting flakiness, not resilience.
"""

from __future__ import annotations

import pytest

from durable_agent.engine import Engine, StepContext
from durable_agent.models import Plan, RunSpec, RunStatus, Step
from durable_agent.providers import ChaosProvider
from durable_agent.store import SQLiteStore

from .support import Decision, EffectLedger


def _build(store: SQLiteStore, ledger: EffectLedger, provider: ChaosProvider) -> Engine:
    def assess(ctx: StepContext) -> dict:
        return ctx.ask("Assess this transaction for risk.", Decision).model_dump()

    def charge(ctx: StepContext) -> dict:
        return {"receipt": ledger.charge(ctx.effect_token)}

    return Engine(store=store, provider=provider, handlers={"assess": assess, "charge": charge})


CHAOS_PLAN = Plan(name="chaos", steps=(Step(name="assess"), Step(name="charge", effect=True)))


@pytest.mark.parametrize("failure_rate", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_every_run_reaches_a_terminal_state(
    store: SQLiteStore, ledger: EffectLedger, failure_rate: float
) -> None:
    engine = _build(store, ledger, ChaosProvider(failure_rate=failure_rate, seed=7))

    for i in range(20):
        spec = RunSpec(plan=CHAOS_PLAN, payload={"n": i}, idempotency_key=f"chaos-run-{i:04d}")
        run = engine.submit(spec)
        final = engine.run(run.run_id, CHAOS_PLAN)

        assert final.status.is_terminal, f"run {i} stalled in {final.status}"
        if final.status is RunStatus.ESCALATED:
            assert final.error, "an escalation must record its reason"


def test_no_effect_is_ever_performed_under_two_tokens(
    store: SQLiteStore, ledger: EffectLedger
) -> None:
    engine = _build(store, ledger, ChaosProvider(failure_rate=0.5, seed=99))

    for i in range(40):
        spec = RunSpec(plan=CHAOS_PLAN, payload={"n": i}, idempotency_key=f"token-run-{i:04d}")
        run = engine.submit(spec)
        engine.run(run.run_id, CHAOS_PLAN)

    assert len(ledger.calls) == len(ledger.distinct), "an effect repeated under one token"


def test_checkpoints_are_contiguous_so_resume_is_well_defined(
    store: SQLiteStore, ledger: EffectLedger
) -> None:
    engine = _build(store, ledger, ChaosProvider(failure_rate=0.6, seed=2024))

    for i in range(25):
        spec = RunSpec(plan=CHAOS_PLAN, payload={"n": i}, idempotency_key=f"gap-run-{i:04d}")
        run = engine.submit(spec)
        final = engine.run(run.run_id, CHAOS_PLAN)

        indices = [s.index for s in final.steps]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices))), f"gap in checkpoints: {indices}"


def test_total_provider_outage_escalates_rather_than_looping(
    store: SQLiteStore, ledger: EffectLedger
) -> None:
    """When the provider is simply down, stopping is the correct behaviour."""
    provider = ChaosProvider(failure_rate=1.0, seed=1, modes=("unavailable",))
    engine = _build(store, ledger, provider)

    spec = RunSpec(plan=CHAOS_PLAN, payload={}, idempotency_key="outage-run-0001")
    run = engine.submit(spec)
    final = engine.run(run.run_id, CHAOS_PLAN)

    assert final.status is RunStatus.ESCALATED
    assert "unavailable" in (final.error or "")
    assert ledger.calls == [], "no money moves when the run never got past assessment"
    assert provider.calls <= 3, "retries are bounded, not infinite"


def test_persistent_schema_violation_escalates_after_two_corrections(
    store: SQLiteStore, ledger: EffectLedger
) -> None:
    provider = ChaosProvider(failure_rate=1.0, seed=5, modes=("bad_schema",), heal_after=99)
    engine = _build(store, ledger, provider)

    spec = RunSpec(plan=CHAOS_PLAN, payload={}, idempotency_key="badschema-run-01")
    run = engine.submit(spec)
    final = engine.run(run.run_id, CHAOS_PLAN)

    assert final.status is RunStatus.ESCALATED
    assert "confidence" in (final.error or ""), "the escalation names the offending field"
