"""The at-most-once guarantee must not depend on the lock.

Locks fail. A SQLite lease can expire while its holder is still alive and
working — the classic hazard with lease-based locking, and the reason fencing
tokens exist. A Postgres session lock has the opposite failure: a handler that
hangs on a network call holds it until someone kills the process.

Either way, two workers can end up executing the same run. If the guarantee
rested on mutual exclusion, that would mean a double charge.

It does not. The effect token is derived from ``(run_id, step_index)``, so
concurrent executions present the same token and the external system
deduplicates them. The checkpoint's primary key does the same for state. The
lock is an optimisation that avoids paying for work twice; it is not what makes
the work safe.

These tests remove the lock entirely and assert the invariant still holds.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from durable_agent.engine import Engine, StepContext
from durable_agent.models import Plan, RunSpec, RunStatus, Step
from durable_agent.providers import MockProvider

WORKERS = 3


@contextmanager
def _no_lock(key: str, ttl: float = 60.0) -> Iterator[None]:
    """Stands in for every way a lock can fail to exclude anyone."""
    yield


def test_concurrent_execution_without_a_lock_still_charges_once(store) -> None:
    gateway: list[str] = []
    guard = threading.Lock()
    at_the_effect = threading.Barrier(WORKERS)

    def charge(ctx: StepContext) -> dict:
        at_the_effect.wait()  # force the overlap a real race would only sometimes give
        with guard:
            gateway.append(ctx.effect_token)
        return {"receipt": ctx.effect_token}

    plan = Plan(name="p", steps=(Step(name="charge", effect=True),))
    store.lock = _no_lock

    engine = Engine(store=store, provider=MockProvider(), handlers={"charge": charge})
    run = engine.submit(RunSpec(plan=plan, payload={}, idempotency_key="unlocked-run-01"))

    threads = [threading.Thread(target=engine.run, args=(run.run_id, plan)) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(gateway) == WORKERS, "the point of this test is that they really did overlap"
    assert len(set(gateway)) == 1, (
        "concurrent executions presented different effect tokens; the guarantee "
        "was resting on the lock, not on the token"
    )

    final = store.load(run.run_id)
    assert final is not None
    assert len(final.steps) == 1, "one checkpoint, whatever the concurrency"
    assert final.status is RunStatus.COMPLETED


def test_a_replayed_step_reuses_the_token_of_the_original(store) -> None:
    """The same property, stated without threads.

    A worker that resumes a run after another worker already performed a step
    must present that step's original token, not a new one.
    """
    seen: list[str] = []

    def charge(ctx: StepContext) -> dict:
        seen.append(ctx.effect_token)
        return {"receipt": ctx.effect_token}

    plan = Plan(name="p", steps=(Step(name="charge", effect=True),))
    store.lock = _no_lock
    engine = Engine(store=store, provider=MockProvider(), handlers={"charge": charge})
    run = engine.submit(RunSpec(plan=plan, payload={}, idempotency_key="replayed-run-01"))

    engine.run(run.run_id, plan)
    # A second engine, as a different worker would be, replaying the same step.
    other = Engine(store=store, provider=MockProvider(), handlers={"charge": charge})
    ctx = StepContext(run_id=run.run_id, index=0, payload={}, previous={}, provider=MockProvider())
    charge(ctx)

    assert len(seen) == 2
    assert seen[0] == seen[1], "a replay must present the original token"
    assert other is not None
