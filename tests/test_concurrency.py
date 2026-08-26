"""Two workers, one key, real threads.

A single-threaded test that calls submit twice proves the check exists. It does
not prove the check holds when both calls are in flight at once, which is the
only case that matters — a duplicate webhook does not arrive politely after the
first one finished.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from durable_agent.engine import Engine
from durable_agent.models import Plan, RunSpec, RunStatus
from durable_agent.store import LockUnavailable, RunAlreadyActive, SQLiteStore


def test_concurrent_submits_produce_exactly_one_run(engine: Engine, spec: RunSpec) -> None:
    barrier = threading.Barrier(8)
    accepted: list[str] = []
    rejected: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()  # release all threads at the same instant
        try:
            state = engine.submit(spec)
        except RunAlreadyActive as exc:
            with lock:
                rejected.append(exc.run_id)
        else:
            with lock:
                accepted.append(state.run_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: attempt(), range(8)))

    assert len(accepted) == 1, f"expected one winner, got {len(accepted)}"
    assert len(rejected) == 7
    assert set(rejected) == set(accepted), "losers must point at the winning run"


def test_submit_or_attach_returns_the_same_run(engine: Engine, spec: RunSpec) -> None:
    """What an idempotent endpoint needs: a repeat is not an error."""
    first = engine.submit_or_attach(spec)
    second = engine.submit_or_attach(spec)
    assert first.run_id == second.run_id


def test_second_worker_cannot_run_a_locked_run(
    engine: Engine, plan: Plan, submitted: str, store: SQLiteStore
) -> None:
    state = store.load(submitted)
    assert state is not None

    with store.lock(state.idempotency_key), pytest.raises(LockUnavailable):
        engine.run(submitted, plan)

    # Once released, the run proceeds normally.
    final = engine.run(submitted, plan)
    assert final.status is RunStatus.COMPLETED


def _second_holder_succeeds(store: SQLiteStore, key: str) -> bool:
    """Can a second worker take this key right now?"""
    try:
        with store.lock(key):
            return True
    except LockUnavailable:
        return False


def test_a_live_lease_blocks_a_second_worker(store: SQLiteStore) -> None:
    with store.lock("stuck-key", ttl=60.0):
        assert _second_holder_succeeds(store, "stuck-key") is False


def test_an_expired_lease_is_reclaimed(store: SQLiteStore) -> None:
    """A worker that died holding the lock must not block the key forever.

    A negative TTL stands in for a lease that lapsed while its holder was gone:
    the successor takes it over instead of waiting for an operator.
    """
    with store.lock("stuck-key", ttl=-1.0):
        assert _second_holder_succeeds(store, "stuck-key") is True


def test_a_new_run_is_allowed_after_the_previous_one_finishes(
    engine: Engine, plan: Plan, spec: RunSpec
) -> None:
    first = engine.submit(spec)
    engine.run(first.run_id, plan)

    second = engine.submit(spec)
    assert second.run_id != first.run_id
