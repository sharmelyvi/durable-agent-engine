"""Properties of the PostgreSQL lock that a comment cannot enforce.

Two design decisions live in `PostgresStore.lock`, and both are the kind that
get quietly reverted by anyone tidying the code later. They are asserted here so
that reverting them breaks the build rather than degrading production silently.
"""

from __future__ import annotations

import threading
import time

import pytest

from durable_agent.models import advisory_lock_id
from durable_agent.store import LockUnavailable

psycopg = pytest.importorskip("psycopg")

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("tests.conftest", reason="").HAVE_POSTGRES,
    reason="no PostgreSQL reachable; run `make pg-up` first",
)


@pytest.fixture
def pg():
    from durable_agent.postgres import PostgresStore

    store = PostgresStore(schema="t_locktest")
    store.setup()
    yield store
    store.drop()


def _sessions(dsn: str) -> list[tuple[str, bool]]:
    """Every non-idle backend on this database, and whether it holds a transaction."""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT state, xact_start IS NOT NULL FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        ).fetchall()
    return [(state, in_xact) for state, in_xact in rows if state and state != "idle"]


def test_holding_the_lock_does_not_hold_a_transaction(pg) -> None:
    """The reason this lock is session-scoped rather than transaction-scoped.

    `pg_try_advisory_xact_lock` would require an open transaction for the whole
    time the lock is held. On a run measured in minutes that leaves the
    connection `idle in transaction`, which pins a snapshot against VACUUM, pins
    a server connection under PgBouncer, and gets killed outright by any
    deployment with `idle_in_transaction_session_timeout` set.

    If someone changes the lock back, this test is what tells them.
    """
    observed: list[tuple[str, bool]] = []

    def watch() -> None:
        time.sleep(0.5)
        observed.extend(_sessions(pg.dsn))

    watcher = threading.Thread(target=watch)
    watcher.start()
    with pg.lock("held-while-observed"):
        time.sleep(1.0)
    watcher.join()

    idle_in_transaction = [s for s in observed if s[0] == "idle in transaction"]
    assert not idle_in_transaction, (
        "holding the run lock left a connection idle in transaction; the lock "
        "must be session-scoped (pg_try_advisory_lock), not transaction-scoped"
    )


def test_the_lock_is_released_after_the_block(pg) -> None:
    with pg.lock("release-me"):
        pass
    with pg.lock("release-me"):
        pass  # a second acquisition proves the first was released


def test_the_lock_is_released_even_when_the_body_raises(pg) -> None:
    """Release lives in a finally, so an exception inside must not strand it."""
    with pytest.raises(RuntimeError), pg.lock("raise-inside"):
        raise RuntimeError("work blew up")

    with pg.lock("raise-inside"):
        pass


def test_a_failed_acquisition_does_not_unlock_the_holder(pg) -> None:
    """The loser must not release the winner's lock on its way out."""
    lock_id = advisory_lock_id("contended")
    holder = psycopg.connect(pg.dsn, autocommit=True)
    try:
        got = holder.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,)).fetchone()
        assert got[0] is True

        with pytest.raises(LockUnavailable), pg.lock("contended"):
            pass  # pragma: no cover

        # The holder still owns it: a third party must still be refused.
        with pytest.raises(LockUnavailable), pg.lock("contended"):
            pass  # pragma: no cover
    finally:
        holder.close()
