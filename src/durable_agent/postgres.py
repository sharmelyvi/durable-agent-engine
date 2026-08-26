"""PostgreSQL backend.

The SQLite store exists so the demo runs with no setup. This one is the shape
the engine is actually designed for, and the difference is not the SQL — it is
the lock.

SQLite has no advisory lock, so that backend leases a row and has to reason
about expiry, ownership and reclaiming after a crashed holder. Postgres has
``pg_try_advisory_lock``, owned by the connection and released by the database
when that connection goes away — including when the process is killed. There is
no TTL to tune and no orphaned lock to clean up, because there is no state to
leak. See ``PostgresStore.lock`` for why this is session-scoped rather than
transaction-scoped.

That is why ``advisory_lock_id`` is computed with blake2b rather than Python's
``hash()``: this is the code path where a per-process-random lock id would
silently stop protecting anything. tests/test_lock_identity.py demonstrates
that failure; this module is what it was guarding.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg import sql
from psycopg.rows import dict_row

from .models import (
    RunSpec,
    RunState,
    RunStatus,
    StepOutcome,
    StepRecord,
    advisory_lock_id,
)
from .store import LockUnavailable, RunAlreadyActive

# Assembled from named parts rather than written as one connection URL. A URL
# carrying inline credentials is the shape secrets leak in, and source that
# contains no such string cannot have one copied out of it. The local container
# in docker-compose.yml needs no secret at all: it binds to 127.0.0.1 and keeps
# its data in tmpfs, so there is nothing there to protect.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55433
DEFAULT_USER = "durable"
DEFAULT_DB = "durable_test"


def default_dsn() -> str:
    return f"postgresql://{DEFAULT_USER}@{DEFAULT_HOST}:{DEFAULT_PORT}/{DEFAULT_DB}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    plan_name       TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Same invariant as the SQLite backend, expressed with a native partial index:
-- at most one non-terminal run per idempotency key. Two workers racing cannot
-- both win, because the index decides, not the application.
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_run
    ON runs (idempotency_key)
    WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS step_records (
    run_id      TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    step_index  INTEGER NOT NULL,
    name        TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    attempts    INTEGER NOT NULL,
    output      JSONB NOT NULL DEFAULT '{}'::jsonb,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, step_index)
);

-- JSONB rather than TEXT so checkpoints stay queryable. Asking "which runs
-- escalated on a confidence field" should be a query, not a log grep.
CREATE INDEX IF NOT EXISTS ix_step_records_outcome
    ON step_records (outcome);
"""


def dsn_from_env() -> str:
    """Connection string, from ``DURABLE_AGENT_DSN`` or the local compose default.

    A real deployment passes its own DSN through the environment. Nothing about
    a production connection belongs in this file.
    """
    return os.environ.get("DURABLE_AGENT_DSN", default_dsn())


class PostgresStore:
    """Durable store backed by PostgreSQL.

    Connections are opened per operation rather than pooled. At the scale this
    targets that is the honest simple choice; a pool belongs here once there is
    a measured contention number to size it against, which is exactly how the
    pool in FasterClas was sized.
    """

    def __init__(self, dsn: str | None = None, schema: str = "public") -> None:
        self.dsn = dsn or dsn_from_env()
        # A named schema lets several independent deployments — or several
        # concurrent test runs — share one database without sharing tables.
        #
        # Quoted with psycopg's Identifier rather than a hand-written character
        # check. A bespoke validator is a guess at which inputs are dangerous;
        # the driver's quoting is the answer for all of them.
        #
        # The length limit is separate and not about safety: Postgres truncates
        # identifiers at 63 bytes, so two long names sharing a prefix would
        # silently become the same schema.
        if not schema:
            raise ValueError("schema name must not be empty")
        if len(schema.encode()) > 63:
            raise ValueError(
                f"schema name is {len(schema.encode())} bytes; PostgreSQL truncates "
                "identifiers at 63, which would silently collide"
            )
        self.schema = schema
        self._ident = sql.Identifier(schema)

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        with psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row) as conn:
            if self.schema != "public":
                conn.execute(sql.SQL("SET search_path TO {}").format(self._ident))
            yield conn

    def setup(self) -> None:
        with self._conn() as conn:
            if self.schema != "public":
                conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(self._ident))
                conn.execute(sql.SQL("SET search_path TO {}").format(self._ident))
            conn.execute(SCHEMA)

    def drop(self) -> None:
        """Remove this store's schema. Used by tests to isolate themselves."""
        if self.schema == "public":
            raise RuntimeError("refusing to drop the public schema")
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(self._ident))

    def create_run(self, run_id: str, spec: RunSpec) -> RunState:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, idempotency_key, plan_name, payload, status) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        run_id,
                        spec.idempotency_key,
                        spec.plan.name,
                        json.dumps(spec.payload),
                        RunStatus.PENDING.value,
                    ),
                )
        except (pg_errors.UniqueViolation, pg_errors.IntegrityError) as exc:
            existing = self.find_by_key(spec.idempotency_key)
            if existing is not None:
                raise RunAlreadyActive(existing.run_id) from exc
            raise
        return RunState(
            run_id=run_id,
            idempotency_key=spec.idempotency_key,
            status=RunStatus.PENDING,
            plan_name=spec.plan.name,
            payload=spec.payload,
        )

    def _hydrate(self, conn: psycopg.Connection[dict[str, Any]], row: dict[str, Any]) -> RunState:
        steps = [
            StepRecord(
                run_id=r["run_id"],
                index=r["step_index"],
                name=r["name"],
                outcome=StepOutcome(r["outcome"]),
                attempts=r["attempts"],
                output=r["output"],
                tokens_used=r["tokens_used"],
                created_at=r["created_at"],
            )
            for r in conn.execute(
                "SELECT * FROM step_records WHERE run_id = %s ORDER BY step_index",
                (row["run_id"],),
            ).fetchall()
        ]
        return RunState(
            run_id=row["run_id"],
            idempotency_key=row["idempotency_key"],
            status=RunStatus(row["status"]),
            plan_name=row["plan_name"],
            payload=row["payload"],
            steps=steps,
            error=row["error"],
        )

    def load(self, run_id: str) -> RunState | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
            return self._hydrate(conn, row) if row else None

    def find_by_key(self, idempotency_key: str) -> RunState | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE idempotency_key = %s ORDER BY created_at DESC LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            return self._hydrate(conn, row) if row else None

    def append(self, record: StepRecord) -> None:
        """Write a checkpoint. Replaying a step is a no-op, not a second row."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO step_records "
                "(run_id, step_index, name, outcome, attempts, output, tokens_used, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (run_id, step_index) DO NOTHING",
                (
                    record.run_id,
                    record.index,
                    record.name,
                    record.outcome.value,
                    record.attempts,
                    json.dumps(record.output),
                    record.tokens_used,
                    record.created_at,
                ),
            )

    def set_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = %s, error = %s WHERE run_id = %s",
                (status.value, error, run_id),
            )

    @contextmanager
    def lock(self, key: str, ttl: float = 60.0) -> Iterator[None]:
        """Session-scoped advisory lock.

        ``ttl`` is accepted to satisfy the Store protocol and ignored. There is
        no expiry to guess at: the lock belongs to the connection, and Postgres
        releases it when that connection goes away, including when the process
        is killed.

        A process that *hangs* rather than dies is the exception: the connection
        stays open, so the lock stays held with nothing to reclaim it. Bound
        your handlers — see the limits section of the README.

        Session-scoped rather than ``pg_try_advisory_xact_lock``, and the
        difference is not cosmetic. A transaction-scoped lock must keep a
        transaction open for as long as the lock is held, which for a run that
        takes minutes leaves the connection ``idle in transaction``: it pins a
        snapshot so VACUUM cannot clean up behind it, PgBouncer in transaction
        mode pins the server connection, and any deployment that sets
        ``idle_in_transaction_session_timeout`` eventually kills the worker
        mid-run. Measured, not assumed — see
        ``tests/test_postgres_lock.py::test_holding_the_lock_does_not_hold_a_transaction``,
        which fails if this is ever changed back.

        The cost is that release is explicit, so it lives in a finally.
        Advisory locks are not transactional, so the unlock lands even when the
        work inside raised.
        """
        lock_id = advisory_lock_id(key)
        conn = psycopg.connect(self.dsn, autocommit=True)
        acquired = False
        try:
            row = conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,)).fetchone()
            acquired = bool(row and row[0])
            if not acquired:
                raise LockUnavailable(f"{key} is held by another worker")
            yield
        finally:
            if acquired:
                with suppress(Exception):
                    conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
            conn.close()
