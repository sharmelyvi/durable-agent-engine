"""Durable storage for runs and checkpoints.

The engine holds no state of its own. Everything that decides whether a step may
execute lives in the database, because a process can die between any two lines
and the answer has to survive that.

Two backends implement the same protocol:

* ``SQLiteStore``   — zero configuration, used by the demo and the test suite.
* ``PostgresStore`` — the production shape, using transaction-scoped advisory
  locks so several workers can compete for the same run safely.

They are not interchangeable in their concurrency guarantees, and that
difference is documented rather than smoothed over. SQLite serialises writers
with ``BEGIN IMMEDIATE`` on a single file; Postgres coordinates workers across
machines. Both enforce at-most-one active run per idempotency key with a partial
unique index, which is the invariant that actually protects the caller.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .models import (
    RunSpec,
    RunState,
    RunStatus,
    StepOutcome,
    StepRecord,
    advisory_lock_id,
)


class RunAlreadyActive(Exception):
    """Another run with the same idempotency key is not finished yet."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} is already active for this idempotency key")
        self.run_id = run_id


class LockUnavailable(Exception):
    """Another worker currently holds this run's lock."""


class Store(Protocol):
    def setup(self) -> None: ...
    def create_run(self, run_id: str, spec: RunSpec) -> RunState: ...
    def load(self, run_id: str) -> RunState | None: ...
    def find_by_key(self, idempotency_key: str) -> RunState | None: ...
    def append(self, record: StepRecord) -> None: ...
    def set_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None: ...
    def lock(self, key: str) -> Iterator[None]: ...


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    plan_name       TEXT NOT NULL,
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The invariant: one non-terminal run per idempotency key. A duplicate request
-- is rejected by the database, not by an application check that races.
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_run
    ON runs (idempotency_key)
    WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS step_records (
    run_id      TEXT NOT NULL REFERENCES runs (run_id),
    step_index  INTEGER NOT NULL,
    name        TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    attempts    INTEGER NOT NULL,
    output      TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, step_index)
);
"""


class SQLiteStore:
    """File-backed store. WAL mode so a reader never blocks the writer."""

    def __init__(self, path: str | Path = "durable_agent.db") -> None:
        self.path = str(path)

    @contextmanager
    def _conn(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def setup(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(SQLITE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def create_run(self, run_id: str, spec: RunSpec) -> RunState:
        try:
            with self._conn(immediate=True) as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, idempotency_key, plan_name, payload, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id,
                        spec.idempotency_key,
                        spec.plan.name,
                        json.dumps(spec.payload),
                        RunStatus.PENDING.value,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.find_by_key(spec.idempotency_key)
            if existing is not None:
                raise RunAlreadyActive(existing.run_id) from exc
            raise
        return RunState(
            run_id=run_id,
            idempotency_key=spec.idempotency_key,
            status=RunStatus.PENDING,
            plan_name=spec.plan.name,
        )

    def _hydrate(self, conn: sqlite3.Connection, row: sqlite3.Row) -> RunState:
        steps = [
            StepRecord(
                run_id=r["run_id"],
                index=r["step_index"],
                name=r["name"],
                outcome=StepOutcome(r["outcome"]),
                attempts=r["attempts"],
                output=json.loads(r["output"]),
                tokens_used=r["tokens_used"],
                created_at=r["created_at"],
            )
            for r in conn.execute(
                "SELECT * FROM step_records WHERE run_id = ? ORDER BY step_index",
                (row["run_id"],),
            )
        ]
        return RunState(
            run_id=row["run_id"],
            idempotency_key=row["idempotency_key"],
            status=RunStatus(row["status"]),
            plan_name=row["plan_name"],
            steps=steps,
            error=row["error"],
        )

    def load(self, run_id: str) -> RunState | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return self._hydrate(conn, row) if row else None

    def find_by_key(self, idempotency_key: str) -> RunState | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE idempotency_key = ? ORDER BY created_at DESC LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            return self._hydrate(conn, row) if row else None

    def append(self, record: StepRecord) -> None:
        """Write a checkpoint. Idempotent: replaying the same step is a no-op."""
        with self._conn(immediate=True) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO step_records "
                "(run_id, step_index, name, outcome, attempts, output, tokens_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.run_id,
                    record.index,
                    record.name,
                    record.outcome.value,
                    record.attempts,
                    json.dumps(record.output),
                    record.tokens_used,
                    record.created_at.isoformat(),
                ),
            )

    def set_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        with self._conn(immediate=True) as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = ? WHERE run_id = ?",
                (status.value, error, run_id),
            )

    @contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """Serialise workers on this key.

        SQLite has no advisory lock, so the write lock on a dedicated row stands
        in for one. Held for the duration of the run, which is acceptable for a
        single-machine demo and explicitly not the production story — see
        PostgresStore.
        """
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=0.5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                raise LockUnavailable(key) from exc
            conn.execute("CREATE TABLE IF NOT EXISTS locks (lock_id INTEGER PRIMARY KEY, key TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO locks (lock_id, key) VALUES (?, ?)",
                (advisory_lock_id(key), key),
            )
            yield
            conn.execute("COMMIT")
        except LockUnavailable:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
