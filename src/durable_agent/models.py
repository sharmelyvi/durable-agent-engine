"""Domain contracts for durable agent runs.

Every value that crosses a boundary — into the database, into a model provider,
back out to a caller — is a Pydantic model. Nothing in this engine passes a bare
dict around, because the whole premise is that non-deterministic output cannot be
trusted until it has been validated at the edge.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    """Lifecycle of a run.

    Terminal states are COMPLETED, FAILED and ESCALATED. A run in any other state
    is eligible for resume by a worker that acquires its lock.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ESCALATED}


class StepOutcome(StrEnum):
    OK = "ok"
    RETRIED = "retried"
    HEALED = "healed"
    FAILED = "failed"


class Step(BaseModel):
    """One unit of work in a plan.

    `effect` marks a step that touches the outside world (charges a card, sends a
    message). Those are the steps that must never run twice, and the ones the
    engine hands a deterministic effect token to.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    effect: bool = False
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("step name must be alphanumeric with - or _")
        return v


class Plan(BaseModel):
    """An ordered list of steps, executed at most once each per run."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    steps: tuple[Step, ...] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def _unique_names(cls, v: tuple[Step, ...]) -> tuple[Step, ...]:
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique within a plan")
        return v


class RunSpec(BaseModel):
    """What a caller submits to start a run."""

    model_config = ConfigDict(frozen=True)

    plan: Plan
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=8, max_length=200)


class StepRecord(BaseModel):
    """An immutable checkpoint. One row per completed step attempt.

    Resume position is derived by reading these back, so a crash between the
    effect and the checkpoint is the only window that matters — and that window
    is closed by the effect token, not by hoping the process stays alive.
    """

    run_id: str
    index: int = Field(ge=0)
    name: str
    outcome: StepOutcome
    attempts: int = Field(ge=1)
    output: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class RunState(BaseModel):
    """Reconstructed view of a run. Never stored as-is; always derived."""

    run_id: str
    idempotency_key: str
    status: RunStatus
    plan_name: str
    steps: list[StepRecord] = Field(default_factory=list)
    error: str | None = None

    @property
    def next_index(self) -> int:
        """First step index that has no successful checkpoint."""
        done = {s.index for s in self.steps if s.outcome is not StepOutcome.FAILED}
        i = 0
        while i in done:
            i += 1
        return i

    @property
    def tokens_used(self) -> int:
        return sum(s.tokens_used for s in self.steps)


def effect_token(run_id: str, step_index: int) -> str:
    """Deterministic token a step hands to an external system for deduplication.

    Same run, same step, same token — across retries, across process restarts,
    across workers. An external API that honours it turns at-least-once delivery
    into at-most-once execution without this engine needing a distributed
    transaction.
    """
    return hashlib.blake2b(f"{run_id}:{step_index}".encode(), digest_size=16).hexdigest()


def advisory_lock_id(key: str) -> int:
    """Stable 63-bit lock id for pg_advisory_lock.

    Deliberately not Python's ``hash()``: string hashing is randomised per
    process by PYTHONHASHSEED, so two workers would compute different lock ids
    for the same key and the lock would silently protect nothing. blake2b is
    stable across processes and restarts, which is the entire requirement.
    """
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF
