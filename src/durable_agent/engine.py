"""The execution engine.

A run is a plan plus an idempotency key. The engine walks the plan one step at a
time, writing an immutable checkpoint after each one, and it decides where to
start by reading those checkpoints back — never by trusting a variable that a
crash could have taken with it.

Three guarantees, each enforced by a specific mechanism rather than by
convention:

at-most-one active run per key
    A partial unique index in the database. A duplicate request is rejected by
    the storage engine, so two workers racing cannot both win.

no repeated work after a crash
    Checkpoints are append-only and keyed on (run_id, step_index). Resume reads
    the highest contiguous completed index and continues from the next one.

at-most-once external effects
    Steps marked ``effect=True`` receive a deterministic token derived from
    (run_id, step_index). Replaying the step produces the same token, so an
    external system that deduplicates on it will not act twice — including in
    the window between performing the effect and writing the checkpoint, which
    is the one window a database transaction cannot close on its own.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from .healing import (
    MAX_CORRECTION_ROUNDS,
    build_retry_prompt,
    parse_or_faults,
)
from .models import (
    Plan,
    RunSpec,
    RunState,
    RunStatus,
    StepOutcome,
    StepRecord,
    effect_token,
)
from .providers import Provider, ProviderTimeout, ProviderUnavailable
from .store import RunAlreadyActive, Store

T = TypeVar("T", bound=BaseModel)

TRANSIENT = (ProviderTimeout, ProviderUnavailable)


class Escalation(Exception):
    """The engine gave up and is handing this run to a human.

    Raised only after retries and correction rounds are exhausted. Carries the
    reason so the escalation record says what failed, not just that it failed.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PlanChanged(Exception):
    """The plan handed to ``run`` disagrees with what this run already did.

    Resume position is an index. If the plan behind that index has changed —
    a step inserted, removed or reordered — the index now points at different
    work, and continuing would silently execute the wrong steps. A run that
    completed under a plan where index 2 was ``charge`` must not finish under
    one where index 2 is ``notify``.

    This is not a hypothetical: it is what a rolling deploy looks like from the
    engine's side. An old worker dies mid-run and a new one, already running
    changed code, picks it up.
    """


class SimulatedCrash(Exception):
    """Injected by the demo to kill a run mid-flight. Never raised in normal use."""


@dataclass
class StepContext:
    """What a step handler is given.

    ``ask`` is the only way a handler reaches a model, which is what keeps
    retries, schema enforcement and healing in one place instead of scattered
    across every handler.
    """

    run_id: str
    index: int
    payload: dict[str, Any]
    previous: dict[str, dict[str, Any]]
    provider: Provider
    max_attempts: int = 3
    _tokens: list[int] = field(default_factory=list)
    _corrections: list[int] = field(default_factory=list)
    _attempts: int = 0

    @property
    def effect_token(self) -> str:
        """Stable across retries and restarts. Hand this to external APIs."""
        return effect_token(self.run_id, self.index)

    @property
    def tokens_used(self) -> int:
        return sum(self._tokens)

    @property
    def attempts(self) -> int:
        """Provider calls this step actually made. At least one, even if the
        step never reached a model — a checkpoint claiming zero attempts would
        be as wrong as one always claiming exactly one."""
        return max(1, self._attempts)

    @property
    def was_healed(self) -> bool:
        """True if any model call needed a correction round to validate."""
        return any(c > 0 for c in self._corrections)

    def ask(self, prompt: str, schema: type[T], max_attempts: int | None = None) -> T:
        """Call the model and return a validated object, or escalate.

        Transient failures are retried with exponential backoff. Schema
        failures are repaired with a delta correction, capped at
        MAX_CORRECTION_ROUNDS. Anything still broken after that is a human's
        problem, and saying so is more useful than looping.
        """
        # Defaults to the budget declared on the Step. Passing a number here
        # overrides it for one call; leaving it out means the plan decides,
        # which is where a retry budget belongs.
        budget = self.max_attempts if max_attempts is None else max_attempts
        current = prompt
        corrections = 0

        for attempt in range(1, budget + 1):
            self._attempts += 1
            try:
                completion = self.provider.complete(current)
            except TRANSIENT as exc:
                if attempt == budget:
                    raise Escalation(
                        f"provider unavailable after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(min(0.05 * (2 ** (attempt - 1)), 0.4))
                continue

            self._tokens.append(completion.tokens)
            parsed, faults = parse_or_faults(completion.text, schema)
            if parsed is not None:
                self._corrections.append(corrections)
                return parsed

            if corrections >= MAX_CORRECTION_ROUNDS:
                fields = ", ".join(f.field for f in faults)
                raise Escalation(f"schema still invalid after {corrections} corrections: {fields}")
            corrections += 1
            # Repairable answers are corrected from themselves, which keeps the
            # fields the model got right; a response with no JSON in it has
            # nothing to preserve and gets the task back instead.
            current = build_retry_prompt(prompt, completion.text, faults, schema.__name__)

        raise Escalation(f"exhausted {budget} attempts without a valid response")


StepHandler = Callable[[StepContext], dict[str, Any]]


def _assert_plan_matches_history(plan: Plan, state: RunState) -> None:
    """Refuse to resume a run whose completed steps no longer match the plan.

    Only the prefix that already ran is checked. Steps after the resume point
    have not happened yet, so changing them is legitimate — a plan may grow a
    tail between deploys without invalidating what is already recorded.
    """
    for record in state.steps:
        if record.index >= len(plan.steps):
            raise PlanChanged(
                f"run {state.run_id} recorded step {record.index} ({record.name!r}) "
                f"but the plan now has only {len(plan.steps)} steps"
            )
        expected = plan.steps[record.index].name
        if expected != record.name:
            raise PlanChanged(
                f"run {state.run_id} recorded step {record.index} as {record.name!r}, "
                f"but the plan given now has {expected!r} there. Resuming would "
                f"execute different work than this run already committed to."
            )


class Engine:
    """Executes plans against a store. Holds no run state of its own."""

    def __init__(
        self,
        store: Store,
        provider: Provider,
        handlers: dict[str, StepHandler],
    ) -> None:
        self.store = store
        self.provider = provider
        self.handlers = handlers

    def submit(self, spec: RunSpec) -> RunState:
        """Register a run. Raises RunAlreadyActive if one is already in flight."""
        missing = [s.name for s in spec.plan.steps if s.name not in self.handlers]
        if missing:
            raise KeyError(f"no handler registered for steps: {', '.join(missing)}")
        return self.store.create_run(uuid.uuid4().hex, spec)

    def submit_or_attach(self, spec: RunSpec) -> RunState:
        """Submit, or return the run already in flight for this key.

        This is what an idempotent HTTP endpoint wants: a repeated request is
        not an error, it is the same run.
        """
        try:
            return self.submit(spec)
        except RunAlreadyActive as exc:
            existing = self.store.load(exc.run_id)
            if existing is None:
                raise RuntimeError(f"run {exc.run_id} vanished between conflict and read") from exc
            return existing

    def run(self, run_id: str, plan: Plan, crash_at: int | None = None) -> RunState:
        """Execute or resume a run to completion.

        ``crash_at`` raises SimulatedCrash before the given step index so the
        demo and tests can prove resumption. It has no effect in normal use.
        """
        state = self.store.load(run_id)
        if state is None:
            raise KeyError(f"unknown run {run_id}")
        if state.status.is_terminal:
            return state

        _assert_plan_matches_history(plan, state)

        with self.store.lock(state.idempotency_key):
            self.store.set_status(run_id, RunStatus.RUNNING)
            payload = state.payload
            previous = {s.name: s.output for s in state.steps}

            for index in range(state.next_index, len(plan.steps)):
                step = plan.steps[index]
                if crash_at is not None and index == crash_at:
                    raise SimulatedCrash(f"process died before step {index} ({step.name})")

                ctx = StepContext(
                    run_id=run_id,
                    index=index,
                    payload=payload,
                    previous=previous,
                    provider=self.provider,
                    max_attempts=step.max_attempts,
                )
                try:
                    output = self.handlers[step.name](ctx)
                except SimulatedCrash:
                    # Models the process dying. A dead process writes nothing,
                    # so neither does this: the run stays resumable, which is
                    # the whole point of the exercise.
                    raise
                except Escalation as exc:
                    self.store.append(
                        StepRecord(
                            run_id=run_id,
                            index=index,
                            name=step.name,
                            outcome=StepOutcome.FAILED,
                            attempts=ctx.attempts,
                            output={"error": exc.reason},
                            tokens_used=ctx.tokens_used,
                        )
                    )
                    self.store.set_status(run_id, RunStatus.ESCALATED, exc.reason)
                    return self._reload(run_id)
                except Exception as exc:
                    # A handler raising something the engine did not expect is a
                    # defect, not a business outcome — so it gets FAILED rather
                    # than ESCALATED, and it is re-raised so the traceback
                    # reaches whoever has to fix it.
                    #
                    # Recording a terminal state first is not tidiness. The
                    # partial unique index only permits one non-terminal run per
                    # idempotency key, so a run left in `running` would block
                    # that key permanently: one bad handler and that customer's
                    # order could never be submitted again.
                    reason = f"{type(exc).__name__}: {exc}"
                    self.store.append(
                        StepRecord(
                            run_id=run_id,
                            index=index,
                            name=step.name,
                            outcome=StepOutcome.FAILED,
                            attempts=ctx.attempts,
                            output={"error": reason},
                            tokens_used=ctx.tokens_used,
                        )
                    )
                    self.store.set_status(run_id, RunStatus.FAILED, reason)
                    raise

                # Deliberately outside the try above, and it must stay there.
                #
                # A handler raising is a defect: the run is marked failed. The
                # store raising is infrastructure: the database is unreachable,
                # nothing is wrong with this run, and it must stay resumable so
                # it finishes when the database comes back. Widening the try to
                # cover this call would turn a transient outage into permanent
                # failure for every run in flight — see
                # tests/test_store_outage.py, which pins the distinction.
                outcome = StepOutcome.HEALED if ctx.was_healed else StepOutcome.OK
                self.store.append(
                    StepRecord(
                        run_id=run_id,
                        index=index,
                        name=step.name,
                        outcome=outcome,
                        attempts=ctx.attempts,
                        output=output,
                        tokens_used=ctx.tokens_used,
                    )
                )
                previous[step.name] = output

            self.store.set_status(run_id, RunStatus.COMPLETED)
            return self._reload(run_id)

    def _reload(self, run_id: str) -> RunState:
        state = self.store.load(run_id)
        if state is None:
            raise RuntimeError(f"run {run_id} disappeared while it was executing")
        return state
