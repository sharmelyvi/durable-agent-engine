"""A runnable story: kill a payment run mid-flight and watch it recover.

    python -m durable_agent.demo

No API key, no database server, no configuration. The point is that a reader
can verify the claims in the README in the time it takes to read them.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from .engine import Engine, SimulatedCrash, StepContext
from .models import Plan, RunSpec, RunStatus, Step
from .providers import ChaosProvider, MockProvider
from .store import RunAlreadyActive, SQLiteStore

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def h(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print(DIM + "─" * len(text) + RESET)


def line(marker: str, text: str, colour: str = "") -> None:
    print(f"  {colour}{marker}{RESET} {text}")


class Assessment(BaseModel):
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)


PLAN = Plan(
    name="checkout",
    steps=(
        Step(name="parse"),
        Step(name="assess"),
        Step(name="charge", effect=True),
        Step(name="receipt"),
    ),
)


class Gateway:
    """Stands in for a payment provider that deduplicates on a token."""

    def __init__(self) -> None:
        self.seen: dict[str, str] = {}
        self.attempts = 0

    def charge(self, token: str, amount: int) -> tuple[str, bool]:
        self.attempts += 1
        if token in self.seen:
            return self.seen[token], True  # replayed, not charged again
        self.seen[token] = f"rcpt_{token[:10]}"
        return self.seen[token], False


def build(db: Path, gateway: Gateway, chaos: bool) -> Engine:
    store = SQLiteStore(db)
    store.setup()

    def parse(ctx: StepContext) -> dict:
        line("·", f"parse    amount={ctx.payload['amount']}")
        return {"amount": ctx.payload["amount"]}

    def assess(ctx: StepContext) -> dict:
        answer = ctx.ask("Assess this transaction for risk.", Assessment)
        line("·", f"assess   {answer.decision} (confidence {answer.confidence})")
        return answer.model_dump()

    def charge(ctx: StepContext) -> dict:
        receipt, replayed = gateway.charge(ctx.effect_token, ctx.payload["amount"])
        marker = "replayed, not charged again" if replayed else "charged"
        line("$", f"charge   {receipt}  {YELLOW}{marker}{RESET}", GREEN)
        return {"receipt": receipt}

    def receipt(ctx: StepContext) -> dict:
        line("·", "receipt  sent")
        return {"sent": True}

    provider = ChaosProvider(failure_rate=0.5, seed=11) if chaos else MockProvider()
    return Engine(
        store=store,
        provider=provider,
        handlers={"parse": parse, "assess": assess, "charge": charge, "receipt": receipt},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crash-at", type=int, default=2, help="step index to die on")
    ap.add_argument("--chaos", action="store_true", help="use a failing model provider")
    args = ap.parse_args(argv)

    db = Path(tempfile.mkdtemp()) / "demo.db"
    gateway = Gateway()
    engine = build(db, gateway, args.chaos)

    spec = RunSpec(plan=PLAN, payload={"amount": 4200}, idempotency_key="order-42-renewal")

    h("1. Submit the run")
    run = engine.submit(spec)
    line("→", f"run {CYAN}{run.run_id[:12]}{RESET} accepted")

    h("2. A duplicate request arrives while it is in flight")
    try:
        engine.submit(spec)
        line("✗", "the duplicate was accepted — the index is not protecting", RED)
    except RunAlreadyActive as exc:
        line("✓", f"rejected by the database, points at {exc.run_id[:12]}", GREEN)

    h(f"3. Execute, and kill the process before step {args.crash_at}")
    try:
        engine.run(run.run_id, PLAN, crash_at=args.crash_at)
    except SimulatedCrash as exc:
        line("✗", f"{RED}{exc}{RESET}")

    state = engine.store.load(run.run_id)
    assert state is not None
    h("4. What survived the crash")
    for s in state.steps:
        line(
            "✓",
            f"step {s.index} {s.name:<8} {DIM}{s.outcome}  {s.tokens_used} tokens{RESET}",
            GREEN,
        )
    line("→", f"resume position: step {state.next_index}")
    line("→", f"gateway charge attempts so far: {gateway.attempts}")

    h("5. A new worker picks it up")
    final = engine.run(run.run_id, PLAN)

    h("6. Result")
    colour = GREEN if final.status is RunStatus.COMPLETED else YELLOW
    line("→", f"status        {colour}{final.status}{RESET}")
    line("→", f"steps         {len(final.steps)} of {len(PLAN.steps)}")
    line("→", f"tokens        {final.tokens_used}")
    line("→", f"distinct charges at the gateway: {BOLD}{len(gateway.seen)}{RESET}")
    if final.error:
        line("→", f"escalated because: {YELLOW}{final.error}{RESET}")

    # The invariant is not "a charge happened". It is "no more than one charge
    # happened, and money only moved on a run that actually completed". An
    # escalated run with zero charges is the system working, not failing.
    charges = len(gateway.seen)
    if final.status is RunStatus.COMPLETED:
        ok, verdict = charges == 1, "crashed, resumed, and the customer was charged once"
    else:
        ok, verdict = charges == 0, "escalated before touching money, and nothing was charged"

    print()
    if ok:
        print(f"  {GREEN}The run {verdict}.{RESET}\n")
    else:
        print(
            f"  {RED}Invariant broken: status {final.status} with "
            f"{charges} distinct charges.{RESET}\n"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
