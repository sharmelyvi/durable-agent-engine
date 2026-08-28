# Durable Agent Engine

Crash-resilient agent execution on top of a relational database. A multi-step
agent run survives the process dying mid-flight: it resumes where it stopped,
does not pay again for work already done, and does not repeat an effect that
already reached the outside world.

Durable execution engines solve this — Temporal, Restate and DBOS all persist
run state, and at scale they are the right answer. This is the same mechanism
built from first principles and small enough to read end to end: what the
database actually enforces, where the guarantee stops holding, and why.

The failure it exists for: a worker is restarted, a deploy rolls, or a container
is evicted, and a five-step plan starts again at step one — spending tokens a
second time and, if one of those steps charged a card or sent a message, doing
it twice.

```bash
git clone https://github.com/sharmelyvi/durable-agent-engine
cd durable-agent-engine && make setup && make demo
```

No API key. No database server. No configuration.

The provider is a deterministic stub, and that is the point rather than a
shortcut: the demo asserts an invariant, and an invariant you can only observe
by spending money is one a reader cannot check. Swapping in a real model API is
one class implementing `Provider.complete`.

## What the demo does

It runs a four-step checkout plan, kills the process before the payment step,
then lets a second worker pick the run up.

```
1. Submit the run
  → run 40fc35b1c576 accepted

2. A duplicate request arrives while it is in flight
  ✓ rejected by the database, points at 40fc35b1c576

3. Execute, and kill the process before step 2
  · parse    amount=4200
  · assess   approve (confidence 0.91)
  ✗ process died before step 2 (charge)

4. What survived the crash
  ✓ step 0 parse    ok  0 tokens
  ✓ step 1 assess   ok  18 tokens
  → resume position: step 2
  → gateway charge attempts so far: 0

5. A new worker picks it up
  $ charge   rcpt_f712989224  charged
  · receipt  sent

6. Result
  → status        completed
  → steps         4 of 4
  → tokens        18
  → distinct charges at the gateway: 1

  The run crashed, resumed, and the customer was charged once.
```

`make demo-late` kills it *after* the charge instead, which is the
window that actually costs money. The result is the same: one charge.

## Four invariants, and the mechanism behind each

An invariant is only as good as the thing enforcing it, so each one names its
mechanism rather than describing an intention.

| # | Guarantee | Enforced by | Fails how |
|:--|:--|:--|:--|
| **I-1** | At most one active run per idempotency key | Partial unique index on `(idempotency_key) WHERE status IN (pending, running)` | The database rejects the second insert. An application-level check would lose the race |
| **I-2** | Monotonic, gap-free state progression | Append-only checkpoints keyed on `(run_id, step_index)`; resume position derived by reading them back | A replayed step is a no-op insert, not a second row, so state never goes backwards |
| **I-2b** | Resume applies the plan the run started under | Recorded step names are compared against the plan before resuming; a mismatch raises `PlanChanged` | Resume position is an index, so a plan edited between deploys would point it at different work |
| **I-3** | At most one external effect | Deterministic token from `blake2b(run_id:step_index)`, handed to the external system | Same token on every replay, so a gateway that deduplicates rejects the repeat |
| **I-4** | Guaranteed terminal state, with a stated cause | Retries capped, corrections capped at two, then escalation; an unexpected exception marks the run failed before re-raising | No run stalls half-done. A run left non-terminal would block its idempotency key forever, since the index in I-1 permits only one active run per key |

**I-3** is the one that matters most and is easiest to get wrong. There is a
window between performing an effect and committing its checkpoint that no single
database transaction can close, because the effect is not in that database. A
deterministic token moves the deduplication to the system that owns the effect,
which is the only place it can actually be enforced.

The guarantee is therefore conditional, and the condition is worth stating
plainly: it holds for an external system that honours the token. Gateways that
implement idempotency keys — MercadoPago, Stripe and most others — do. One that
ignores the key will happily charge twice, and no amount of care on this side of
the call changes that.

Within that condition, **the guarantee does not depend on the lock**. Locks fail: a lease
can expire while its holder is still working, and a session lock outlives a
handler that hangs. `tests/test_lock_is_an_optimisation.py` removes the lock
entirely, runs three workers into the same effect simultaneously, and asserts
the gateway still sees one token. The lock stops the engine paying twice for the
same work; the token is what stops the customer being charged twice.

## Verifying the claims

Everything above is executable.

```bash
make verify         # lint, types, the suite, both demo modes
make test-postgres  # the same suite again against real PostgreSQL
make bench          # measure the token claim rather than assert it
```

`make types` runs mypy over `src/`, and CI fails on a type error. The `Store`
protocol is the load-bearing abstraction here — two backends behind one
interface — and a protocol nothing checks is documentation wearing a contract's
clothes.

The suite is parametrised over both backends, so every invariant below is
asserted twice. Two exceptions are marked and skipped rather than faked: lease
expiry is a SQLite concern, and transaction-scoped lock release is a Postgres
one. Each backend is tested for the property it actually has.

Each invariant has tests that try to break it:

- **I-1** — `test_concurrency.py` releases sixteen threads on a barrier so they
  submit the same key at the same instant, ten trials, and asserts exactly one
  wins.
- **I-2** — `test_chaos.py` checks that checkpoint indices stay contiguous under
  provider failure, so resume position is always defined. `test_plan_drift.py`
  covers the other half: a run whose plan was edited mid-flight is refused
  rather than resumed against the wrong steps.
- **I-3** — `test_resume.py` kills a run at every step position and asserts the
  effect token is identical across replays; one test deletes a checkpoint to
  force a genuine re-execution and checks the token still matches.
- **I-4** — `test_chaos.py` runs the engine against a provider failing at rates
  from 0% to 100% and asserts every run reaches a terminal state carrying a
  reason. `test_handler_defects.py` covers the other direction: a handler
  raising something the engine never expected still lands terminal, and the
  idempotency key stays usable afterwards.

## Measured, including where it loses

When a model returns output that violates the schema, the reflexive fix is to
resend the whole prompt. Sending the model its own answer and the faults in it
is cheaper — but not always, and the benchmark says where the line is.

```
context                              naive resend*   correction    saved
short (a one-line question)                     43          127        —
medium (a page of policy)                      636          127      80%
long (a full customer history)                3071          127      96%
very long (history + policy + logs)           9159          127      99%
```

`*` A bare resend never recovers. A deterministic model given identical input
returns its identical invalid answer, so the round is spent and the run
escalates anyway. Every figure in that column buys nothing, which is why the
short row shows no saving rather than a negative one.

The correction is a fixed 127 tokens whether it replaces a one-line question or
nine thousand tokens of history. That is the property worth having: the cost of
a retry stops scaling with the context it is repairing.

It carries the model's own previous answer, and that is not padding. A fault
list names only the fields that *failed* — a field that validated has no fault,
so it is absent. A correction built from faults alone therefore asks for the
full object while withholding the part the model got right, and the model has to
invent it. An invented value in a valid shape passes validation and reaches the
caller as fact; on a decision schema that is approve or deny, decided by a
guess. Thirty tokens buy that away.

One retry does not use the correction at all. When a response contained no JSON
there is no draft to repair, so the original task goes back with an explicit
instruction about the format. The branch is about what can be repaired, not
about which prompt is shorter — an earlier version chose on token count and got
this exactly backwards.

Token counts are a character-count estimate at 4 chars/token, applied
identically to both arms. The absolute numbers are approximate; the ratio and
the recovery outcome are the claim. Reproduce with `make bench`.

## Limits

Stated here rather than discovered later. Four shape the engine; the operational
trade-offs behind them are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#operational-trade-offs).

- **SQLite is the default, not the only backend.** It is what makes the demo run
  with no setup, and its lock is a lease row that serialises a single machine.
  `PostgresStore` uses native advisory locks and coordinates workers across
  machines; `make test-postgres` runs the whole suite against it.
- **Plans are sequences, not graphs.** Steps run in order, in-process. There is
  no conditional branch, no parallel fan-out and no worker pool. Conditional
  logic lives inside a step's handler today; explicit dependencies between steps
  would be the change that makes it a DAG.
- **A hung handler holds its lock.** The engine hands network calls to the
  handler and does not impose a deadline. A handler that blocks forever — an
  HTTP call with no timeout — keeps the Postgres session lock indefinitely,
  because the connection never closes. Bound your handlers. The at-most-once
  guarantee survives this anyway, for the reason in I-3.
- **No compensation.** A run that escalates after a committed effect leaves that
  effect in place; there is no rollback step. Compensating actions are the
  obvious next layer and are deliberately out of scope.

## Layout

```
src/durable_agent/
  models.py     contracts, effect tokens, process-stable lock ids
  store.py      SQLite backend: partial unique index, lease locks
  postgres.py   PostgreSQL backend: native advisory locks, JSONB checkpoints
  engine.py     step execution, resume, retries, escalation
  healing.py    schema faults reduced to a delta, and when not to use it
  providers.py  provider protocol, mock, and the fault injector
  demo.py       the crash story above
tests/          resume, concurrency, chaos, lock identity — run twice,
                once per backend
scripts/        the benchmark behind the numbers
docs/           architecture and design decisions
```

MIT licensed.
