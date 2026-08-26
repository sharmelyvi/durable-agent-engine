# Durable Agent Engine

Crash-resilient agent execution on top of a relational database. A multi-step
agent run survives the process dying mid-flight: it resumes where it stopped,
does not pay again for work already done, and does not repeat an effect that
already reached the outside world.

Most agent frameworks keep run state in memory. That is fine until a worker is
restarted, a deploy rolls, or a container is evicted — and then a five-step plan
starts again at step one, spending tokens a second time and, if one of those
steps charged a card or sent a message, doing it twice.

```bash
git clone https://github.com/sharmelyvi/durable-agent-engine
cd durable-agent-engine && make setup && make demo
```

No API key. No database server. No configuration.

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

`make demo --crash-at 3` kills it *after* the charge instead, which is the
window that actually costs money. The result is the same: one charge.

## Four invariants, and the mechanism behind each

An invariant is only as good as the thing enforcing it, so each one names its
mechanism rather than describing an intention.

| # | Guarantee | Enforced by | Fails how |
|:--|:--|:--|:--|
| **I-1** | At most one active run per idempotency key | Partial unique index on `(idempotency_key) WHERE status IN (pending, running)` | The database rejects the second insert. An application-level check would lose the race |
| **I-2** | Monotonic, gap-free state progression | Append-only checkpoints keyed on `(run_id, step_index)`; resume position derived by reading them back | A replayed step is a no-op insert, not a second row, so state never goes backwards |
| **I-3** | At most one external effect | Deterministic token from `blake2b(run_id:step_index)`, handed to the external system | Same token on every replay, so a gateway that deduplicates rejects the repeat |
| **I-4** | Guaranteed terminal state, with a stated cause | Retries capped, correction rounds capped at two, then escalation | No run stalls half-done, and an escalated run records *why* — the field or the provider that failed |

**I-3** is the one that matters most and is easiest to get wrong. There is
a window between performing an effect and committing its checkpoint that no
single database transaction can close, because the effect is not in that
database. A deterministic token moves the deduplication to the system that owns
the effect, which is the only place it can actually be enforced.

## Verifying the claims

Everything above is executable.

```bash
make verify         # lint, the suite, both demo modes
make test-postgres  # the same suite again against real PostgreSQL
make bench          # measure the token claim rather than assert it
```

The suite is parametrised over both backends, so every invariant below is
asserted twice. Two exceptions are marked and skipped rather than faked: lease
expiry is a SQLite concern, and transaction-scoped lock release is a Postgres
one. Each backend is tested for the property it actually has.

Each invariant has tests that try to break it:

- **I-1** — `test_concurrency.py` releases sixteen threads on a barrier so they
  submit the same key at the same instant, ten trials, and asserts exactly one
  wins.
- **I-2** — `test_chaos.py` checks that checkpoint indices stay contiguous under
  provider failure, so resume position is always defined.
- **I-3** — `test_resume.py` kills a run at every step position and asserts the
  effect token is identical across replays; one test deletes a checkpoint to
  force a genuine re-execution and checks the token still matches.
- **I-4** — `test_chaos.py` runs the engine against a provider failing at rates
  from 0% to 100% and asserts every run reaches a terminal state carrying a
  reason.

## Measured, including where it loses

When a model returns output that violates the schema, the reflexive fix is to
resend the whole prompt. Sending only the fields that failed is cheaper — but
not always, and the benchmark says where the line is.

```
context                                naive    delta    delta%   engine  engine%
short (a one-line question)               43       88   -104.7%       43     0.0%
medium (a page of policy)                636       88     86.2%       88    86.2%
long (a full customer history)          3071       88     97.1%       88    97.1%
very long (history + policy + logs)     9159       88     99.0%       88    99.0%
```

On a one-line prompt the fault description is longer than the prompt it
replaces, so delta correction costs **more than twice as much**. `cheaper_retry()`
sends whichever retry is smaller, which is the `engine` column: never worse than
resending, and 86–99% cheaper once there is real context to avoid repeating.
Reproduce it with `make bench`.

Token counts are a character-count estimate at 4 chars/token, applied
identically to both arms. The absolute numbers are approximate; the ratio is
the claim.

## Limits

Stated here rather than discovered later.

- **SQLite is the default, not the only backend.** It is what makes the demo run
  with no setup, and its lock is a lease row that serialises a single machine.
  `PostgresStore` uses native advisory locks and coordinates workers across
  machines; `make test-postgres` runs the whole suite against it. What is *not*
  here is a connection pool — connections are opened per operation, which is
  honest at this scale and wrong at a larger one.
- **Steps run in-process and sequentially.** There is no worker pool, no queue
  and no parallel branch. The durability model does not depend on that, but the
  throughput story does.
- **No compensation.** A run that escalates after a committed effect leaves that
  effect in place; there is no rollback step. Compensating actions are the
  obvious next layer and are deliberately out of scope for now.
- **The provider is a stub.** No adapter for a real model API ships here on
  purpose: a claim that can only be checked by spending money is not a claim a
  reader can check.
- **The token estimator is crude.** Four characters per token, not a real
  tokenizer. Fine for a ratio, wrong for a bill.

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
