"""Why the lock id is not Python's hash().

A distributed lock is worthless if two workers disagree on what to lock. Python
randomises string hashing per process, so `hash(key)` is a different number in
every worker. This test demonstrates the failure rather than asserting the fix,
because the failure is the reason the code looks the way it does.
"""

from __future__ import annotations

import subprocess
import sys

from durable_agent.models import advisory_lock_id

KEY = "order-42"
SEEDS = ("1", "2", "12345", "99999")


def _in_subprocess(expr: str, seed: str) -> str:
    """Run an expression in a fresh interpreter with a fixed hash seed."""
    return subprocess.run(
        [sys.executable, "-c", expr],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    ).stdout.strip()


def test_python_hash_is_not_stable_across_processes() -> None:
    """The bug this module exists to avoid."""
    values = {_in_subprocess(f"print(hash({KEY!r}) & 0x7FFFFFFFFFFFFFFF)", seed) for seed in SEEDS}
    assert len(values) > 1, (
        "expected hash() to differ across interpreters; if this ever passes with "
        "one value, re-check the assumption before relying on it"
    )


def test_advisory_lock_id_is_stable_across_processes() -> None:
    """The property a distributed lock actually needs."""
    expr = (
        "import sys; sys.path.insert(0, 'src'); "
        "from durable_agent.models import advisory_lock_id; "
        f"print(advisory_lock_id({KEY!r}))"
    )
    values = {_in_subprocess(expr, seed) for seed in SEEDS}
    assert values == {str(advisory_lock_id(KEY))}


def test_advisory_lock_id_fits_postgres_bigint() -> None:
    """pg_advisory_lock takes a signed 64-bit integer; stay inside it."""
    for key in ("a", "order-42", "x" * 200, "unicode-ñ-key"):
        assert 0 <= advisory_lock_id(key) <= 0x7FFFFFFFFFFFFFFF


def test_distinct_keys_do_not_collide_in_practice() -> None:
    ids = {advisory_lock_id(f"order-{i}") for i in range(10_000)}
    assert len(ids) == 10_000
