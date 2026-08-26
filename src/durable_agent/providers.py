"""Model providers, and the fault injector used to prove the engine survives them.

The engine never talks to a real API in tests or in the demo. That is a
deliberate constraint: a reliability claim you can only check by spending money
is not a claim a reader can verify, and an unverifiable claim does not belong in
a repository that exists to be inspected.

``ChaosProvider`` reproduces the four ways a model call actually fails in
production — not the way tutorials assume it fails:

1. the response is not JSON at all (prose, markdown fences, an apology);
2. the response is JSON but violates the schema (wrong type, missing field);
3. the call times out;
4. the connection drops.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Protocol


class ProviderTimeout(Exception):
    """The provider did not answer inside the budget."""


class ProviderUnavailable(Exception):
    """The connection failed. Distinct from a timeout on purpose: one means the
    request may have been received, the other means it was not."""


@dataclass(frozen=True)
class Completion:
    """What a provider returns. Text, not a parsed object — parsing is the
    engine's job and it happens behind a schema."""

    text: str
    tokens_in: int
    tokens_out: int

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Provider(Protocol):
    def complete(self, prompt: str) -> Completion: ...


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token.

    Deliberately crude and clearly labelled. The benchmark compares two prompts
    measured the same way, so the ratio is meaningful even though the absolute
    number is an approximation.
    """
    return max(1, len(text) // 4)


class MockProvider:
    """Always returns a well-formed answer. The happy path, for baselines."""

    def __init__(self, answer: dict[str, Any] | None = None) -> None:
        self.answer = answer or {"decision": "approve", "confidence": 0.91}
        self.calls = 0

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        body = json.dumps(self.answer)
        return Completion(
            text=body, tokens_in=_estimate_tokens(prompt), tokens_out=_estimate_tokens(body)
        )


class ChaosProvider:
    """Fails on purpose, reproducibly.

    Seeded so a failing test can be replayed exactly. A chaos suite that cannot
    be replayed reports flakiness, not resilience.
    """

    MODES = ("not_json", "bad_schema", "timeout", "unavailable")

    def __init__(
        self,
        failure_rate: float = 0.4,
        seed: int = 1234,
        answer: dict[str, Any] | None = None,
        modes: tuple[str, ...] | None = None,
        heal_after: int = 1,
    ) -> None:
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be between 0 and 1")
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        self.answer = answer or {"decision": "approve", "confidence": 0.91}
        self.modes = modes or self.MODES
        self.heal_after = heal_after
        self.calls = 0
        self.failures: list[str] = []

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        # A correction prompt is expected to succeed after `heal_after` attempts,
        # which is what makes the self-healing path observable instead of endless.
        correcting = "CORRECTION" in prompt
        if correcting and len(self.failures) >= self.heal_after:
            body = json.dumps(self.answer)
            return Completion(
                text=body, tokens_in=_estimate_tokens(prompt), tokens_out=_estimate_tokens(body)
            )

        if self.rng.random() >= self.failure_rate:
            body = json.dumps(self.answer)
            return Completion(
                text=body, tokens_in=_estimate_tokens(prompt), tokens_out=_estimate_tokens(body)
            )

        mode = self.rng.choice(self.modes)
        self.failures.append(mode)
        if mode == "timeout":
            raise ProviderTimeout("provider exceeded the deadline")
        if mode == "unavailable":
            raise ProviderUnavailable("connection reset by peer")
        if mode == "not_json":
            body = (
                "Sure! Here's the result:\n\n```json\n{ decision: approve, }\n```\n"
                "Let me know if you need anything else."
            )
        else:  # bad_schema
            body = json.dumps({"decision": "approve", "confidence": "very high"})
        return Completion(
            text=body, tokens_in=_estimate_tokens(prompt), tokens_out=_estimate_tokens(body)
        )
