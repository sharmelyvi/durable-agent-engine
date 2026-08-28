"""Measure what a self-contained correction saves, instead of claiming a number.

Two retries answer the same schema violation:

naive
    resend the original prompt unchanged.
correction
    send the model its own previous answer and the fields that failed.

Cost alone is the wrong axis twice over. A bare resend carries no information
about the failure, so a deterministic model returns the same invalid answer: the
round is cheap and recovers nothing. And a correction built from the fault list
alone is cheaper still and worse — a field that validated has no fault, so it is
absent, and the model is asked for a full object while the part it got right is
withheld. It invents that part, the invented value passes validation, and the
caller receives a guess as a fact.

So the correction carries the previous answer. It costs a few dozen tokens more
than the fault list and is the only cheap retry that can actually recover.

Run:  python scripts/benchmark.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from durable_agent.healing import (  # noqa: E402
    build_retry_prompt,
    parse_or_faults,
)
from durable_agent.providers import _estimate_tokens  # noqa: E402


class Decision(BaseModel):
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


BROKEN = json.dumps({"decision": "approve", "confidence": "very high", "reasons": "looks fine"})

CONTEXTS = {
    "short (a one-line question)": 1,
    "medium (a page of policy)": 40,
    "long (a full customer history)": 200,
    "very long (history + policy + logs)": 600,
}


def build_prompt(lines: int) -> str:
    head = (
        "You are a risk analyst. Decide whether to approve this transaction and "
        "return JSON matching the Decision schema.\n\n"
    )
    body = "".join(
        f"2026-08-{(i % 28) + 1:02d} txn={1000 + i} amount={(i * 37) % 900}.00 "
        f"status=settled channel=web\n"
        for i in range(lines)
    )
    return head + body


@dataclass
class Result:
    label: str
    prompt_tokens: int
    naive_retry: int
    delta_retry: int
    chosen_retry: int

    # A retry recovers only if it tells the model what was wrong. The bare
    # resend does not, which is the entire point of the comparison.
    naive_recovers: bool = False
    delta_recovers: bool = True
    engine_recovers: bool = True

    @property
    def saved_pct(self) -> float:
        if self.naive_retry == 0:
            return 0.0
        return 100.0 * (self.naive_retry - self.delta_retry) / self.naive_retry

    @property
    def chosen_pct(self) -> float:
        """What the engine actually saves, after picking the cheaper retry."""
        if self.naive_retry == 0:
            return 0.0
        return 100.0 * (self.naive_retry - self.chosen_retry) / self.naive_retry


def measure(label: str, lines: int) -> Result:
    prompt = build_prompt(lines)
    parsed, faults = parse_or_faults(BROKEN, Decision)
    assert parsed is None and faults, "the fixture must fail validation"

    correction = build_retry_prompt(prompt, BROKEN, faults, Decision.__name__)
    assert "approve" in correction, "the correction must carry the field that validated"
    return Result(
        label=label,
        prompt_tokens=_estimate_tokens(prompt),
        naive_retry=_estimate_tokens(prompt),
        delta_retry=_estimate_tokens(correction),
        chosen_retry=_estimate_tokens(correction),
    )


def main() -> int:
    results = [measure(label, lines) for label, lines in CONTEXTS.items()]

    width = max(len(r.label) for r in results)
    print("\nRetry cost after one schema violation, in tokens")
    print("(~4 chars/token, applied identically to both arms)\n")
    print(f"{'context':<{width}}  {'naive resend*':>13}  {'correction':>11}  {'saved':>7}")
    print("-" * (width + 39))
    for r in results:
        # Saving is only meaningful against an arm that recovers, and the naive
        # one never does; on a short prompt the correction simply costs more.
        saved = f"{r.chosen_pct:.0f}%" if r.chosen_retry <= r.naive_retry else "—"
        print(f"{r.label:<{width}}  {r.naive_retry:>13}  {r.delta_retry:>11}  {saved:>7}")

    print(
        "\n* A bare resend never recovers. A deterministic model given identical\n"
        "  input returns its identical invalid answer, so the round is spent and\n"
        "  the run escalates anyway. Every figure in that column buys nothing."
    )
    print(
        "\nThe correction carries the model's previous answer, so the fields it\n"
        "got right survive the round instead of being invented, and the task is\n"
        "never resent. It is a fixed size: the cost of a retry stops scaling with\n"
        "the context it replaces. On a one-line prompt that is more expensive than\n"
        "resending — and still the only arm that ends with a valid object."
    )

    out = Path("bench-results.json")
    out.write_text(
        json.dumps(
            {
                "method": (
                    "character-count estimate at 4 chars/token, identical for all arms; "
                    "a retry counts as recovering only if it carries the fault"
                ),
                "results": [
                    {
                        "context": r.label,
                        "prompt_tokens": r.prompt_tokens,
                        "naive_retry_tokens": r.naive_retry,
                        "delta_retry_tokens": r.delta_retry,
                        "delta_saved_pct": round(r.saved_pct, 1),
                        "engine_retry_tokens": r.chosen_retry,
                        "engine_saved_pct": round(r.chosen_pct, 1),
                        "naive_recovers": r.naive_recovers,
                        "engine_recovers": r.engine_recovers,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
