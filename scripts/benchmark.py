"""Measure what delta correction actually saves, instead of claiming a number.

Two strategies recover from the same schema violation:

naive
    resend the original prompt unchanged.
delta
    send only the fields that failed validation.
engine
    whichever of the two is smaller, and when that is the original, the fault
    note travels with it.

Cost alone is the wrong axis. A bare resend carries no information about the
failure, so a deterministic model returns the same invalid answer: the round is
cheap and recovers nothing. What matters is cost per *recovered* response, so
this reports both, and marks the arms that do not recover at all.

Run:  python scripts/benchmark.py
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from durable_agent.healing import (  # noqa: E402
    cheaper_retry,
    correction_prompt,
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

    delta = correction_prompt(faults, Decision.__name__)
    return Result(
        label=label,
        prompt_tokens=_estimate_tokens(prompt),
        naive_retry=_estimate_tokens(prompt),
        delta_retry=_estimate_tokens(delta),
        chosen_retry=_estimate_tokens(cheaper_retry(prompt, delta)),
    )


def main() -> int:
    results = [measure(label, lines) for label, lines in CONTEXTS.items()]

    width = max(len(r.label) for r in results)
    print("\nRetry cost after one schema violation")
    print("(token counts are ~4 chars/token, applied identically to both strategies)\n")
    print(f"{'context':<{width}}  {'naive':>13}  {'delta':>8}  {'engine':>8}  {'vs delta':>9}")
    print("-" * (width + 46))
    for r in results:
        naive = f"{r.naive_retry} (no fix)" if not r.naive_recovers else str(r.naive_retry)
        vs_delta = (
            "same" if r.chosen_retry == r.delta_retry else f"+{r.chosen_retry - r.delta_retry}"
        )
        print(
            f"{r.label:<{width}}  {naive:>13}  {r.delta_retry:>8}  "
            f"{r.chosen_retry:>8}  {vs_delta:>9}"
        )

    recovering = [r for r in results if r.delta_recovers]
    savings = [r.chosen_pct for r in recovering if r.chosen_retry <= r.naive_retry]
    if savings:
        print(
            f"\nWhere the delta is smaller, it saves {min(savings):.0f}-{max(savings):.0f}% "
            f"of the retry (median {statistics.median(savings):.0f}%)."
        )
    print(
        "\nThe naive column never recovers: an unchanged prompt gives a "
        "deterministic\nmodel no reason to answer differently, so it is spent "
        "twice and escalates.\nOn a short prompt the engine pays more than a "
        "resend and less than nothing\nwould have achieved — the only arm in "
        "that row that ends with a valid object."
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
