"""Measure what delta correction actually saves, instead of claiming a number.

Two strategies recover from the same schema violation:

naive
    resend the original prompt and hope for a better answer.
delta
    send only the fields that failed validation.

Both end with a valid object. The difference is what the retry costs, and that
difference depends on how long the original prompt was — which is why this
reports a curve rather than a single headline percentage. A short prompt saves
little. That is a real result and it belongs in the output.

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
    print(
        f"{'context':<{width}}  {'naive':>7}  {'delta':>7}  {'delta%':>8}  "
        f"{'engine':>7}  {'engine%':>8}"
    )
    print("-" * (width + 44))
    for r in results:
        print(
            f"{r.label:<{width}}  {r.naive_retry:>7}  {r.delta_retry:>7}  "
            f"{r.saved_pct:>7.1f}%  {r.chosen_retry:>7}  {r.chosen_pct:>7.1f}%"
        )

    savings = [r.chosen_pct for r in results]
    print(
        f"\nmedian saving {statistics.median(savings):.1f}%  ·  "
        f"range {min(savings):.1f}% to {max(savings):.1f}%"
    )
    print(
        "\nThe delta column is the raw technique; the engine column is what the\n"
        "engine actually spends, because cheaper_retry() falls back to resending\n"
        "the original whenever the correction would cost more. That fallback is\n"
        "why the engine never posts a negative saving.\n"
    )

    out = Path("bench-results.json")
    out.write_text(
        json.dumps(
            {
                "method": "character-count estimate at 4 chars/token, identical for both arms",
                "results": [
                    {
                        "context": r.label,
                        "prompt_tokens": r.prompt_tokens,
                        "naive_retry_tokens": r.naive_retry,
                        "delta_retry_tokens": r.delta_retry,
                        "delta_saved_pct": round(r.saved_pct, 1),
                        "engine_retry_tokens": r.chosen_retry,
                        "engine_saved_pct": round(r.chosen_pct, 1),
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
