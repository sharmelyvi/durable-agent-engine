# Benchmark: what delta correction actually costs

Reproduce with `make bench`. Results are written to `bench-results.json`.

## What is being compared

When a model returns output that a schema rejects, the run needs a second
attempt. Two ways to ask for it:

**naive** — resend the original prompt unchanged.
**delta** — send only the fields that failed validation, with what was wrong
with each.

Both end with a valid object. The question is what the retry costs.

## Method and its limits

Token counts are a character-count estimate at four characters per token,
applied identically to both arms. This is not a real tokenizer and the absolute
numbers should not be used to predict a bill. The ratio between the two arms is
what the benchmark is measuring, and that ratio is stable under any tokenizer
that is roughly linear in text length.

The fixture is one schema violation — a `float` field returned as the string
`"very high"`, plus a `list[str]` returned as a bare string. The correction
prompt therefore carries two faults, which is a realistic middle case: a single
fault would make delta correction look better than it usually is.

## Results

```
context                                naive    delta    delta%   engine  engine%
short (a one-line question)               43       88   -104.7%       43     0.0%
medium (a page of policy)                636       88     86.2%       88    86.2%
long (a full customer history)          3071       88     97.1%       88    97.1%
very long (history + policy + logs)     9159       88     99.0%       88    99.0%
```

## The result that changed the code

On a one-line prompt, delta correction costs **more than twice as much** as
resending the original. The fault description is longer than the prompt it
replaces.

This is the case the technique is usually presented without. It is also easy to
hit: short, tightly scoped prompts are exactly what a well-factored agent step
looks like.

`healing.build_retry_prompt()` does not choose on size. It sends the correction
whenever the previous response can be repaired from itself, and falls back to
the original task only when the response contained no JSON at all — there being
nothing to repair in that case.

An earlier version chose on token count, and that was wrong in a way the numbers
hid: the cheapest correction omitted the fields that had validated, so the model
had to invent them, and an invented value in a valid shape passes validation.
The fix was not a better heuristic. It was making the cheap retry correct, after
which size stopped being the question.

## What the numbers do not say

- Nothing about *quality*. A delta correction might produce a worse answer than
  a full resend even when it is cheaper. The benchmark measures cost only.
- Nothing about latency. Fewer input tokens usually means a faster response, but
  that is an inference, not a measurement here.
- Nothing about multi-fault responses at scale. Two faults are measured; a
  response with fifteen invalid fields would carry a much larger correction and
  shift the crossover point to the right.
