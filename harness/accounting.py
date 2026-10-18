"""Token sums by phase. Round-1 calls are split by phase; every later-round call goes in extra_rounds."""

PHASES = ("plan", "search", "write", "decide")
BUCKETS = PHASES + ("extra_rounds",)


def bucket(calls) -> dict:
    return {
        "calls": len(calls),
        "input": sum(c.input_tokens for c in calls),
        "output": sum(c.output_tokens for c in calls),
        "cached": sum(c.cached_input_tokens or 0 for c in calls),
        "reasoning": sum(c.reasoning_tokens or 0 for c in calls),
    }


def summarize(calls, return_size: list[int]) -> dict:
    unknown = {c.phase for c in calls} - set(PHASES)
    if unknown:
        raise ValueError(f"calls with unknown phases: {sorted(unknown)}")
    summary = {phase: bucket([c for c in calls if c.round == 1 and c.phase == phase]) for phase in PHASES}
    summary["extra_rounds"] = bucket([c for c in calls if c.round > 1])
    summary["total"] = bucket(calls)
    summary["return_size"] = list(return_size)
    summary["usage_gaps"] = {
        "cached_not_reported": sum(c.cached_input_tokens is None for c in calls),
        "reasoning_estimated": sum(c.usage_source.get("reasoning") == "estimated" for c in calls),
        "reasoning_not_reported": sum(c.usage_source.get("reasoning") == "not_reported" for c in calls),
    }
    return summary
