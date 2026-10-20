"""Budget calibration (task C6): typical sizes from the released BrowseComp-Plus runs and from our own runs.

    python -m harness.calibrate [runs/pilot ...] [--released data/released_runs] [--tokenizer tiktoken:o200k_base]

The released runs (python -m harness.data --released) are one agent per question on the same BM25 index,
5 hits per search, so they show how much search output a whole question takes and how long answers are;
--tokenizer counts their tokens. Our runs show, per subagent, how much text it saw (the uncapped raw report)
and flagged (the uncapped quotes report), how long each format's report came out, and how often the cap cut
it. Run sizes use the orchestrator's tokenizer, the one the caps are measured in. A cap at the 90th
percentile of an uncapped size binds for one subagent in ten.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from . import tokens
from .config import FORMATS, RunConfig
from .pipeline import results

WRITTEN = ("short", "medium", "long", "claims")
COLUMNS = f"{'':<36}{'count':>7}{'mean':>9}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}{'max':>8}"


def percentile(values, p: float) -> float:
    values = sorted(values)
    return values[round(p * (len(values) - 1))]


def describe(name: str, values) -> str:
    values = list(values)
    if not values:
        return f"{name:<36}{0:>7}"
    cells = "".join(f"{percentile(values, p):>8.0f}" for p in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    return f"{name:<36}{len(values):>7}{statistics.fmean(values):>9.1f}{cells}"


def released_report(path: Path, tokenizer: str) -> str:
    stats, completed = defaultdict(list), 0
    with open(path, encoding="utf-8") as f:
        for line in filter(str.strip, f):
            run = json.loads(line)
            searches = [item for item in run["result"] if item["type"] == "tool_call"]
            snippets, docids = [], set()
            for item in searches:
                try:
                    hits = json.loads(item["output"])
                except (TypeError, ValueError):
                    hits = None
                if isinstance(hits, list):
                    snippets += [hit.get("snippet") or "" for hit in hits]
                    docids |= {hit["docid"] for hit in hits}
                else:
                    snippets.append(str(item["output"] or ""))
            answers = [i["output"] for i in run["result"] if i["type"] == "output_text" and isinstance(i["output"], str)]
            completed += run.get("status") == "completed"
            stats["searches"].append(len(searches))
            stats["search output tokens"].append(sum(tokens.count(tokenizer, s) for s in snippets))
            stats["distinct documents retrieved"].append(len(docids))
            stats["final answer tokens"].append(tokens.count(tokenizer, answers[-1]) if answers else 0)
    lines = [f"== Released run {path}: {len(stats['searches'])} questions, {completed} completed "
             f"(tokens: {tokenizer})", COLUMNS]
    return "\n".join(lines + [describe(name, values) for name, values in stats.items()])


def _passages(tokenizer: str, trajectory, pids) -> int:
    return sum(tokens.count(tokenizer, f"[{pid}] {trajectory.seen[pid]}") for pid in pids)


def round_up(value: float, step: int = 100) -> int:
    return int(math.ceil(value / step) * step)


def run_report(run_dir: Path) -> str:
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    if cfg.mode != "multi":
        return f"== {run_dir}: a single-agent run has no subagent reports to calibrate"
    tokenizer = cfg.endpoints[cfg.roles.orchestrator].tokenizer
    subagents, reports, returns = {}, defaultdict(list), defaultdict(list)
    for _, result in results(run_dir):
        if not result.rounds:
            continue
        first = result.rounds[0]
        for trajectory in first.trajectories:  # stage 1 is shared by every format, so count each subagent once
            subagents.setdefault((result.qid, result.n, result.seed, trajectory.subagent), trajectory)
        reports[result.format] += first.reports
        returns[(result.format, result.n)].append(sum(r.tokens for r in first.reports))
    seen = [_passages(tokenizer, t, t.seen) for t in subagents.values()]
    flagged = [_passages(tokenizer, t, t.flagged) for t in subagents.values() if t.flagged]
    budgets = cfg.budgets.model_dump()

    lines = [f"== {run_dir}: {len(subagents)} first-round subagents (tokens: {tokenizer})", COLUMNS,
             describe("uncapped raw: all passages read", seen),
             describe("uncapped quotes: flagged text", flagged)]
    if len(flagged) < len(subagents):
        lines.append(f"  ({len(subagents) - len(flagged)} subagents flagged nothing; quotes falls back to ranked passages)")
    notes = []
    for fmt in (f for f in FORMATS if reports.get(f)):
        rs, cap = reports[fmt], budgets[fmt]
        rate = sum(r.hit_cap for r in rs) / len(rs)
        lines.append(f"{describe(f'{fmt} report content (cap {cap})', [r.content_tokens for r in rs])}"
                     f"   cap cut {rate:.0%}")
        if fmt in WRITTEN:
            lines.append(describe(f"  {fmt} writer visible output",
                                  [sum(c.output_tokens - (c.reasoning_tokens or 0) for c in r.calls) for r in rs]))
            p90 = percentile([r.content_tokens for r in rs], 0.9)
            if rate > 0.25:
                notes.append(f"{fmt}: the cap cut {rate:.0%} of reports; a higher cap would let writers finish")
            elif p90 < cap / 2:
                notes.append(f"{fmt}: 90% of reports use under half the cap ({p90} of {cap}); it rarely binds")
    for (fmt, n), totals in sorted(returns.items(), key=lambda kv: (kv[0][1], FORMATS.index(kv[0][0]))):
        lines.append(describe(f"total return per question {fmt} N={n}", totals))

    suggested = {fmt: round_up(percentile(values, 0.9)) for fmt, values in (("raw", seen), ("quotes", flagged)) if any(values)}
    if suggested:
        lines.append("\nSuggested caps, the 90th percentile of the uncapped size: "
                     + ", ".join(f"{fmt} {cap} (now {budgets[fmt]})" for fmt, cap in suggested.items()))
    lines += [f"Note: {note}" for note in notes]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*")
    parser.add_argument("--released", default="data/released_runs")
    parser.add_argument("--tokenizer", default="tiktoken:o200k_base", help="for the released runs")
    args = parser.parse_args(argv)
    released = sorted(Path(args.released).glob("*.jsonl"))
    if not released:
        print(f"no released runs in {args.released}; fetch them with: python -m harness.data --released\n")
    for path in released:
        print(released_report(path, args.tokenizer) + "\n")
    for run_dir in args.run_dirs:
        print(run_report(Path(run_dir)) + "\n")


if __name__ == "__main__":
    main()
