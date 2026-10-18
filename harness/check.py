"""Invariant checks on a run's results. Exits with status 1 if any check fails.

    python -m harness.check runs/pilot
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from .accounting import BUCKETS, summarize
from .config import RunConfig
from .formats import WRITTEN
from .pipeline import results


def check(run_dir) -> tuple[list[str], dict]:
    run_dir = Path(run_dir)
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    problems, groups = [], defaultdict(list)
    stats = {"results": 0, "failed": 0, "failed_decisions": 0, "calls": 0,
             "cached_not_reported": 0, "reasoning_estimated": 0, "reasoning_not_reported": 0}
    for rel, result in results(run_dir):
        where = f"results/{rel}"
        calls = result.all_calls()
        stats["results"] += 1
        stats["calls"] += len(calls)
        stats["failed_decisions"] += result.answer.get("decision") == "failed"
        for gap, count in result.tokens.get("usage_gaps", {}).items():
            stats[gap] += count
        if result.status != "ok" or not result.answer.get("response"):
            problems.append(f"{where}: no answer")
        if summarize(calls, result.tokens.get("return_size", [])) != result.tokens:
            problems.append(f"{where}: token totals do not match the call records")
        for key in ("calls", "input", "output", "cached", "reasoning"):
            if sum(result.tokens[b][key] for b in BUCKETS) != result.tokens["total"][key]:
                problems.append(f"{where}: phase sums of {key} do not add up to the total")
        for rnd in result.rounds:
            for trajectory, report in zip(rnd.trajectories, rnd.reports):
                label = f"{where}: round {rnd.round} subagent {trajectory.subagent}"
                if (report.format in WRITTEN) != bool(report.calls):
                    problems.append(f"{label} made {len(report.calls)} writing calls in format {report.format}")
                if report.content_tokens > getattr(cfg.budgets, report.format):
                    problems.append(f"{label} report is over its {report.format} cap")
        if result.mode == "multi":
            groups[(result.qid, result.n, result.seed)].append(result)
    for (qid, n, seed), group in groups.items():
        if len({r.stage1_key for r in group}) > 1:
            problems.append(f"{qid} n={n} seed={seed}: formats used different stage-1 keys")
        searches = {json.dumps([t.model_dump() for t in r.rounds[0].trajectories], sort_keys=True) for r in group}
        if len(searches) > 1:
            problems.append(f"{qid} n={n} seed={seed}: formats did not share identical round-1 searches")
    for path in sorted((run_dir / "results").rglob("*.error.json")):
        stats["failed"] += 1
        problems.append(f"{path.relative_to(run_dir)}: failed; rerun with --retry-failed")
    return problems, stats


def main(argv=None) -> None:
    run_dir = (argv or sys.argv[1:])[0]
    problems, stats = check(run_dir)
    for problem in problems:
        print(problem)
    print(json.dumps(stats, indent=2))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
