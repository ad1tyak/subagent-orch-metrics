"""Evidence recall (task D2) and the analysis of finished runs (task D3).

    python -m harness.analyze runs/pilot [runs/baseline ...] [--labels data/labels.jsonl] [--out DIR]

Writes to --out (default: <first run>/analysis):
  rows.csv     one row per result: answer, grade, rounds, tokens by phase, return sizes and evidence recall
  configs.csv  one row per configuration: accuracy with a 95% CI, mean tokens in every phase, tokens per
               correct answer, re-delegation rate, how often caps bind, and evidence recall
  paired.csv   every format against the medium summary at the same N, on the same questions and seeds:
               the accuracy difference (exact McNemar test, bootstrap CI) and the change in tokens
  accuracy_vs_tokens.png, tokens_by_phase.png   if matplotlib is installed (pip install -e '.[analysis]')
Accuracy needs grades from `python -m harness.grade judge`; until then those columns stay empty.
Evidence recall is the share of a question's labeled evidence documents that the subagents retrieved
(saw in search results), opened, or delivered to the orchestrator in their reports.
"""

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .accounting import BUCKETS
from .config import FORMATS
from .data import read_jsonl
from .grade import load_grades
from .pipeline import results

REFERENCE = "medium"
TARGET = {"accuracy_points": -2.0, "output_change_pct": -20.0}  # tasks.md: within 2 points, 20% fewer output tokens


def recall(found: set, relevant: set) -> float | None:
    return len(found & relevant) / len(relevant) if relevant else None


def result_row(run: str, rel: str, result, label: dict, grade: dict | None) -> dict:
    evidence, gold = set(label.get("evidence_docids", [])), set(label.get("gold_docids", []))
    trajectories = result.trajectories()
    reports = [r for rnd in result.rounds for r in rnd.reports]
    first = result.rounds[0].reports if result.rounds else []
    delivered = {d for r in reports for d in r.delivered_docids}
    multi = result.mode == "multi"
    row = {
        "run": run, "result": rel, "qid": result.qid, "mode": result.mode,
        "format": result.format if multi else "single", "n": result.n, "seed": result.seed,
        "decision": result.answer.get("decision"), "exact_answer": result.answer.get("exact_answer"),
        "confidence": result.answer.get("confidence"), "correct": grade["correct"] if grade else None,
        "rounds": len(result.rounds), "redelegated": len(result.rounds) > 1,
    }
    for b in (*BUCKETS, "total"):
        for kind in ("input", "output", "cached", "reasoning"):
            row[f"{b}_{kind}"] = result.tokens[b][kind]
    row["total_calls"] = result.tokens["total"]["calls"]
    sizes = result.tokens.get("return_size", [])
    row.update(
        subagents=len(sizes), return_tokens=sum(sizes),
        return_per_subagent=statistics.fmean(sizes) if sizes else None,
        hit_cap=sum(r.hit_cap for r in first), trimmed=sum(r.trimmed for r in reports),
        invalid_reports=sum(not r.valid for r in reports),
        search_steps=sum(t.steps for t in trajectories), invalid_steps=sum(t.invalid_steps for t in trajectories),
        evidence_docs=len(evidence),
        recall_retrieved=recall({d for t in trajectories for d in t.retrieved}, evidence),
        recall_opened=recall({d for t in trajectories for d in t.opened}, evidence),
        recall_delivered=recall(delivered, evidence) if multi else None,
        gold_recall_delivered=recall(delivered, gold) if multi else None,
    )
    return row


def load_rows(run_dirs, labels_path) -> list[dict]:
    labels = {r["qid"]: r for r in read_jsonl(labels_path)}
    rows = []
    for run_dir in map(Path, run_dirs):
        grades = load_grades(run_dir)
        rows += [result_row(run_dir.name, rel, result, labels.get(result.qid, {}), grades.get(rel))
                 for rel, result in results(run_dir)]
    return rows


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def bootstrap(items: list, stat, rounds: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    """95% percentile interval of stat over resamples of items (one item per question)."""
    if len(items) < 2:
        return None, None
    rng = random.Random(seed)
    values = sorted(stat([rng.choice(items) for _ in items]) for _ in range(rounds))
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def per_question(pairs) -> list:
    """Group (qid, value) pairs by question, so seeds of one question resample together."""
    groups = defaultdict(list)
    for qid, value in pairs:
        groups[qid].append(value)
    return list(groups.values())


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def config_key(row: dict):
    order = FORMATS.index(row["format"]) if row["format"] in FORMATS else -1
    return (row["mode"] == "multi", row["n"] or 0, order)


def group_configs(rows) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in sorted(rows, key=config_key):
        groups[(row["mode"], row["format"], row["n"])].append(row)
    return groups


def summarize_configs(rows) -> list[dict]:
    out = []
    for (mode, fmt, n), rs in group_configs(rows).items():
        graded = [r for r in rs if r["correct"] is not None]
        correct = sum(r["correct"] for r in graded)
        low, high = bootstrap(per_question((r["qid"], r["correct"]) for r in graded),
                              lambda sample: 100 * statistics.fmean(v for q in sample for v in q))
        summary = {
            "mode": mode, "format": fmt, "n": n, "results": len(rs), "questions": len({r["qid"] for r in rs}),
            "graded": len(graded), "accuracy": 100 * correct / len(graded) if graded else None,
            "accuracy_ci_low": low, "accuracy_ci_high": high,
            "redelegation_rate": mean(r["redelegated"] for r in rs),
            "failed_decisions": sum(r["decision"] != "answer" for r in rs),
        }
        for b in (*BUCKETS, "total"):
            for kind in ("input", "output"):
                summary[f"{b}_{kind}"] = mean(r[f"{b}_{kind}"] for r in rs)
        summary.update(
            total_cached=mean(r["total_cached"] for r in rs), total_reasoning=mean(r["total_reasoning"] for r in rs),
            output_per_correct=sum(r["total_output"] for r in graded) / correct if correct else None,
            input_per_correct=sum(r["total_input"] for r in graded) / correct if correct else None,
            return_per_subagent=mean(r["return_per_subagent"] for r in rs),
            return_tokens=mean(r["return_tokens"] for r in rs),
            hit_cap_rate=sum(r["hit_cap"] for r in rs) / max(sum(r["subagents"] for r in rs), 1),
            recall_retrieved=mean(r["recall_retrieved"] for r in rs), recall_opened=mean(r["recall_opened"] for r in rs),
            recall_delivered=mean(r["recall_delivered"] for r in rs),
            gold_recall_delivered=mean(r["gold_recall_delivered"] for r in rs),
        )
        out.append(summary)
    return out


def _change(sample) -> float:
    """Percent change of a format's total from the reference's, over (format, reference) value pairs."""
    ours, ref = sum(a for q in sample for a, _ in q), sum(b for q in sample for _, b in q)
    return 100 * (ours / ref - 1) if ref else float("nan")


def paired(rows, reference: str = REFERENCE) -> list[dict]:
    """Each multi-agent configuration against the reference format at the same N, on matching questions and seeds."""
    index = {(r["format"], r["n"], r["qid"], r["seed"]): r for r in rows if r["mode"] == "multi"}
    out = []
    for (mode, fmt, n), rs in group_configs(rows).items():
        if mode != "multi" or fmt == reference:
            continue
        pairs = [(r, index[(reference, n, r["qid"], r["seed"])]) for r in rs
                 if (reference, n, r["qid"], r["seed"]) in index]
        if not pairs:
            continue
        graded = [(a, b) for a, b in pairs if a["correct"] is not None and b["correct"] is not None]
        wins = sum(a["correct"] and not b["correct"] for a, b in graded)
        losses = sum(b["correct"] and not a["correct"] for a, b in graded)
        diff = 100 * (wins - losses) / len(graded) if graded else None
        low, high = bootstrap(per_question((a["qid"], a["correct"] - b["correct"]) for a, b in graded),
                              lambda sample: 100 * statistics.fmean(v for q in sample for v in q))
        row = {"format": fmt, "n": n, "reference": reference, "pairs": len(pairs), "graded_pairs": len(graded),
               "accuracy": 100 * sum(a["correct"] for a, _ in graded) / len(graded) if graded else None,
               "reference_accuracy": 100 * sum(b["correct"] for _, b in graded) / len(graded) if graded else None,
               "accuracy_diff_points": diff, "accuracy_diff_ci_low": low, "accuracy_diff_ci_high": high,
               "wins": wins, "losses": losses, "mcnemar_p": mcnemar(wins, losses) if graded else None}
        for kind in ("output", "input"):
            items = per_question((a["qid"], (a[f"total_{kind}"], b[f"total_{kind}"])) for a, b in pairs)
            row[f"{kind}_change_pct"] = _change(items)
            row[f"{kind}_change_ci_low"], row[f"{kind}_change_ci_high"] = bootstrap(items, _change)
        row["meets_target"] = (diff >= TARGET["accuracy_points"]
                               and row["output_change_pct"] <= TARGET["output_change_pct"]) if graded else None
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else ("" if v is None else v)
                             for k, v in row.items()})


def label(config: dict) -> str:
    return "single agent" if config["mode"] == "single" else f"{config['format']} N={config['n']}"


def plot(configs: list[dict], out_dir: Path) -> list[Path]:
    """Accuracy against tokens, and tokens by phase. Returns the files written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib is not installed, so no plots (pip install -e '.[analysis]')")
        return []
    colors = {fmt: plt.cm.tab10(i) for i, fmt in enumerate((*FORMATS, "single"))}
    markers = {1: "o", 3: "s", 6: "^", None: "*"}
    written = []
    graded = [c for c in configs if c["accuracy"] is not None]
    if graded:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, kind in zip(axes, ("output", "input")):
            for c in graded:
                err = None
                if c["accuracy_ci_low"] is not None:
                    err = [[c["accuracy"] - c["accuracy_ci_low"]], [c["accuracy_ci_high"] - c["accuracy"]]]
                ax.errorbar(c[f"total_{kind}"], c["accuracy"], yerr=err, fmt=markers.get(c["n"], "o"),
                            color=colors[c["format"]], capsize=3)
            ax.set_xlabel(f"mean {kind} tokens per question")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("accuracy (%)")
        handles = [Line2D([], [], color=colors[f], marker="o", linestyle="", label=f)
                   for f in colors if any(c["format"] == f for c in graded)]
        handles += [Line2D([], [], color="gray", marker=m, linestyle="", label=f"N = {n}")
                    for n, m in markers.items() if n and any(c["n"] == n for c in graded)]
        axes[1].legend(handles=handles, loc="best", fontsize=8)
        fig.suptitle("Accuracy against tokens, one point per configuration (95% CI)")
        fig.tight_layout()
        written.append(out_dir / "accuracy_vs_tokens.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)
    if configs:
        fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.6 * len(configs) + 3), 8), sharex=True)
        names = [label(c) for c in configs]
        for ax, kind in zip(axes, ("output", "input")):
            bottom = [0.0] * len(configs)
            for b in BUCKETS:
                values = [c[f"{b}_{kind}"] or 0 for c in configs]
                ax.bar(names, values, bottom=bottom, label=b.replace("_", " "))
                bottom = [x + y for x, y in zip(bottom, values)]
            ax.set_ylabel(f"mean {kind} tokens")
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].tick_params(axis="x", labelrotation=60)
        fig.suptitle("Tokens per question by phase")
        fig.tight_layout()
        written.append(out_dir / "tokens_by_phase.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)
    return written


def table(configs: list[dict]) -> str:
    def cell(v, width):
        return f"{v:>{width}.1f}" if isinstance(v, float) else f"{'' if v is None else v:>{width}}"

    columns = [("accuracy", "accuracy %", 10), ("total_output", "output tok", 11), ("total_input", "input tok", 11),
               ("return_per_subagent", "return/sub", 11), ("redelegation_rate", "re-asked", 9),
               ("recall_delivered", "recall", 7)]
    lines = [f"{'configuration':<16}" + "".join(f" {header:>{width}}" for _, header, width in columns)]
    for c in configs:
        lines.append(f"{label(c):<16}" + "".join(" " + cell(c[key], width) for key, _, width in columns))
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels", default="data/labels.jsonl")
    parser.add_argument("--out", help="default: <first run>/analysis")
    args = parser.parse_args(argv)
    out = Path(args.out or Path(args.run_dirs[0]) / "analysis")
    rows = load_rows(args.run_dirs, args.labels)
    configs = summarize_configs(rows)
    write_csv(out / "rows.csv", rows)
    write_csv(out / "configs.csv", configs)
    write_csv(out / "paired.csv", paired(rows))
    figures = plot(configs, out)
    print(table(configs))
    print(f"\n{len(rows)} results, {sum(r['correct'] is not None for r in rows)} graded. Wrote rows.csv, "
          f"configs.csv, paired.csv{''.join(', ' + p.name for p in figures)} to {out}")


if __name__ == "__main__":
    main()
