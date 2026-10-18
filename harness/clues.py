"""Clue extraction (task A4): list the separate clues in each question, for the planner to group into subtasks.

    python -m harness.clues configs/endpoints.toml gptoss-120b-vllm [--limit 30]

Each reply is appended to data/clues.raw.jsonl as soon as it arrives, so a rerun only asks about questions
that are missing or came back invalid. At the end, data/clues.json is rebuilt from it, a random sample is
written to data/clues.review.md for the hand check, and a chart of clue counts shows which N values the
data supports.
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from string import Template

from .config import load_endpoints
from .data import read_jsonl
from .llm import ModelClient
from .parsing import first_json

# Not in prompts.toml: that file's hash versions the study's runs, and this prompt runs once, before them.
PROMPT = Template('''
Below is a hard research question. It describes something through several clues, and the answer must satisfy all of them.

List every separate clue in the question. A clue is one fact or condition stated in the question that can be checked or searched on its own, such as a date, a place, a relationship or an event. Rewrite each clue so it makes sense without the question: name who or what it is about (for example "the author" or "the bridge"), and keep the question's own names, numbers and dates. Do not add facts, guesses or answers, and do not list what the question finally asks for.

Question:
$question

Reply with only a JSON object:
{"clues": ["...", "..."]}
'''.strip())
REPAIR = Template('Invalid reply: $problem. Reply again with only the JSON object: {"clues": ["...", "..."]}')
TAG = {"role": "clues", "phase": "clues"}


def parse_clues(text: str) -> list[str]:
    clues = first_json(text).get("clues")
    if not isinstance(clues, list):
        raise ValueError('the reply needs a "clues" list')
    clues = [c.strip() for c in clues if isinstance(c, str) and c.strip()]
    if not clues:
        raise ValueError("no clues listed")
    return clues


async def extract(client: ModelClient, endpoint: str, q: dict) -> dict:
    """One question's clues, with one repair call if the first reply is unusable."""
    messages = [{"role": "user", "content": PROMPT.substitute(question=q["question"])}]
    calls = []
    for attempt in range(2):
        reply = await client.chat(endpoint, messages, {**TAG, "step": attempt}, thinking=True, max_tokens=4096)
        calls.append(reply.record.model_dump())
        try:
            return {"qid": q["qid"], "clues": parse_clues(reply.text), "valid": True, "calls": calls}
        except ValueError as error:
            messages += [{"role": "assistant", "content": reply.text},
                         {"role": "user", "content": REPAIR.substitute(problem=error)}]
    return {"qid": q["qid"], "clues": [], "valid": False, "calls": calls}


def load_raw(path: Path) -> dict[str, dict]:
    """Records by qid; a later record for the same question replaces an earlier one."""
    return {r["qid"]: r for r in read_jsonl(path)} if path.exists() else {}


async def run(client: ModelClient, endpoint: str, questions: list[dict], raw_path: Path) -> dict[str, dict]:
    records = load_raw(raw_path)
    todo = [q for q in questions if not records.get(q["qid"], {}).get("valid")]
    print(f"asking for clues on {len(todo)} of {len(questions)} questions", file=sys.stderr)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "a", encoding="utf-8") as raw:

        async def one(q):
            try:
                record = await extract(client, endpoint, q)
            except Exception as error:  # a failed question is retried on the next run
                print(f"{q['qid']}: {type(error).__name__}: {error}", file=sys.stderr)
                return
            records[q["qid"]] = record
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw.flush()

        await asyncio.gather(*(one(q) for q in todo))
    return records


def chart(counts: list[int], n_values=(1, 3, 6)) -> str:
    """Questions per clue count as a text bar chart, and how many questions each N can split."""
    if not counts:
        return "no questions have clues yet"
    by_count = Counter(counts)
    width = 50 / max(by_count.values())
    lines = [f"{k:>3} clues | {'#' * max(1, round(v * width))} {v}" for k, v in sorted(by_count.items())]
    lines.append("")
    lines += [f"N = {n}: {sum(c >= n for c in counts)} of {len(counts)} questions have at least {n} clues"
              for n in n_values]
    return "\n".join(lines)


def review(questions: list[dict], records: dict[str, dict], sample: int, seed: int = 0) -> str:
    """A random sample of questions and their clues, with boxes to tick during the hand check."""
    valid = [q for q in questions if records.get(q["qid"], {}).get("valid")]
    picked = random.Random(seed).sample(valid, min(sample, len(valid)))
    parts = [f"# Clue hand check: {len(picked)} random questions\n",
             "For each question, tick the boxes that hold; note any clue that is wrong, missing or not self-contained.\n"]
    for q in picked:
        clues = "\n".join(f"{i}. {c}" for i, c in enumerate(records[q["qid"]]["clues"], 1))
        parts.append(f"## Question {q['qid']}\n\n{q['question']}\n\n{clues}\n\n"
                     "- [ ] every clue is stated in the question, with nothing added\n"
                     "- [ ] every clue makes sense on its own\n"
                     "- [ ] no clue from the question is missing\n")
    return "\n".join(parts)


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_outputs(questions: list[dict], records: dict[str, dict], out: Path, sample: int) -> str:
    """Write the clue file and the hand-check sample; return the chart and a summary."""
    clues = {q["qid"]: records[q["qid"]]["clues"] for q in questions if records.get(q["qid"], {}).get("valid")}
    write_text(out, json.dumps(clues, ensure_ascii=False, indent=1))
    write_text(out.with_suffix(".review.md"), review(questions, records, sample))
    calls = [c for r in records.values() for c in r["calls"]]
    return (f"{chart([len(c) for c in clues.values()])}\n\n"
            f"{len(clues)} of {len(questions)} questions have clues in {out}; "
            f"{len(questions) - len(clues)} are missing or invalid (rerun to retry).\n"
            f"Tokens so far: {sum(c['input_tokens'] for c in calls)} input, "
            f"{sum(c['output_tokens'] for c in calls)} output. Hand check: {out.with_suffix('.review.md')}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoints", help="TOML file with [endpoints.<name>] tables")
    parser.add_argument("endpoint")
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--out", default="data/clues.json")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--review", type=int, default=30, help="questions in the hand-check sample")
    args = parser.parse_args(argv)
    out = Path(args.out)
    questions = read_jsonl(args.questions)[: args.limit]
    client = ModelClient(load_endpoints(args.endpoints))
    records = asyncio.run(run(client, args.endpoint, questions, out.with_suffix(".raw.jsonl")))
    print(write_outputs(questions, records, out, args.review))


if __name__ == "__main__":
    main()
