"""Grading (task D1): judge every answer against the correct one with the official BrowseComp-Plus grader prompt.

    python -m harness.grade judge runs/pilot configs/endpoints.toml qwen3-32b-judge
    python -m harness.grade review runs/pilot [--sample 30]
    python -m harness.grade export runs/pilot

judge   writes one grade per result to runs/<run>/grades/, mirroring results/, and skips results already
        graded. Any OpenAI-compatible endpoint can judge. The official judge is Qwen3-32B with thinking off
        (temperature 0.7, top_p 0.8, top_k 20); their evaluate_with_openai.py judges with GPT-4.1 instead.
review  writes a random sample of judgments to grades/review.md for the hand check.
export  writes the results in the BrowseComp-Plus run format, one folder per configuration, with the ground
        truth and evidence qrels beside them, so their scripts_evaluation/evaluate_run.py can grade them too.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from .config import RunConfig, load_endpoints
from .data import read_jsonl, write_jsonl
from .llm import ModelClient
from .pipeline import results, write_atomic

# Verbatim from BrowseComp-Plus scripts_evaluation/evaluate_run.py, itself from BrowseComp.
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()
TAG = {"role": "judge", "phase": "judge"}


def _field(name: str, value: str, text: str):
    """The official parser's lookup: **name:** value, then **name**: value, then name: value."""
    for label in (rf"\*\*{name}:\*\*", rf"\*\*{name}\*\*:", rf"{name}:"):
        match = re.search(rf"{label}\s*{value}", text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_judgment(text: str) -> dict:
    correct = _field("correct", r"(yes|no)", text or "")
    confidence = _field("confidence", r"(\d+(?:\.\d+)?)\s*%?", text or "")
    answer = _field("extracted_final_answer", r"(.*?)(?=\n|$)", text or "")
    return {
        "correct": None if correct is None else correct.lower() == "yes",
        "extracted_final_answer": answer.strip() if answer is not None else None,
        "confidence": None if confidence is None else min(float(confidence), 100.0),
        "parse_error": correct is None,
    }


def grade_path(run_dir: Path, rel: str) -> Path:
    return Path(run_dir) / "grades" / (rel.removesuffix(".json.gz") + ".json")


def load_grades(run_dir) -> dict[str, dict]:
    """Grades by result path (relative to results/)."""
    root = Path(run_dir) / "grades"
    return {g["result"]: g for g in (json.loads(p.read_text()) for p in sorted(root.rglob("*.json")))} \
        if root.exists() else {}


def ground_truth(run_dir, labels_path) -> dict[str, dict]:
    """qid -> {"question", "answer"}, from the run's questions file and the labels."""
    cfg = RunConfig.model_validate_json((Path(run_dir) / "config.json").read_text())
    labels = {r["qid"]: r for r in read_jsonl(labels_path)}
    return {q["qid"]: {"question": q["question"], "answer": labels[q["qid"]]["answer"]}
            for q in read_jsonl(cfg.questions) if q["qid"] in labels}


async def judge(client: ModelClient, endpoint: str, run_dir, labels_path) -> dict:
    """Grade every result without a usable grade. Returns counts of graded, skipped and failed results."""
    run_dir = Path(run_dir)
    truth = ground_truth(run_dir, labels_path)
    done = {rel for rel, g in load_grades(run_dir).items() if not g["parse_error"]}
    counts = {"graded": 0, "already_graded": 0, "failed": 0}
    ep = client.endpoints[endpoint]

    async def one(rel, result):
        if rel in done:
            counts["already_graded"] += 1
            return
        gt = truth[result.qid]
        grade = {"result": rel, "qid": result.qid, "mode": result.mode, "format": result.format, "n": result.n,
                 "seed": result.seed, "correct_answer": gt["answer"], "response": result.answer.get("response", ""),
                 "judge_endpoint": endpoint, "judge_model": ep.model}
        if result.answer.get("decision") != "answer":
            # As in the official evaluation: an incomplete response is wrong and is not sent to the judge.
            grade.update(correct=False, extracted_final_answer=None, confidence=None, parse_error=False,
                         judged=False, judge_response=None, call=None)
        else:
            prompt = GRADER_TEMPLATE.format(question=gt["question"], response=grade["response"],
                                            correct_answer=gt["answer"])
            try:
                reply = await client.chat(endpoint, [{"role": "user", "content": prompt}], TAG,
                                          thinking=False, max_tokens=4096)
            except Exception as error:  # left ungraded; the next run retries it
                counts["failed"] += 1
                print(f"{rel}: {type(error).__name__}: {error}", file=sys.stderr)
                return
            grade.update(parse_judgment(reply.text), judged=True, judge_response=reply.text,
                         call=reply.record.model_dump())
            grade["correct"] = bool(grade["correct"])
        write_atomic(grade_path(run_dir, rel), json.dumps(grade, ensure_ascii=False, indent=1).encode())
        counts["graded"] += 1

    await asyncio.gather(*(one(rel, result) for rel, result in results(run_dir)))
    return counts


def review(run_dir, sample: int = 30, seed: int = 0) -> str:
    """A random sample of judgments, with a box to tick for each during the hand check."""
    judged = [g for g in load_grades(run_dir).values() if g["judged"]]
    picked = random.Random(seed).sample(judged, min(sample, len(judged)))
    parts = [f"# Judge hand check: {len(picked)} random judgments\n",
             "Tick the box when the verdict is right; note the question when it is not.\n"]
    for g in picked:
        verdict = "correct" if g["correct"] else "incorrect"
        if g["parse_error"]:
            verdict += " (the judge's reply could not be parsed)"
        parts.append(f"## {g['result']}\n\n**Correct answer:** {g['correct_answer']}\n\n"
                     f"**Response:**\n\n{g['response']}\n\n**Judge extracted:** {g['extracted_final_answer']}  \n"
                     f"**Verdict:** {verdict}\n\n- [ ] the verdict is right\n")
    return "\n".join(parts)


def export(run_dir, labels_path, out_dir=None) -> Path:
    """The results in the BrowseComp-Plus run format, for their evaluate_run.py."""
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "browsecomp"
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    for rel, result in results(run_dir):
        name = f"single-s{result.seed}" if result.mode == "single" else f"n{result.n}-s{result.seed}-{result.format}"
        actions = [a["action"] for t in result.trajectories() for a in t.actions]
        role = cfg.roles.single if result.mode == "single" else cfg.roles.orchestrator
        record = {
            "query_id": result.qid,
            "tool_call_counts": {"search": actions.count("search"), "open": actions.count("open")},
            "status": "completed" if result.answer.get("decision") == "answer" else "failed",
            "retrieved_docids": sorted({d for t in result.trajectories() for d in t.retrieved}),
            "result": [{"type": "output_text", "output": result.answer.get("response", "")}],
            "metadata": {"model": cfg.endpoints[role].model, "harness_result": rel},
        }
        write_atomic(out / name / f"{result.qid}.json", json.dumps(record, ensure_ascii=False, indent=1).encode())
    truth = ground_truth(run_dir, labels_path)
    write_jsonl(out / "ground_truth.jsonl",
                ({"query_id": qid, "query": t["question"], "answer": t["answer"]} for qid, t in truth.items()))
    qrels = "".join(f"{r['qid']} 0 {d} 1\n" for r in read_jsonl(labels_path) for d in r["evidence_docids"])
    write_atomic(out / "qrel_evidence.txt", qrels.encode())
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("judge", help="grade every ungraded result")
    p.add_argument("run_dir")
    p.add_argument("endpoints", help="TOML file with [endpoints.<name>] tables")
    p.add_argument("endpoint", help="the judge endpoint")
    p.add_argument("--labels", default="data/labels.jsonl")
    p = sub.add_parser("review", help="write a hand-check sample of judgments")
    p.add_argument("run_dir")
    p.add_argument("--sample", type=int, default=30)
    p = sub.add_parser("export", help="write the results in the BrowseComp-Plus run format")
    p.add_argument("run_dir")
    p.add_argument("--labels", default="data/labels.jsonl")
    p.add_argument("--out", help="default: <run_dir>/browsecomp")
    args = parser.parse_args(argv)
    if args.command == "judge":
        client = ModelClient(load_endpoints(args.endpoints))
        print(json.dumps(asyncio.run(judge(client, args.endpoint, args.run_dir, args.labels)), indent=2))
    elif args.command == "review":
        path = Path(args.run_dir) / "grades" / "review.md"
        write_atomic(path, review(args.run_dir, args.sample).encode())
        print(f"wrote {path}")
    else:
        out = export(args.run_dir, args.labels, args.out)
        print(f"wrote {out}; grade one configuration with BrowseComp-Plus's\n  python scripts_evaluation/evaluate_run.py "
              f"--input_dir {out}/<config> --ground_truth {out}/ground_truth.jsonl --qrel_evidence {out}/qrel_evidence.txt")


if __name__ == "__main__":
    main()
