"""One question end to end: stage 1 (plan and first-round searches, cached) and stage 2 once per format."""

import asyncio
import gzip
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

from . import prompts
from .accounting import summarize
from .config import RunConfig, stage1_key
from .formats import make_report
from .orchestrator import FAILED, decide, fit_round, render_answer
from .planner import make_plan
from .prompts import render
from .records import QuestionResult, Round, Stage1
from .search import SearchSession
from .subagent import run_search


def git_commit() -> str:
    out = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                         cwd=Path(__file__).parent)
    return out.stdout.strip() or "unknown"


def read_gz(path) -> str:
    return gzip.decompress(Path(path).read_bytes()).decode()


def results(run_dir):
    """(path relative to results/, result) for every result in a run, in path order."""
    root = Path(run_dir) / "results"
    for path in sorted(root.rglob("*.json.gz")):
        yield path.relative_to(root).as_posix(), QuestionResult.model_validate_json(read_gz(path))


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary name, then rename, so a killed run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


async def gather_all(coros) -> list:
    """Run coroutines together; if one fails, cancel the rest rather than letting them spend tokens."""
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(c) for c in coros]
    return [t.result() for t in tasks]


def _numbered(tasks: list[str]) -> str:
    return "\n".join(f"{i}. {task}" for i, task in enumerate(tasks, 1))


class Harness:
    def __init__(self, cfg: RunConfig, client, corpus, index, run_dir):
        self.cfg, self.client, self.corpus, self.index = cfg, client, corpus, index
        self.limits = cfg.limits
        self.run_dir = Path(run_dir)
        self.commit = git_commit()

    def _session(self, endpoint: str) -> SearchSession:
        limits = self.limits
        return SearchSession(self.corpus, self.index, k=limits.search_k, snippet_tokens=limits.snippet_tokens,
                             open_tokens=limits.open_tokens, tokenizer=self.cfg.endpoints[endpoint].tokenizer)

    async def _search_round(self, tasks: list[str], round_no: int, seed: int) -> list:
        sub = self.cfg.roles.subagent
        return await gather_all(
            run_search(self.client, sub, self._session(sub), task=task, limits=self.limits, seed=seed,
                       tag={"role": "subagent", "round": round_no, "subagent": i})
            for i, task in enumerate(tasks, 1)
        )

    async def stage1(self, q: dict, n: int, seed: int) -> Stage1:
        """The plan and first-round searches, shared by every format; loaded from disk when cached."""
        key = stage1_key(self.cfg, q, n, seed)
        path = self.run_dir / "stage1" / f"{q['qid']}-{key}.json.gz"
        if path.exists():
            return Stage1.model_validate_json(read_gz(path))
        plan = await make_plan(self.client, self.cfg.roles.orchestrator, q["question"], q.get("clues", []), n,
                               n_mode=self.cfg.n_mode, limits=self.limits, seed=seed)
        trajectories = await self._search_round([s["task"] for s in plan.subtasks], 1, seed)
        stage1 = Stage1(key=key, qid=q["qid"], n=n, seed=seed, plan=plan, trajectories=trajectories)
        write_atomic(path, gzip.compress(stage1.model_dump_json().encode()))
        return stage1

    async def stage2(self, q: dict, fmt: str, stage1: Stage1) -> QuestionResult:
        """Reports in one format, the orchestrator's decision, and any extra rounds."""
        cfg, limits = self.cfg, self.limits
        orch, sub = cfg.roles.orchestrator, cfg.roles.subagent
        tokenizer, context_window = cfg.endpoints[orch].tokenizer, cfg.endpoints[orch].context_window
        budget, n, seed = getattr(cfg.budgets, fmt), stage1.n, stage1.seed
        messages = [{"role": "system", "content": render("orchestrator_system")}]
        rounds, decide_calls = [], []
        tasks = [s["task"] for s in stage1.plan.subtasks]
        trajectories = stage1.trajectories
        last_round = limits.max_extra_rounds + 1
        for round_no in range(1, last_round + 1):
            if round_no > 1:
                trajectories = await self._search_round(tasks, round_no, seed)
            reports = await gather_all(
                make_report(fmt, t, budget, tokenizer, client=self.client, endpoint=sub, limits=limits, seed=seed)
                for t in trajectories
            )
            final = round_no == last_round
            if round_no == 1:
                header = render("decide_first", question=q["question"], subtasks=_numbered(tasks))
            else:
                header = render("decide_later", round=round_no)
            instruction = render("decide_final") if final else render("decide", n=n)
            messages.append({"role": "user", "content": fit_round(messages, header, reports, instruction, tokenizer,
                                                                  context_window, limits.call_max_tokens)})
            decision, calls = await decide(self.client, orch, messages, n=n, final=final, round_no=round_no,
                                           limits=limits, seed=seed)
            decide_calls += calls
            rounds.append(Round(round=round_no, trajectories=trajectories, reports=reports, decision=decision))
            if decision["decision"] != "redelegate":
                break
            tasks = [s["task"] for s in decision["subtasks"]]
        result = QuestionResult(
            qid=q["qid"], mode="multi", format=fmt, n=n, seed=seed, stage1_key=stage1.key,
            prompt_version=prompts.VERSION, git_commit=self.commit, plan=stage1.plan, rounds=rounds,
            orchestrator_messages=messages, orchestrator_calls=decide_calls,
            answer={**decision, "response": render_answer(decision)},
        )
        result.tokens = summarize(result.all_calls(), [r.tokens for r in rounds[0].reports])
        return result

    async def single(self, q: dict, seed: int) -> QuestionResult:
        """The single-agent baseline: one agent searches the full question and answers."""
        endpoint = self.cfg.roles.single
        trajectory = await run_search(self.client, endpoint, self._session(endpoint), task=q["question"],
                                      limits=self.limits, seed=seed, tag={"role": "single", "round": 1, "subagent": 1},
                                      single=True)
        answer = {"decision": "answer", **trajectory.answer} if trajectory.answer else dict(FAILED)
        result = QuestionResult(
            qid=q["qid"], mode="single", format=None, n=None, seed=seed, prompt_version=prompts.VERSION,
            git_commit=self.commit, rounds=[Round(round=1, trajectories=[trajectory])],
            answer={**answer, "response": render_answer(answer)},
        )
        result.tokens = summarize(result.all_calls(), [])
        return result

    def result_path(self, qid: str, fmt: str | None, n: int | None, seed: int) -> Path:
        if fmt is None:
            return self.run_dir / "results" / "single" / f"s{seed}" / f"{qid}.json.gz"
        return self.run_dir / "results" / f"n{n}" / f"s{seed}" / fmt / f"{qid}.json.gz"

    async def run_unit(self, q: dict, n: int | None, seed: int, retry_failed: bool = False) -> None:
        """Every unfinished format of one question, N and seed (single-agent mode when n is None)."""
        formats = [None] if n is None else list(self.cfg.formats)
        random.Random(f"{q['qid']}/{n}/{seed}").shuffle(formats)
        todo = []
        for fmt in formats:
            path = self.result_path(q["qid"], fmt, n, seed)
            if not path.exists() and (retry_failed or not _error_path(path).exists()):
                todo.append(fmt)
        if not todo:
            return
        try:
            stage1 = None if n is None else await self.stage1(q, n, seed)
        except Exception:
            for fmt in todo:
                self._fail(q["qid"], fmt, n, seed)
            return
        for fmt in todo:
            path = self.result_path(q["qid"], fmt, n, seed)
            try:
                result = await (self.single(q, seed) if n is None else self.stage2(q, fmt, stage1))
            except Exception:
                self._fail(q["qid"], fmt, n, seed)
                continue
            write_atomic(path, gzip.compress(result.model_dump_json().encode()))
            _error_path(path).unlink(missing_ok=True)
            print(f"{q['qid']} n={n} seed={seed} {fmt}: {result.answer['decision']}", file=sys.stderr)

    def _fail(self, qid: str, fmt: str | None, n: int | None, seed: int) -> None:
        error = traceback.format_exc()
        path = _error_path(self.result_path(qid, fmt, n, seed))
        write_atomic(path, json.dumps({"qid": qid, "format": fmt, "n": n, "seed": seed, "error": error}).encode())
        print(f"{qid} n={n} seed={seed} {fmt}: FAILED\n{error}", file=sys.stderr)


def _error_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.name.removesuffix(".json.gz") + ".error.json")
