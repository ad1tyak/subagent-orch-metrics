"""Run a config's grid of questions × N × seeds × formats, skipping results that already exist.

    python -m harness.run configs/pilot.toml [--limit 5] [--shard 0/4] [--retry-failed]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import RunConfig, load_config
from .llm import ModelClient
from .pipeline import Harness
from .search import Corpus, MemoryIndex, PyseriniIndex

# Settings that can change between restarts of one run without changing its results.
_OPERATIONAL = {
    "limit": True,
    "max_questions_in_flight": True,
    "endpoints": {"__all__": {"api_key_env", "max_concurrency", "timeout_s", "max_retries"}},
}


def load_questions(cfg: RunConfig) -> list[dict]:
    """JSONL lines of {"qid", "question", "clues"}. Clues may instead come from a {qid: [clue, ...]} JSON file."""
    clues = json.loads(Path(cfg.clues).read_text()) if cfg.clues else {}
    questions = []
    for line in Path(cfg.questions).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            q["qid"] = str(q["qid"])
            q["clues"] = q.get("clues") or clues.get(q["qid"], [])
            questions.append(q)
    return questions


async def run(cfg: RunConfig, questions: list[dict], run_dir: Path, retry_failed: bool = False,
              http_client=None) -> None:
    corpus = Corpus.load(cfg.corpus)
    index = PyseriniIndex(cfg.index) if cfg.index else MemoryIndex(corpus)
    harness = Harness(cfg, ModelClient(cfg.endpoints, http_client), corpus, index, run_dir)
    slots = asyncio.Semaphore(cfg.max_questions_in_flight)
    n_values = [None] if cfg.mode == "single" else cfg.n_values

    async def unit(q, n, seed):
        async with slots:
            await harness.run_unit(q, n, seed, retry_failed)

    await asyncio.gather(*(unit(q, n, seed) for q in questions for n in n_values for seed in cfg.seeds))


def save_config(cfg: RunConfig, run_dir: Path) -> None:
    """Keep the config with the run, and refuse to mix results from a different config into it."""
    saved, current = run_dir / "config.json", cfg.model_dump_json(indent=2, exclude=_OPERATIONAL)
    if saved.exists() and saved.read_text() != current:
        sys.exit(f"{saved} was written by a different config; give this run a new name")
    run_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(current)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--shard", help="i/k: run every k-th question, starting at question i")
    parser.add_argument("--retry-failed", action="store_true", help="rerun questions that failed before")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for path in (cfg.questions, cfg.corpus):
        if not Path(path).exists():
            sys.exit(f"{path} not found; download BrowseComp-Plus with: python -m harness.data")
    if cfg.clues and not Path(cfg.clues).exists():
        sys.exit(f"{cfg.clues} not found; extract clues with: python -m harness.clues <endpoints file> <endpoint>. "
                 "Or remove `clues` from the config to plan from the question alone.")
    questions = load_questions(cfg)[: args.limit or cfg.limit]
    if args.shard:
        i, k = map(int, args.shard.split("/"))
        questions = questions[i::k]
    run_dir = Path(args.runs_dir) / cfg.name
    save_config(cfg, run_dir)
    asyncio.run(run(cfg, questions, run_dir, args.retry_failed))


if __name__ == "__main__":
    main()
