"""Run and endpoint configuration, loaded from TOML and validated."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from . import prompts

Format = Literal["short", "medium", "long", "claims", "quotes", "raw"]
FORMATS: tuple[str, ...] = get_args(Format)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Endpoint(Strict):
    """One OpenAI-compatible model server: a hosted API, or vLLM/SGLang on our own GPU."""

    base_url: str
    model: str
    tokenizer: str  # "hf:<repo>", "tiktoken:<encoding>" or "words"
    api_key_env: str = ""
    context_window: int = 131072
    max_concurrency: int = 16
    timeout_s: float = 600
    max_retries: int = 5
    max_tokens_param: str = "max_tokens"
    sampling: dict = {}
    thinking_on: dict = {}
    thinking_off: dict = {}


class Roles(Strict):
    orchestrator: str = ""
    subagent: str = ""
    single: str = ""


class Budgets(Strict):
    """Return caps in tokens of the orchestrator's tokenizer."""

    short: int = 200
    medium: int = 1000
    long: int = 3000
    claims: int = 800
    quotes: int = 2000
    raw: int = 8000


class Limits(Strict):
    max_steps: int = 15
    max_invalid_streak: int = 3
    max_extra_rounds: int = 1
    search_k: int = 5
    snippet_tokens: int = 512
    open_tokens: int = 1500
    call_max_tokens: int = 8192  # output limit for search, plan and decide calls
    writing_reasoning_allowance: int = 1024


class RunConfig(Strict):
    name: str
    endpoints: dict[str, Endpoint]
    roles: Roles
    questions: str
    corpus: str
    clues: str | None = None
    index: str | None = None  # Pyserini index directory; without it, BM25 runs in memory
    limit: int | None = None
    mode: Literal["multi", "single"] = "multi"
    n_mode: Literal["fixed", "free"] = "fixed"
    formats: list[Format] = list(FORMATS)
    n_values: list[int] = [1, 3, 6]
    seeds: list[int] = [0]
    max_questions_in_flight: int = 8
    budgets: Budgets = Budgets()
    limits: Limits = Limits()

    @model_validator(mode="after")
    def _roles_name_endpoints(self):
        needed = ("single",) if self.mode == "single" else ("orchestrator", "subagent")
        for role in needed:
            if getattr(self.roles, role) not in self.endpoints:
                raise ValueError(f"roles.{role} must name one of the endpoints {sorted(self.endpoints)}")
        return self


def load_config(path: str | Path) -> RunConfig:
    """Load a run config; `endpoints` may be inline tables or the path of an endpoints file."""
    data = tomllib.loads(Path(path).read_text())
    if isinstance(data.get("endpoints"), str):
        data["endpoints"] = tomllib.loads(Path(data["endpoints"]).read_text())["endpoints"]
    return RunConfig.model_validate(data)


def load_endpoints(path: str | Path) -> dict[str, Endpoint]:
    """The [endpoints.<name>] tables of an endpoints file."""
    tables = tomllib.loads(Path(path).read_text())["endpoints"]
    return {name: Endpoint.model_validate(table) for name, table in tables.items()}


_STAGE1_ENDPOINT_FIELDS = {
    "base_url", "model", "tokenizer", "context_window", "max_tokens_param", "sampling", "thinking_on", "thinking_off",
}
_STAGE1_LIMITS = {"max_steps", "max_invalid_streak", "search_k", "snippet_tokens", "open_tokens", "call_max_tokens"}


def stage1_key(cfg: RunConfig, question: dict, n: int, seed: int) -> str:
    """Hash of everything that can change the plan or the first-round searches."""
    parts = {
        "question": [question["qid"], question["question"], question.get("clues", [])],
        "n": n,
        "seed": seed,
        "n_mode": cfg.n_mode,
        "corpus": cfg.corpus,
        "index": cfg.index,
        "endpoints": {
            role: cfg.endpoints[getattr(cfg.roles, role)].model_dump(include=_STAGE1_ENDPOINT_FIELDS)
            for role in ("orchestrator", "subagent")
        },
        "limits": cfg.limits.model_dump(include=_STAGE1_LIMITS),
        "prompts": prompts.VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
