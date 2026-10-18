"""What the harness writes to disk. Every model call becomes a CallRecord, nested where it happened."""

from pydantic import BaseModel


class CallRecord(BaseModel):
    role: str
    phase: str
    round: int = 1
    subagent: int | None = None
    step: int | None = None
    endpoint: str
    model: str
    messages_sent: int  # how many messages of the owning conversation went into this call
    input_tokens: int
    output_tokens: int  # includes reasoning tokens
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    usage_source: dict[str, str]  # "reported", "estimated" or "not_reported" for cached and reasoning
    finish_reason: str | None
    t_start: float
    t_end: float
    response_text: str
    reasoning_text: str


class Trajectory(BaseModel):
    """One subagent's search: its conversation, what it saw, and what it flagged."""

    subagent: int
    round: int
    subtask: str
    messages: list[dict]
    actions: list[dict] = []
    retrieved: list[str] = []
    opened: list[str] = []
    seen: dict[str, str] = {}
    flagged: list[str] = []
    dropped_flags: int = 0
    status: str = ""
    steps: int = 0
    invalid_steps: int = 0
    answer: dict | None = None  # single-agent mode only
    calls: list[CallRecord] = []


class Report(BaseModel):
    """What one subagent sends back, exactly as the orchestrator reads it."""

    format: str
    text: str
    tokens: int  # return size: the wrapped text
    content_tokens: int  # what the cap applies to
    hit_cap: bool = False
    valid: bool = True
    trimmed: bool = False
    delivered_docids: list[str] = []
    write_messages: list[dict] = []  # the writing turn, continuing the subagent's conversation
    calls: list[CallRecord] = []


class Plan(BaseModel):
    requested_n: int
    actual_n: int
    subtasks: list[dict]  # {"task": str, "clues": [clue numbers]}
    repaired: bool = False
    fixed: bool = False
    messages: list[dict] = []
    calls: list[CallRecord] = []


class Stage1(BaseModel):
    key: str
    qid: str
    n: int
    seed: int
    plan: Plan
    trajectories: list[Trajectory]


class Round(BaseModel):
    round: int
    trajectories: list[Trajectory]
    reports: list[Report] = []
    decision: dict | None = None


class QuestionResult(BaseModel):
    qid: str
    mode: str
    format: str | None
    n: int | None
    seed: int
    stage1_key: str | None = None
    prompt_version: str
    git_commit: str
    status: str = "ok"
    plan: Plan | None = None
    rounds: list[Round] = []
    orchestrator_messages: list[dict] = []
    orchestrator_calls: list[CallRecord] = []
    answer: dict = {}
    tokens: dict = {}

    def trajectories(self) -> list[Trajectory]:
        return [t for rnd in self.rounds for t in rnd.trajectories]

    def all_calls(self) -> list[CallRecord]:
        calls = list(self.plan.calls) if self.plan else []
        for rnd in self.rounds:
            for trajectory in rnd.trajectories:
                calls += trajectory.calls
            for report in rnd.reports:
                calls += report.calls
        return calls + self.orchestrator_calls
