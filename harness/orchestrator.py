"""The orchestrator's decision after reading the reports: answer, or send subagents out again."""

from . import tokens
from .parsing import confidence, first_json
from .prompts import render

FAILED = {"decision": "failed", "explanation": "", "exact_answer": "", "confidence": None}


def parse_decision(text: str, n: int, final: bool) -> dict:
    """A decision from the orchestrator's reply. Raises ValueError when the reply is unusable."""
    decision = first_json(text)
    kind = decision.get("decision")
    if kind == "answer":
        answer = decision.get("exact_answer")
        if not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
            raise ValueError("an answer needs exact_answer")
        return {
            "decision": "answer",
            "explanation": str(decision.get("explanation", "")),
            "exact_answer": str(answer).strip(),
            "confidence": confidence(decision.get("confidence")),
        }
    if kind == "redelegate" and not final:
        items = decision.get("subtasks") if isinstance(decision.get("subtasks"), list) else []
        tasks = [item.get("task") if isinstance(item, dict) else item for item in items]
        tasks = [task.strip() for task in tasks if isinstance(task, str) and task.strip()]
        if not tasks:
            raise ValueError("redelegate needs at least one subtask with a task")
        return {"decision": "redelegate", "subtasks": [{"task": task} for task in tasks[:n]]}
    raise ValueError('decision must be "answer"' if final else 'decision must be "answer" or "redelegate"')


def render_answer(decision: dict) -> str:
    """The answer in BrowseComp grading style. Without a confidence the grader assumes 100%."""
    lines = [f"Explanation: {decision.get('explanation', '')}", f"Exact Answer: {decision.get('exact_answer', '')}"]
    if decision.get("confidence") is not None:
        lines.append(f"Confidence: {decision['confidence']:g}%")
    return "\n".join(lines)


def round_message(header: str, reports: list, instruction: str, tokenizer: str, room: int | None = None) -> str:
    """One round's user message. With room set, every report is cut to an equal share of it."""
    texts = [report.text for report in reports]
    if room is not None:
        share = max(room, 0) // max(len(texts), 1)
        for report in reports:
            report.trimmed = tokens.count(tokenizer, report.text) > share
        texts = [tokens.truncate(tokenizer, text, share) for text in texts]
    return "\n\n".join([header, *texts, instruction])


def fit_round(messages: list[dict], header: str, reports: list, instruction: str, tokenizer: str,
              context_window: int, reserve: int) -> str:
    """The round message, with reports trimmed if the whole conversation would overflow the context window."""
    used = sum(tokens.count(tokenizer, m["content"]) for m in messages)
    message = round_message(header, reports, instruction, tokenizer)
    if used + tokens.count(tokenizer, message) + reserve <= context_window:
        return message
    room = context_window - reserve - used - tokens.count(tokenizer, round_message(header, [], instruction, tokenizer))
    return round_message(header, reports, instruction, tokenizer, room)


async def decide(client, endpoint: str, messages: list[dict], *, n: int, final: bool, round_no: int, limits,
                 seed: int | None) -> tuple[dict, list]:
    """Ask for a decision, with one repair call if the first reply is unusable. Appends to messages."""
    calls = []
    for attempt in range(2):
        reply = await client.chat(endpoint, messages, {"role": "orchestrator", "phase": "decide", "round": round_no,
                                  "step": attempt}, thinking=True, max_tokens=limits.call_max_tokens, seed=seed)
        calls.append(reply.record)
        messages.append({"role": "assistant", "content": reply.text})
        try:
            return parse_decision(reply.text, n, final), calls
        except ValueError as error:
            if attempt == 0:
                messages.append({"role": "user", "content": render("decide_repair", problem=str(error))})
    return dict(FAILED), calls
