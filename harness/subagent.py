"""The subagent search loop: one JSON action per turn, run against a SearchSession."""

from . import tokens
from .parsing import confidence, first_json
from .prompts import render
from .records import Trajectory

STATUSES = ("found", "partial", "not_found")


def parse_action(text: str, single: bool = False) -> dict:
    """The action in a reply. Raises ValueError with a message the model can act on."""
    action = first_json(text)
    kind = action.get("action")
    if kind == "search":
        query = action.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search needs a non-empty query")
        return {"action": "search", "query": query.strip()}
    if kind == "open":
        docid = action.get("docid")
        if not isinstance(docid, (str, int)) or not str(docid).strip():
            raise ValueError("open needs a docid")
        try:
            offset = int(action.get("offset") or 0)
        except (TypeError, ValueError):
            raise ValueError("offset must be an integer") from None
        return {"action": "open", "docid": str(docid).strip(), "offset": offset}
    if kind == "finish" and single:
        answer = action.get("exact_answer")
        if not isinstance(answer, (str, int, float)) or not str(answer).strip():
            raise ValueError("finish needs an exact_answer")
        return {
            "action": "finish",
            "explanation": str(action.get("explanation", "")),
            "exact_answer": str(answer).strip(),
            "confidence": confidence(action.get("confidence")),
        }
    if kind == "finish":
        passages = action.get("key_passages") or []
        if not isinstance(passages, list):
            raise ValueError("key_passages must be a list of passage IDs")
        status = action.get("status")
        return {
            "action": "finish",
            "status": status if status in STATUSES else "partial",
            "key_passages": [str(p) for p in passages],
        }
    raise ValueError('action must be "search", "open" or "finish"')


async def run_search(client, endpoint: str, session, *, task: str, limits, seed: int | None, tag: dict,
                     single: bool = False) -> Trajectory:
    """Search until the model finishes, hits the step limit, keeps sending invalid actions, or fills its context.

    tag needs role, round and subagent. In single-agent mode, task is the full question and finish carries the answer.
    """
    if single:
        system, user = render("single_system", max_steps=limits.max_steps), render("single_question", question=task)
    else:
        system, user = render("subagent_system", max_steps=limits.max_steps), render("subagent_task", task=task)
    trajectory = Trajectory(subagent=tag["subagent"], round=tag["round"], subtask=task,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    messages = trajectory.messages  # pydantic copies lists on validation, so append to the model's own
    ep = client.endpoints[endpoint]
    streak = 0
    for step in range(limits.max_steps + 1):
        out_of_room = bool(trajectory.calls) and _next_input(trajectory, ep.tokenizer) + limits.call_max_tokens > ep.context_window
        forced = step == limits.max_steps or out_of_room
        if forced:
            messages.append({"role": "user", "content": render("force_finish")})
        reply = await client.chat(endpoint, messages, {**tag, "phase": "search", "step": step}, thinking=True,
                                  max_tokens=limits.call_max_tokens, seed=seed)
        trajectory.calls.append(reply.record)
        trajectory.steps = step + 1
        messages.append({"role": "assistant", "content": reply.text})
        try:
            action = parse_action(reply.text, single)
            if action["action"] == "finish":
                trajectory.actions.append(action)
                _finish(trajectory, action, session)
                break
            if forced:
                trajectory.status = "context_limit" if out_of_room else "step_limit"
                break
            if action["action"] == "search":
                observation = session.search(action["query"])
            else:
                observation = session.open(action["docid"], action["offset"])
            trajectory.actions.append(action)
            streak = 0
        except ValueError as error:
            trajectory.invalid_steps += 1
            streak += 1
            if forced or streak >= limits.max_invalid_streak:
                trajectory.status = ("context_limit" if out_of_room else "step_limit") if forced else "invalid_actions"
                break
            observation = render("invalid_action", error=str(error))
        messages.append({"role": "user", "content": observation})
    trajectory.retrieved, trajectory.opened, trajectory.seen = session.retrieved, session.opened, session.seen
    return trajectory


def _next_input(trajectory: Trajectory, tokenizer: str) -> int:
    """Estimated input of the next call: the last call's input and output plus the newest observation."""
    last = trajectory.calls[-1]
    return last.input_tokens + last.output_tokens + tokens.count(tokenizer, trajectory.messages[-1]["content"])


def _finish(trajectory: Trajectory, action: dict, session) -> None:
    if "exact_answer" in action:
        trajectory.answer = {k: action[k] for k in ("explanation", "exact_answer", "confidence")}
        trajectory.status = "answered"
        return
    flagged = list(dict.fromkeys(action["key_passages"]))
    trajectory.flagged = [pid for pid in flagged if pid in session.seen]
    trajectory.dropped_flags = len(flagged) - len(trajectory.flagged)
    trajectory.status = action["status"]
