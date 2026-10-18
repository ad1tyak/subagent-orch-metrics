"""The orchestrator's plan: group the question's clues into N subtasks."""

from .parsing import first_json
from .prompts import render
from .records import Plan


def clue_task(ids: list[int], clues: list[str]) -> dict:
    return {"task": render("clue_task", clues="\n".join(f"- {clues[i - 1]}" for i in ids)), "clues": list(ids)}


def parse_plan(text: str, n_clues: int) -> list[dict]:
    """Subtasks from a planner reply. Raises ValueError when the reply is unusable."""
    items = first_json(text).get("subtasks")
    if not isinstance(items, list) or not items:
        raise ValueError('the reply needs a non-empty "subtasks" list')
    subtasks = []
    for item in items:
        task = item.get("task") if isinstance(item, dict) else item
        if not isinstance(task, str) or not task.strip():
            raise ValueError('every subtask needs a non-empty "task"')
        raw = item.get("clues") if isinstance(item, dict) else None
        ids = [int(c) for c in raw if str(c).isdigit() and 1 <= int(c) <= n_clues] if isinstance(raw, list) else []
        subtasks.append({"task": task.strip(), "clues": ids})
    return subtasks


def fix_count(subtasks: list[dict], target: int, clues: list[str]) -> list[dict]:
    """Merge the two smallest subtasks, or split the one with the most clues, until there are target of them."""
    subtasks = [dict(s) for s in subtasks]
    while len(subtasks) > target:
        a, b = sorted(sorted(range(len(subtasks)), key=lambda i: (len(subtasks[i]["clues"]), -i))[:2])
        subtasks[a] = {
            "task": subtasks[a]["task"] + "\n" + subtasks[b]["task"],
            "clues": subtasks[a]["clues"] + subtasks[b]["clues"],
        }
        del subtasks[b]
    while len(subtasks) < target:
        i = max(range(len(subtasks)), key=lambda i: len(subtasks[i]["clues"]))
        ids = subtasks[i]["clues"]
        if len(ids) < 2:
            break
        subtasks[i:i + 1] = [clue_task(ids[: len(ids) // 2], clues), clue_task(ids[len(ids) // 2:], clues)]
    return subtasks


async def make_plan(client, endpoint: str, question: str, clues: list[str], n: int, *, n_mode: str, limits,
                    seed: int | None) -> Plan:
    target = min(n, len(clues)) if clues else n
    if target == 1:
        return Plan(requested_n=n, actual_n=1, subtasks=[{"task": question, "clues": list(range(1, len(clues) + 1))}])
    rule = render("plan_exact" if n_mode == "fixed" else "plan_free", n=target)
    listed = "\n".join(f"{i}. {clue}" for i, clue in enumerate(clues, 1)) or render("plan_no_clues")
    prompt = render("planner", question=question, clues=listed, count_rule=rule)
    plan = Plan(requested_n=n, actual_n=0, subtasks=[], messages=[{"role": "user", "content": prompt}])
    messages = plan.messages  # pydantic copies lists on validation, so append to the model's own
    subtasks = None
    for attempt in range(2):
        reply = await client.chat(endpoint, messages, {"role": "orchestrator", "phase": "plan", "step": attempt},
                                  thinking=True, max_tokens=limits.call_max_tokens, seed=seed)
        plan.calls.append(reply.record)
        messages.append({"role": "assistant", "content": reply.text})
        try:
            subtasks = parse_plan(reply.text, len(clues))
        except ValueError as error:
            problem = str(error)
        else:
            if len(subtasks) == target or (n_mode == "free" and len(subtasks) < target):
                break
            problem = f"the reply has {len(subtasks)} subtasks, but it needs {rule}"
        if attempt == 0:
            messages.append({"role": "user", "content": render("plan_repair", problem=problem)})
    plan.repaired = len(plan.calls) > 1
    if subtasks is None:
        ids = list(range(1, len(clues) + 1))
        groups = [ids[k * len(ids) // target:(k + 1) * len(ids) // target] for k in range(target)]
        subtasks = [clue_task(g, clues) for g in groups] if clues else [{"task": question, "clues": []}]
        plan.fixed = True
    elif len(subtasks) > target or (n_mode == "fixed" and len(subtasks) < target):
        subtasks = fix_count(subtasks, target, clues)
        plan.fixed = True
    plan.subtasks, plan.actual_n = subtasks, len(subtasks)
    return plan
