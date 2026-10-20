"""The six return formats and the report wrapper they all share."""

import json
import re

from . import tokens
from .parsing import first_json
from .prompts import render
from .records import Report, Trajectory
from .search import bm25, terms

SUMMARIES = ("short", "medium", "long")
WRITTEN = SUMMARIES + ("claims",)
_PASSAGE_ID = re.compile(r"[\w.:/-]+#p\d+")


def wrap(trajectory: Trajectory, content: str) -> str:
    return (f'<report subagent="{trajectory.subagent}" status="{trajectory.status}">\n'
            f"<subtask>{trajectory.subtask}</subtask>\n{content}\n</report>")


def pack(items: list[tuple[str, str]], budget: int, tokenizer: str) -> tuple[str, list[str], bool]:
    """Join (passage ID, text) items in order, stopping before the one that would pass the budget."""
    kept, ids = [], []
    for pid, text in items:
        if tokens.count(tokenizer, "\n\n".join(kept + [text])) > budget:
            return "\n\n".join(kept), ids, True
        kept.append(text)
        ids.append(pid)
    return "\n\n".join(kept), ids, False


def _doc(pid: str) -> str:
    return pid.rsplit("#p", 1)[0]


def opened_order(trajectory: Trajectory) -> list[str]:
    """Seen passages grouped by document in the order documents were opened, then by position."""
    return sorted(trajectory.seen, key=lambda pid: (trajectory.opened.index(_doc(pid)), int(pid.rsplit("#p", 1)[1])))


def ranked_passages(trajectory: Trajectory) -> list[str]:
    """Seen passages by BM25 score against the subtask: the quotes fallback when nothing was flagged."""
    pids = list(trajectory.seen)
    scores = bm25(terms(trajectory.subtask), [terms(trajectory.seen[pid]) for pid in pids])
    return [pid for _, pid in sorted(zip(scores, pids), key=lambda pair: -pair[0])]


def parse_claims(text: str) -> dict | None:
    try:
        obj = first_json(text)
    except ValueError:
        return None
    claims = obj.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, dict) and isinstance(c.get("claim"), str) for c in claims):
        return None
    evidence = [c.get("evidence") or [] for c in claims]
    return {
        "searched": obj["searched"] if isinstance(obj.get("searched"), list) else [],
        "claims": [{"claim": c["claim"], "evidence": [str(e) for e in (ev if isinstance(ev, list) else [ev])]}
                   for c, ev in zip(claims, evidence)],
        "status": str(obj.get("status", "")),
    }


async def make_report(fmt: str, trajectory: Trajectory, budget: int, tokenizer: str, *, client=None, endpoint=None,
                      limits=None, seed=None) -> Report:
    """One subagent's report. tokenizer is the orchestrator's, since the cap is on what the orchestrator reads.

    Formats 1 to 4 add a writing turn to the subagent's conversation, with thinking off; quotes and raw make no call.
    """
    write_messages, calls = [], []
    valid, hit_cap, cited = True, False, None

    async def write(prompt: str) -> str:
        write_messages.append({"role": "user", "content": prompt})
        reply = await client.chat(
            endpoint, trajectory.messages + write_messages,
            {"role": "subagent", "phase": "write", "round": trajectory.round, "subagent": trajectory.subagent,
             "step": len(calls)},
            thinking=False, max_tokens=budget + limits.writing_reasoning_allowance, seed=seed,
        )
        calls.append(reply.record)
        write_messages.append({"role": "assistant", "content": reply.text})
        return reply.text.strip()

    if fmt in SUMMARIES:
        content = await write(render("write_summary", budget=budget))
    elif fmt == "claims":
        content = await write(render("write_claims", budget=budget))
        claims = parse_claims(content)
        if claims is None:
            content = await write(render("claims_repair"))
            claims = parse_claims(content)
        if claims is None:
            valid = False
        else:
            for claim in claims["claims"]:
                claim["evidence"] = [pid for pid in claim["evidence"] if pid in trajectory.seen]
            claims["status"] = trajectory.status
            content = json.dumps(claims, ensure_ascii=False)
            while claims["claims"] and tokens.count(tokenizer, content) > budget:
                claims["claims"].pop()
                content, hit_cap = json.dumps(claims, ensure_ascii=False), True
            cited = [pid for claim in claims["claims"] for pid in claim["evidence"]]
    elif fmt == "quotes":
        pids = trajectory.flagged or ranked_passages(trajectory)
        content, cited, hit_cap = pack([(pid, f"[{pid}] {trajectory.seen[pid]}") for pid in pids], budget, tokenizer)
    elif fmt == "raw":
        pids = opened_order(trajectory)
        content, cited, hit_cap = pack([(pid, f"[{pid}] {trajectory.seen[pid]}") for pid in pids], budget, tokenizer)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    if tokens.count(tokenizer, content) > budget:
        content, hit_cap = tokens.truncate(tokenizer, content, budget), True
    if cited is None:
        cited = _PASSAGE_ID.findall(content)
    cited = [pid for pid in dict.fromkeys(cited) if pid in trajectory.seen]
    content = content or render("empty_report")
    text = wrap(trajectory, content)
    return Report(
        format=fmt, text=text, tokens=tokens.count(tokenizer, text), content_tokens=tokens.count(tokenizer, content),
        hit_cap=hit_cap, valid=valid, delivered_docids=list(dict.fromkeys(_doc(pid) for pid in cited)),
        write_messages=write_messages, calls=calls,
    )
