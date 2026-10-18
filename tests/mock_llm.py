"""A scripted OpenAI-compatible model server for offline tests, served through httpx2.MockTransport.

Requests go through the real openai SDK. The mock tells the roles apart by their prompts. A test can
queue replies per role in `script` (a str is the reply text, a dict is sent as JSON, an int is an HTTP
error status); otherwise a simple policy plays the role: subagents search their subtask, open the
top two hits and flag their first passages; writers copy what they saw; the orchestrator answers
with any registry code (like ZEBRA-42) found in the reports, or re-delegates; the clue extractor
splits the question at commas and conjunctions; the judge says yes when the exact answer contains
the correct one.
"""

import asyncio
import json
import re
from collections import defaultdict, deque

import httpx2

from harness.clues import PROMPT as CLUE_PROMPT
from harness.grade import GRADER_TEMPLATE
from harness.llm import ModelClient
from harness.prompts import TEMPLATES

CODE = re.compile(r"\b[A-Z]+-\d+\b")
HIT = re.compile(r"^\d+\. \[([^\]]+)\]", re.M)
PASSAGE = re.compile(r"^\[(\S+#p\d+)\] (.*)$", re.M)
REASONING = "Let me think about the next step."


def prefix(name: str) -> str:
    return TEMPLATES[name].split("$")[0]


def classify(messages: list[dict]) -> str:
    first, last = messages[0]["content"], messages[-1]["content"]
    if first.startswith(prefix("planner")):
        return "plan"
    if first.startswith(prefix("orchestrator_system")):
        return "decide"
    if first.startswith(prefix("single_system")):
        return "single"
    if first.startswith(prefix("subagent_system")):
        if last.startswith(prefix("write_summary")):
            return "summary"
        if last.startswith((prefix("write_claims"), prefix("claims_repair"))):
            return "claims"
        return "search"
    if first.startswith(CLUE_PROMPT.template.split("$")[0]):
        return "clues"
    if first.startswith(GRADER_TEMPLATE.split("{")[0]):
        return "judge"
    return "other"


def answer(code: str, confidence: int = 90) -> dict:
    return {"decision": "answer", "explanation": "from the reports", "exact_answer": code, "confidence": confidence}


def seen_passages(messages: list[dict]) -> list[tuple[str, str]]:
    return PASSAGE.findall("\n".join(m["content"] for m in messages[2:] if m["role"] == "user"))


def _json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class MockLLM:
    def __init__(self, *, delay: float = 0.0, cached: bool = True, reasoning: str = "reported", usage: bool = True):
        self.script = defaultdict(deque)
        self.requests: list[tuple[str, dict]] = []  # (role, request body)
        self.delay, self.cached, self.reasoning, self.usage = delay, cached, reasoning, usage
        self.active = self.max_active = 0

    def count(self, role: str) -> int:
        return sum(r == role for r, _ in self.requests)

    def bodies(self, role: str) -> list[dict]:
        return [body for r, body in self.requests if r == role]

    async def __call__(self, request):
        body = json.loads(request.content)
        role = classify(body["messages"])
        self.requests.append((role, body))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        reply = self.script[role].popleft() if self.script[role] else getattr(self, role)(body["messages"])
        if isinstance(reply, int):
            return httpx2.Response(reply, headers={"retry-after-ms": "1"}, json={"error": {"message": "mock failure"}})
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx2.Response(200, json=self.completion(body, text))

    def completion(self, body: dict, text: str) -> dict:
        prompt_tokens = sum(len((m.get("content") or "").split()) for m in body["messages"])
        message = {"role": "assistant", "content": text}
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split())}
        if self.reasoning != "none":
            message["reasoning_content"] = REASONING
            usage["completion_tokens"] += len(REASONING.split())
        if self.reasoning == "reported":
            usage["completion_tokens_details"] = {"reasoning_tokens": len(REASONING.split())}
        if self.cached:
            usage["prompt_tokens_details"] = {"cached_tokens": prompt_tokens // 2}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage if self.usage else None,
        }

    # Default policies, one per role.

    def plan(self, messages):
        prompt = messages[0]["content"]
        n = int(re.search(r"(?:exactly|between 1 and) (\d+) subtasks", prompt).group(1))
        section = prompt.split("Clues in the question:\n", 1)[1].split("\n\n", 1)[0]
        clues = re.findall(r"^(\d+)\. (.+)$", section, re.M)
        if not clues:
            return {"subtasks": [{"task": f"Part {i + 1} of the question"} for i in range(n)]}
        groups = [clues[i::n] for i in range(n)]
        return {"subtasks": [{"clues": [int(i) for i, _ in g], "task": "Find " + "; ".join(t for _, t in g)}
                             for g in groups if g]}

    def search(self, messages, single=False):
        actions = [parsed for m in messages if m["role"] == "assistant" and (parsed := _json(m["content"]))]
        hits = next((h for m in messages[2:] if m["role"] == "user" and (h := HIT.findall(m["content"]))), [])
        opened = {a.get("docid") for a in actions if a.get("action") == "open"}
        to_open = [docid for docid in hits[:2] if docid not in opened]
        if messages[-1]["content"] != TEMPLATES["force_finish"]:
            if not any(a.get("action") == "search" for a in actions):
                return {"action": "search", "query": messages[1]["content"].split("\n", 1)[1]}
            if to_open:
                return {"action": "open", "docid": to_open[0]}
        seen = seen_passages(messages)
        if single:
            codes = CODE.findall(" ".join(text for _, text in seen))
            return {"action": "finish", "explanation": "found it", "exact_answer": codes[0] if codes else "unknown"}
        return {"action": "finish", "status": "found" if seen else "not_found",
                "key_passages": [pid for pid, _ in seen if pid.endswith("#p0")]}

    def single(self, messages):
        return self.search(messages, single=True)

    def summary(self, messages):
        return " ".join(f"{text} [{pid}]" for pid, text in seen_passages(messages)) or "Nothing found."

    def claims(self, messages):
        return {"searched": ["mock query"], "status": "found",
                "claims": [{"claim": text, "evidence": [pid]} for pid, text in seen_passages(messages)]}

    def decide(self, messages):
        rounds = [m["content"] for m in messages if m["role"] == "user"
                  and not m["content"].startswith(prefix("decide_repair"))]
        codes = CODE.findall("\n".join(rounds))
        if codes:
            return answer(codes[0])
        if prefix("decide_final") in rounds[-1]:
            return answer("unknown", 10)
        return {"decision": "redelegate",
                "subtasks": [{"task": "Find the heritage registry code of the iron bridge over the river Tamel"}]}

    def clues(self, messages):
        question = messages[0]["content"].split("Question:\n", 1)[1].split("\n\nReply", 1)[0]
        parts = [part.strip(" ?.") for part in re.split(r",| that | and ", question)]
        return {"clues": [part for part in parts if len(part.split()) > 2]}

    def judge(self, messages):
        prompt = messages[0]["content"]
        response = prompt.split("[response]: ", 1)[1].split("\n\n[correct_answer]: ", 1)[0]
        correct = prompt.split("\n\n[correct_answer]: ", 1)[1].split("\n\n", 1)[0]
        found = re.search(r"Exact Answer: (.*)", response)
        extracted = found.group(1).strip() if found else response.strip()
        verdict = "yes" if correct.lower() in extracted.lower() else "no"
        return (f"**extracted_final_answer:** {extracted}\n\n**[correct_answer]:** {correct}\n\n"
                f"**reasoning:** compared the strings\n\n**correct:** {verdict}\n\n**confidence:** 90%")

    def other(self, messages):
        return "ok"


def mock_client(mock: MockLLM, endpoints: dict) -> ModelClient:
    return ModelClient(endpoints, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(mock)))
