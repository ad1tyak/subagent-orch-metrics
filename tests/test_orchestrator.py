import asyncio

import pytest

from harness.config import Limits
from harness.orchestrator import decide, fit_round, parse_decision, render_answer, round_message
from harness.records import Report
from mock_llm import mock_client


def test_parse_answer_and_redelegate():
    d = parse_decision('{"decision": "answer", "exact_answer": " ZEBRA-42 ", "confidence": "80%"}', 3, False)
    assert d == {"decision": "answer", "explanation": "", "exact_answer": "ZEBRA-42", "confidence": 80}
    d = parse_decision('{"decision": "redelegate", "subtasks": [{"task": "a"}, "b", {"task": ""}, {"task": "c"}]}',
                       2, False)
    assert d == {"decision": "redelegate", "subtasks": [{"task": "a"}, {"task": "b"}]}


def test_final_round_rejects_redelegation():
    with pytest.raises(ValueError, match='must be "answer"'):
        parse_decision('{"decision": "redelegate", "subtasks": ["a"]}', 3, True)


@pytest.mark.parametrize("reply", ["no json", '{"decision": "maybe"}', '{"decision": "answer"}',
                                   '{"decision": "redelegate", "subtasks": []}'])
def test_unusable_decisions_raise(reply):
    with pytest.raises(ValueError):
        parse_decision(reply, 3, False)


def test_answer_renders_in_browsecomp_style():
    assert render_answer({"explanation": "Seen in d01.", "exact_answer": "ZEBRA-42", "confidence": 80.0}) == (
        "Explanation: Seen in d01.\nExact Answer: ZEBRA-42\nConfidence: 80%")
    assert "Confidence" not in render_answer({"explanation": "", "exact_answer": "", "confidence": None})


def make_reports(words=100):
    return [Report(format="raw", text=" ".join(["w"] * words), tokens=words, content_tokens=words) for _ in range(3)]


def test_round_message_trims_reports_to_equal_shares():
    reports = make_reports()
    message = round_message("HEADER", reports, "INSTRUCTION", "words", room=60)
    assert all(r.trimmed for r in reports)
    assert message.split("\n\n") == ["HEADER", *[" ".join(["w"] * 20)] * 3, "INSTRUCTION"]


def test_fit_round_trims_only_when_the_context_would_overflow():
    messages = [{"role": "system", "content": "s " * 50}]
    reports = make_reports()
    assert fit_round(messages, "H", reports, "I", "words", 1000, 100).count("w") == 300
    assert not any(r.trimmed for r in reports)
    trimmed = fit_round(messages, "H", reports, "I", "words", 300, 100)
    assert all(r.trimmed for r in reports) and trimmed.count("w") <= 150


def run_decide(mock, make_cfg, final=False):
    client = mock_client(mock, make_cfg().endpoints)
    messages = [{"role": "system", "content": "You are the lead agent"}, {"role": "user", "content": "reports"}]
    return asyncio.run(decide(client, "orch", messages, n=3, final=final, round_no=1, limits=Limits(), seed=0))


def test_decide_repairs_one_bad_reply(mock, make_cfg):
    mock.script["other"].extend(["garbage", '{"decision": "answer", "exact_answer": "ZEBRA-42"}'])
    decision, calls = run_decide(mock, make_cfg)
    assert decision["exact_answer"] == "ZEBRA-42" and len(calls) == 2
    assert all(c.phase == "decide" for c in calls)


def test_decide_fails_after_two_bad_replies(mock, make_cfg):
    mock.script["other"].extend(["garbage", "more garbage"])
    decision, calls = run_decide(mock, make_cfg)
    assert decision["decision"] == "failed" and decision["exact_answer"] == "" and len(calls) == 2
