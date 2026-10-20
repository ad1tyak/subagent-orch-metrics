import asyncio

import pytest

from harness.config import Limits
from harness.prompts import TEMPLATES
from harness.subagent import parse_action, run_search
from mock_llm import mock_client

TAG = {"role": "subagent", "round": 1, "subagent": 1}
TASK = "Find the heritage registry code of the iron bridge over the river Tamel"


def search(mock, session, make_cfg, limits=None, endpoint=None, single=False):
    cfg = make_cfg(endpoints={"orch": cfg_endpoint(), "sub": endpoint or cfg_endpoint()})
    client = mock_client(mock, cfg.endpoints)
    return asyncio.run(run_search(client, "sub", session, task=TASK, limits=limits or Limits(max_steps=6), seed=0,
                                  tag=TAG, single=single))


def cfg_endpoint(**overrides):
    from conftest import mock_endpoint
    return mock_endpoint(**overrides)


def test_parse_valid_actions():
    assert parse_action('{"action": "search", "query": " iron bridge "}') == {"action": "search", "query": "iron bridge"}
    assert parse_action('{"action": "open", "docid": 17, "offset": "2"}') == {"action": "open", "docid": "17", "offset": 2}
    assert parse_action('{"action": "finish", "status": "found", "key_passages": ["d01#p0"]}') == {
        "action": "finish", "status": "found", "key_passages": ["d01#p0"]}
    assert parse_action('{"action": "finish", "status": "done?"}')["status"] == "partial"


def test_parse_json_inside_prose_and_code_fences():
    reply = 'I will search now.\n```json\n{"action": "search", "query": "Tamel"}\n```'
    assert parse_action(reply)["query"] == "Tamel"


@pytest.mark.parametrize("reply, message", [
    ("no json here", "no JSON"),
    ('{"action": "dance"}', "action must be"),
    ('{"action": "search", "query": ""}', "non-empty query"),
    ('{"action": "open", "docid": "d01", "offset": "x"}', "offset"),
    ('{"action": "finish", "key_passages": "d01#p0"}', "list"),
])
def test_parse_rejects_invalid_actions(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_action(reply)


def test_single_agent_finish_needs_an_answer():
    with pytest.raises(ValueError, match="exact_answer"):
        parse_action('{"action": "finish", "status": "found"}', single=True)
    assert parse_action('{"action": "finish", "exact_answer": "ZEBRA-42", "confidence": "75%"}', single=True)[
        "confidence"] == 75


def test_default_run_searches_opens_and_finishes(mock, session, make_cfg):
    trajectory = search(mock, session, make_cfg)
    assert [a["action"] for a in trajectory.actions] == ["search", "open", "open", "finish"]
    assert trajectory.status == "found" and trajectory.opened[0] == "d01"
    assert trajectory.flagged == [f"{d}#p0" for d in trajectory.opened]
    assert trajectory.retrieved[0] == "d01" and "d01#p0" in trajectory.seen
    assert [c.step for c in trajectory.calls] == [0, 1, 2, 3]
    assert all(c.phase == "search" and c.role == "subagent" for c in trajectory.calls)
    assert all(body["reasoning_effort"] == "medium" for body in mock.bodies("search"))  # thinking on


def test_step_limit_forces_finish(mock, session, make_cfg):
    mock.script["search"].extend([{"action": "search", "query": "bridge"}] * 10)
    trajectory = search(mock, session, make_cfg, limits=Limits(max_steps=3))
    assert len(trajectory.calls) == 4 and trajectory.status == "step_limit" and trajectory.flagged == []
    assert mock.bodies("search")[-1]["messages"][-1]["content"] == TEMPLATES["force_finish"]


def test_forced_finish_is_obeyed(mock, session, make_cfg):
    mock.script["search"].extend([{"action": "search", "query": "bridge"}] * 3)
    trajectory = search(mock, session, make_cfg, limits=Limits(max_steps=3))
    assert trajectory.actions[-1]["action"] == "finish" and trajectory.status == "not_found"


def test_three_invalid_actions_in_a_row_end_the_search(mock, session, make_cfg):
    mock.script["search"].extend(["no json", "{broken", '{"action": "dance"}'])
    trajectory = search(mock, session, make_cfg)
    assert trajectory.status == "invalid_actions" and trajectory.invalid_steps == 3 and len(trajectory.calls) == 3


def test_an_invalid_action_gets_feedback_and_the_search_continues(mock, session, make_cfg):
    mock.script["search"].extend(['{"action": "open", "docid": "d99"}', {"action": "search", "query": TASK}])
    trajectory = search(mock, session, make_cfg)
    assert trajectory.messages[3]["content"].startswith("Invalid action: no document has docid d99")
    assert trajectory.invalid_steps == 1 and trajectory.status == "found"


def test_finishing_before_any_search_is_invalid(mock, session, make_cfg):
    mock.script["search"].append({"action": "finish", "status": "not_found", "key_passages": []})
    trajectory = search(mock, session, make_cfg)
    assert trajectory.messages[3]["content"].startswith("Invalid action: search the collection before you finish")
    assert trajectory.invalid_steps == 1 and trajectory.status == "found"
    assert [a["action"] for a in trajectory.actions][0] == "search"


def test_flags_on_unseen_passages_are_dropped(mock, session, make_cfg):
    mock.script["search"].extend([
        {"action": "search", "query": "iron bridge Tamel"},
        {"action": "open", "docid": "d01"},
        {"action": "finish", "status": "found", "key_passages": ["d01#p0", "d01#p0", "d01#p9", "d02#p0"]},
    ])
    trajectory = search(mock, session, make_cfg)
    assert trajectory.flagged == ["d01#p0"] and trajectory.dropped_flags == 2


def test_a_full_context_forces_finish(mock, session, make_cfg):
    mock.script["search"].extend([{"action": "search", "query": "bridge"}] * 5)
    trajectory = search(mock, session, make_cfg, limits=Limits(max_steps=6, call_max_tokens=50),
                        endpoint=cfg_endpoint(context_window=300))
    assert trajectory.status == "context_limit" and len(trajectory.calls) == 2


def test_single_agent_mode_answers(mock, session, make_cfg):
    trajectory = search(mock, session, make_cfg, single=True)
    assert trajectory.status == "answered" and trajectory.answer["exact_answer"] == "ZEBRA-42"
