import asyncio

from harness.config import Limits
from harness.planner import fix_count, make_plan
from mock_llm import mock_client


def plan(mock, make_cfg, q, n, n_mode="fixed"):
    cfg = make_cfg()
    client = mock_client(mock, cfg.endpoints)
    return asyncio.run(make_plan(client, "orch", q["question"], q["clues"], n, n_mode=n_mode, limits=Limits(), seed=0))


def subtasks(*clue_groups):
    return {"subtasks": [{"clues": list(g), "task": f"task {i}"} for i, g in enumerate(clue_groups)]}


def all_clues(p):
    return sorted(c for s in p.subtasks for c in s["clues"])


def test_n_equals_one_makes_no_call(mock, make_cfg, questions):
    p = plan(mock, make_cfg, questions[0], 1)
    assert mock.requests == [] and p.actual_n == 1 and p.subtasks[0]["task"] == questions[0]["question"]


def test_well_behaved_planner_gives_exactly_n(mock, make_cfg, questions):
    p = plan(mock, make_cfg, questions[0], 3)
    assert (p.requested_n, p.actual_n, p.repaired, p.fixed) == (3, 3, False, False)
    assert all_clues(p) == [1, 2, 3, 4, 5, 6] and len(p.calls) == 1
    assert p.calls[0].phase == "plan" and p.calls[0].role == "orchestrator"


def test_too_many_subtasks_are_repaired_then_merged(mock, make_cfg, questions):
    mock.script["plan"].extend([subtasks([1], [2], [3, 4], [5, 6])] * 2)
    p = plan(mock, make_cfg, questions[0], 3)
    assert (p.actual_n, p.repaired, p.fixed) == (3, True, True) and all_clues(p) == [1, 2, 3, 4, 5, 6]
    assert "exactly 3 subtasks" in mock.bodies("plan")[1]["messages"][-1]["content"]


def test_too_few_subtasks_are_split_by_clues(mock, make_cfg, questions):
    mock.script["plan"].extend([subtasks([1, 2, 3, 4], [5, 6])] * 2)
    p = plan(mock, make_cfg, questions[0], 3)
    assert p.actual_n == 3 and p.fixed and [s["clues"] for s in p.subtasks] == [[1, 2], [3, 4], [5, 6]]
    assert "an iron bridge completed in 1887" in p.subtasks[0]["task"]


def test_malformed_reply_gets_one_repair(mock, make_cfg, questions):
    mock.script["plan"].append("I think we should split it up.")
    p = plan(mock, make_cfg, questions[0], 3)
    assert (p.actual_n, p.repaired, p.fixed) == (3, True, False)


def test_malformed_twice_falls_back_to_even_clue_groups(mock, make_cfg, questions):
    mock.script["plan"].extend(["nope", "still nope"])
    p = plan(mock, make_cfg, questions[0], 3)
    assert p.fixed and [s["clues"] for s in p.subtasks] == [[1, 2], [3, 4], [5, 6]]


def test_fewer_clues_than_n_sets_actual_n(mock, make_cfg, questions):
    p = plan(mock, make_cfg, questions[1], 6)
    assert (p.requested_n, p.actual_n) == (6, 2)
    assert "exactly 2 subtasks" in mock.bodies("plan")[0]["messages"][0]["content"]


def test_free_mode_accepts_fewer_subtasks(mock, make_cfg, questions):
    mock.script["plan"].append(subtasks([1, 2, 3], [4, 5, 6]))
    p = plan(mock, make_cfg, questions[0], 3, n_mode="free")
    assert (p.actual_n, p.repaired, p.fixed) == (2, False, False)


def test_fix_count_stops_when_clues_cannot_be_split():
    one_clue_each = [{"task": "a", "clues": [1]}, {"task": "b", "clues": [2]}]
    assert len(fix_count(one_clue_each, 3, ["x", "y"])) == 2
