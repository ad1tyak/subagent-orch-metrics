import asyncio
import json

import pytest

from harness.config import FORMATS, Limits
from harness.formats import make_report
from harness.prompts import render
from harness.records import Trajectory
from mock_llm import mock_client

SUBTASK = "Find the registry code of the iron bridge"


@pytest.fixture
def trajectory():
    return Trajectory(
        subagent=2, round=1, subtask=SUBTASK, status="found",
        messages=[{"role": "system", "content": render("subagent_system", max_steps=6)},
                  {"role": "user", "content": render("subagent_task", task=SUBTASK)}],
        opened=["d01", "d07"],
        seen={  # insertion order differs from document order on purpose
            "d07#p0": "The heritage registry assigns a code to every protected structure.",
            "d01#p1": "It was designed by Joaquim Seabra.",
            "d01#p0": "The bridge registry code is ZEBRA-42.",
        },
        flagged=["d01#p0"],
    )


def report(mock, make_cfg, fmt, trajectory, budget):
    client = mock_client(mock, make_cfg().endpoints)
    return asyncio.run(make_report(fmt, trajectory, budget, "words", client=client, endpoint="sub", limits=Limits(),
                                   seed=0))


def test_quotes_golden_output_with_no_model_call(mock, make_cfg, trajectory):
    r = report(mock, make_cfg, "quotes", trajectory, 2000)
    assert r.text == ('<report subagent="2" status="found">\n<subtask>Find the registry code of the iron bridge'
                      '</subtask>\n[d01#p0] The bridge registry code is ZEBRA-42.\n</report>')
    assert mock.requests == [] and r.calls == [] and r.delivered_docids == ["d01"]
    assert r.tokens == 19 and r.content_tokens == 7 and not r.hit_cap


def test_quotes_fall_back_to_bm25_ranking_when_nothing_is_flagged(mock, make_cfg, trajectory):
    trajectory.flagged = []
    r = report(mock, make_cfg, "quotes", trajectory, 2000)
    assert r.text.split("\n")[2].startswith("[d01#p0]")
    assert r.delivered_docids == ["d01", "d07"]


def test_raw_groups_by_document_in_opened_order_and_caps(mock, make_cfg, trajectory):
    full = report(mock, make_cfg, "raw", trajectory, 8000)
    assert [line.split("]")[0] + "]" for line in full.text.split("\n")[2:-1] if line] == [
        "[d01#p0]", "[d01#p1]", "[d07#p0]"]
    capped = report(mock, make_cfg, "raw", trajectory, 14)
    assert capped.hit_cap and capped.delivered_docids == ["d01"] and capped.content_tokens <= 14
    assert mock.requests == []


def test_summary_writes_with_thinking_off_and_truncates_at_the_cap(mock, make_cfg, trajectory):
    mock.script["summary"].append("The code is ZEBRA-42 [d01#p0] by Seabra [d01#p1] and [d99#p5]. " + "filler " * 300)
    r = report(mock, make_cfg, "short", trajectory, 50)
    assert r.hit_cap and r.content_tokens == 50 and len(r.calls) == 1
    assert r.delivered_docids == ["d01"]  # d99 was never seen
    body = mock.bodies("summary")[0]
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 50 + Limits().writing_reasoning_allowance
    assert body["messages"][:2] == trajectory.messages and len(trajectory.messages) == 2  # not mutated
    assert r.calls[0].phase == "write" and r.calls[0].subagent == 2


def test_claims_valid_json_is_compacted_and_cited(mock, make_cfg, trajectory):
    mock.script["claims"].append('Sure:\n{"searched": ["registry code"], "status": "found", "claims": [\n'
                                 '  {"claim": "The code is ZEBRA-42", "evidence": ["d01#p0"]},\n'
                                 '  {"claim": "The registry assigns codes", "evidence": "d07#p0"}]}')
    r = report(mock, make_cfg, "claims", trajectory, 800)
    content = json.loads(r.text.split("\n", 2)[2].rsplit("\n", 1)[0])
    assert content["claims"][1]["evidence"] == ["d07#p0"] and r.valid and len(r.calls) == 1
    assert r.delivered_docids == ["d01", "d07"]


def test_claims_are_dropped_from_the_end_to_fit(mock, make_cfg, trajectory):
    claims = [{"claim": f"fact number {i} " + "word " * 10, "evidence": [f"d01#p{i % 2}"]} for i in range(20)]
    mock.script["claims"].append({"searched": [], "claims": claims, "status": "found"})
    r = report(mock, make_cfg, "claims", trajectory, 60)
    assert r.hit_cap and r.valid and r.content_tokens <= 60
    json.loads(r.text.split("\n", 2)[2].rsplit("\n", 1)[0])  # still valid JSON


def test_invalid_claims_get_one_repair(mock, make_cfg, trajectory):
    mock.script["claims"].append("Here are my claims: none.")
    r = report(mock, make_cfg, "claims", trajectory, 800)
    assert r.valid and len(r.calls) == 2


def test_claims_invalid_twice_are_delivered_raw_and_flagged(mock, make_cfg, trajectory):
    mock.script["claims"].extend(["not json", "still not json [d01#p0]"])
    r = report(mock, make_cfg, "claims", trajectory, 800)
    assert not r.valid and "still not json" in r.text and r.delivered_docids == ["d01"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_shares_the_wrapper_and_respects_its_cap(mock, make_cfg, trajectory, fmt):
    r = report(mock, make_cfg, fmt, trajectory, 20)
    lines = r.text.split("\n")
    assert lines[:2] == ['<report subagent="2" status="found">', f"<subtask>{SUBTASK}</subtask>"]
    assert lines[-1] == "</report>" and r.content_tokens <= 20 and r.format == fmt
    assert bool(r.calls) == (fmt not in ("quotes", "raw"))
