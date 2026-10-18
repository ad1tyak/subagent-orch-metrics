import asyncio
import json

import pytest

from harness import clues
from harness.config import Endpoint
from conftest import QUESTIONS, mock_endpoint
from mock_llm import mock_client


@pytest.fixture
def client(mock):
    return mock_client(mock, {"e": Endpoint.model_validate(mock_endpoint())})


def test_parse_clues_takes_the_list_inside_prose():
    assert clues.parse_clues('Here:\n```json\n{"clues": [" built in 1887 ", "", 3, "over the Tamel"]}\n```') == [
        "built in 1887", "over the Tamel"]
    for bad in ("no json", '{"clue": ["x"]}', '{"clues": []}', '{"clues": "x"}'):
        with pytest.raises(ValueError):
            clues.parse_clues(bad)


def test_extract_lists_the_clues_of_a_question(mock, client):
    record = asyncio.run(clues.extract(client, "e", QUESTIONS[0]))
    assert record["valid"] and len(record["clues"]) >= 3 and len(record["calls"]) == 1
    body = mock.bodies("clues")[0]
    assert QUESTIONS[0]["question"] in body["messages"][0]["content"]
    assert body["reasoning_effort"] == "medium"  # thinking on


def test_a_bad_reply_gets_one_repair_and_then_counts_as_invalid(mock, client):
    mock.script["clues"].extend(["no json here", '{"clues": ["a", "b"]}'])
    record = asyncio.run(clues.extract(client, "e", QUESTIONS[0]))
    assert record["valid"] and record["clues"] == ["a", "b"] and len(record["calls"]) == 2
    assert "Invalid reply" in mock.bodies("clues")[1]["messages"][-1]["content"]

    mock.script["clues"].extend(["still not json", '{"clues": []}'])
    record = asyncio.run(clues.extract(client, "e", QUESTIONS[0]))
    assert not record["valid"] and record["clues"] == []


def test_a_rerun_asks_only_about_missing_or_invalid_questions(mock, client, tmp_path):
    raw = tmp_path / "clues.raw.jsonl"
    mock.script["clues"].extend(["bad", "bad"])  # the first question comes back invalid
    asyncio.run(clues.run(client, "e", QUESTIONS[:1], raw))
    assert mock.count("clues") == 2

    records = asyncio.run(clues.run(client, "e", QUESTIONS, raw))
    assert mock.count("clues") == 4  # one retry for the invalid question, one for the new one
    assert all(r["valid"] for r in records.values())

    asyncio.run(clues.run(client, "e", QUESTIONS, raw))
    assert mock.count("clues") == 4
    assert len(raw.read_text().splitlines()) == 3  # every reply is kept; the latest one wins


def test_a_failed_request_is_left_for_the_next_run(mock, client, tmp_path):
    raw = tmp_path / "clues.raw.jsonl"
    mock.script["clues"].extend([500, 500, 500])
    records = asyncio.run(clues.run(client, "e", QUESTIONS[:1], raw))
    assert records == {} and raw.read_text() == ""


def test_outputs_chart_and_hand_check_sample(mock, client, tmp_path):
    out = tmp_path / "clues.json"
    records = asyncio.run(clues.run(client, "e", QUESTIONS, out.with_suffix(".raw.jsonl")))
    summary = clues.write_outputs(QUESTIONS, records, out, sample=1)
    saved = json.loads(out.read_text())
    assert set(saved) == {"q1", "q2"} and saved["q1"] == records["q1"]["clues"]
    assert "N = 3:" in summary and "2 of 2 questions have clues" in summary
    review = out.with_suffix(".review.md").read_text()
    assert review.count("## Question") == 1 and "- [ ] no clue from the question is missing" in review


def test_chart_shows_which_n_values_the_data_supports():
    text = clues.chart([1, 3, 3, 6, 7])
    assert "  3 clues | " in text and text.splitlines()[1].endswith(" 2")
    assert "N = 6: 2 of 5 questions have at least 6 clues" in text
    assert clues.chart([]) == "no questions have clues yet"
