import asyncio
import gzip
import json

import pytest

from harness import grade
from harness.config import Endpoint
from harness.pipeline import read_gz, write_atomic
from harness.run import load_questions
from conftest import mock_endpoint, run_grid
from mock_llm import mock_client


@pytest.fixture
def run_dir(mock, make_cfg, tmp_path):
    """Both test questions through the quotes and short formats at N = 1 and 3."""
    cfg = make_cfg(formats=["quotes", "short"], n_values=[1, 3])
    path = tmp_path / "runs" / "test"
    run_grid(mock, cfg, path, questions=load_questions(cfg))
    return path


@pytest.fixture
def labels(data_dir):
    return data_dir / "labels.jsonl"


def judge(mock, run_dir, labels, **endpoint):
    client = mock_client(mock, {"judge": Endpoint.model_validate(mock_endpoint(model="judge-model", **endpoint))})
    return asyncio.run(grade.judge(client, "judge", run_dir, labels))


@pytest.mark.parametrize("reply", [
    "**extracted_final_answer:** ZEBRA-42\n\n**reasoning:** same\n\n**correct:** yes\n\n**confidence:** 85%",
    "**extracted_final_answer**: ZEBRA-42\n**correct**: Yes\n**confidence**: 85",
    "extracted_final_answer: ZEBRA-42\n[correct_answer]: ZEBRA-42\nreasoning: same\ncorrect: yes\nconfidence: 85%",
])
def test_parse_judgment_reads_the_official_layouts(reply):
    assert grade.parse_judgment(reply) == {"correct": True, "extracted_final_answer": "ZEBRA-42",
                                           "confidence": 85.0, "parse_error": False}


def test_parse_judgment_flags_unreadable_replies():
    assert grade.parse_judgment("I think it is right.")["parse_error"]
    assert grade.parse_judgment("correct: no\nconfidence: 250")["confidence"] == 100.0


def test_judge_grades_every_result_with_the_official_prompt(mock, run_dir, labels):
    counts = judge(mock, run_dir, labels)
    grades = grade.load_grades(run_dir)
    assert counts == {"graded": 8, "already_graded": 0, "failed": 0} and len(grades) == 8
    assert grades["n1/s0/quotes/q1.json.gz"]["correct"] is True
    assert grades["n1/s0/quotes/q2.json.gz"]["correct"] is False  # the mock orchestrator answers ZEBRA-42 to q2 too
    body = mock.bodies("judge")[0]
    prompt = body["messages"][0]["content"]
    assert prompt.startswith("Judge whether the following [response]") and "[correct_answer]: " in prompt
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 4096  # thinking off
    assert grades["n1/s0/quotes/q1.json.gz"]["call"]["input_tokens"] > 0


def test_judge_skips_graded_results_and_retries_unparsed_ones(mock, run_dir, labels):
    mock.script["judge"].append("I cannot tell.")
    judge(mock, run_dir, labels)
    assert sum(g["parse_error"] for g in grade.load_grades(run_dir).values()) == 1
    before = mock.count("judge")
    assert judge(mock, run_dir, labels) == {"graded": 1, "already_graded": 7, "failed": 0}
    assert mock.count("judge") == before + 1
    assert not any(g["parse_error"] for g in grade.load_grades(run_dir).values())


def test_judge_failures_are_left_ungraded(mock, run_dir, labels):
    mock.script["judge"].append(500)
    counts = judge(mock, run_dir, labels, max_retries=0)
    assert counts["failed"] == 1 and len(grade.load_grades(run_dir)) == 7


def test_a_failed_decision_is_wrong_without_asking_the_judge(mock, run_dir, labels):
    path = run_dir / "results" / "n1" / "s0" / "quotes" / "q1.json.gz"
    data = json.loads(read_gz(path))
    data["answer"] = {"decision": "failed", "exact_answer": "", "response": "Explanation: \nExact Answer: "}
    write_atomic(path, gzip.compress(json.dumps(data).encode()))
    judge(mock, run_dir, labels)
    g = grade.load_grades(run_dir)["n1/s0/quotes/q1.json.gz"]
    assert g["correct"] is False and g["judged"] is False and mock.count("judge") == 7


def test_review_samples_judgments_for_the_hand_check(mock, run_dir, labels):
    judge(mock, run_dir, labels)
    text = grade.review(run_dir, sample=3)
    assert text.count("- [ ] the verdict is right") == 3 and "**Correct answer:**" in text


def test_export_writes_the_browsecomp_plus_run_format(run_dir, labels):
    out = grade.export(run_dir, labels)
    assert sorted(p.name for p in out.iterdir()) == ["ground_truth.jsonl", "n1-s0-quotes", "n1-s0-short",
                                                    "n3-s0-quotes", "n3-s0-short", "qrel_evidence.txt"]
    record = json.loads((out / "n3-s0-quotes" / "q1.json").read_text())
    assert record["status"] == "completed" and record["result"][-1]["type"] == "output_text"
    assert "Exact Answer: ZEBRA-42" in record["result"][-1]["output"]
    assert "d01" in record["retrieved_docids"] and record["tool_call_counts"]["search"] >= 3
    truth = [json.loads(line) for line in (out / "ground_truth.jsonl").read_text().splitlines()]
    assert truth[0] == {"query_id": "q1", "query": truth[0]["query"], "answer": "ZEBRA-42"}
    assert (out / "qrel_evidence.txt").read_text().splitlines()[0] == "q1 0 d01 1"
