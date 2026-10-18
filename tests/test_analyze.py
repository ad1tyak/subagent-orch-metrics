import asyncio
import csv

import pytest

from harness import analyze, grade
from harness.config import Endpoint
from harness.run import load_questions
from conftest import mock_endpoint, run_grid
from mock_llm import mock_client

FORMATS = ["short", "medium", "quotes", "raw"]


def graded_run(mock, cfg, run_dir, labels):
    run_grid(mock, cfg, run_dir, questions=load_questions(cfg))
    client = mock_client(mock, {"judge": Endpoint.model_validate(mock_endpoint())})
    asyncio.run(grade.judge(client, "judge", run_dir, labels))
    return run_dir


@pytest.fixture
def labels(data_dir):
    return data_dir / "labels.jsonl"


@pytest.fixture
def run_dir(mock, make_cfg, tmp_path, labels):
    """Both test questions through four formats at N = 1 and 3, graded by the mock judge."""
    return graded_run(mock, make_cfg(formats=FORMATS, n_values=[1, 3]), tmp_path / "runs" / "multi", labels)


@pytest.fixture
def rows(run_dir, labels):
    return analyze.load_rows([run_dir], labels)


def test_recall():
    assert analyze.recall({"a", "b", "x"}, {"a", "b", "c", "d"}) == 0.5
    assert analyze.recall({"a"}, set()) is None


def test_mcnemar_exact_p_values():
    assert analyze.mcnemar(0, 0) == 1.0
    assert analyze.mcnemar(5, 0) == pytest.approx(2 / 32)
    assert analyze.mcnemar(2, 10) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_bootstrap_is_repeatable_and_brackets_the_mean():
    items = [[1], [0], [1], [1], [0], [1]]
    stat = lambda sample: sum(v for q in sample for v in q) / len(sample)
    low, high = analyze.bootstrap(items, stat)
    assert (low, high) == analyze.bootstrap(items, stat) and low <= 4 / 6 <= high
    assert analyze.bootstrap([[1]], stat) == (None, None)


def test_one_row_per_result_with_grades_and_evidence_recall(rows):
    assert len(rows) == 2 * len(FORMATS) * 2
    q1 = next(r for r in rows if (r["qid"], r["format"], r["n"]) == ("q1", "quotes", 1))
    assert q1["correct"] is True and q1["exact_answer"] == "ZEBRA-42" and q1["gold_recall_delivered"] == 1.0
    assert q1["write_output"] == 0 and q1["return_tokens"] == q1["return_per_subagent"] > 0
    for r in rows:
        assert r["recall_delivered"] <= r["recall_opened"] <= r["recall_retrieved"]
        assert r["total_output"] == sum(r[f"{b}_output"] for b in analyze.BUCKETS)


def test_configs_report_accuracy_tokens_per_correct_and_recall(rows):
    configs = {(c["format"], c["n"]): c for c in analyze.summarize_configs(rows)}
    assert list(configs)[:4] == [("short", 1), ("medium", 1), ("quotes", 1), ("raw", 1)]
    medium = configs[("medium", 3)]
    assert medium["results"] == medium["graded"] == 2 and medium["accuracy"] == 50.0  # q1 right, q2 wrong
    group = [r for r in rows if (r["format"], r["n"]) == ("medium", 3)]
    assert medium["output_per_correct"] == sum(r["total_output"] for r in group)
    assert medium["write_output"] > 0 == configs[("quotes", 3)]["write_output"]
    assert 0 < medium["recall_delivered"] <= 1


def test_paired_comparison_against_the_medium_summary(rows):
    by_config = {(p["format"], p["n"]): p for p in analyze.paired(rows)}
    assert set(by_config) == {(f, n) for f in ("short", "quotes", "raw") for n in (1, 3)}
    quotes = by_config[("quotes", 3)]
    assert quotes["pairs"] == 2 and quotes["accuracy_diff_points"] == 0 and quotes["mcnemar_p"] == 1.0
    assert quotes["output_change_pct"] < -20 and quotes["meets_target"] is True  # no writing tokens


def test_paired_matches_questions_and_seeds():
    def row(fmt, qid, seed, correct, output):
        return {"mode": "multi", "format": fmt, "n": 3, "qid": qid, "seed": seed, "correct": correct,
                "total_output": output, "total_input": 10}

    rows = [row("medium", q, s, True, 100) for q in "abcd" for s in (0, 1)]
    rows += [row("quotes", q, s, q != "a", 50) for q in "abcd" for s in (0, 1)]
    rows.append(row("quotes", "e", 0, True, 1))  # no reference to pair with
    (p,) = analyze.paired(rows)
    assert p["pairs"] == 8 and p["losses"] == 2 and p["wins"] == 0
    assert p["accuracy_diff_points"] == -25.0 and p["mcnemar_p"] == 0.5
    assert p["output_change_pct"] == -50.0 and p["input_change_pct"] == 0.0 and p["meets_target"] is False


def test_main_writes_tables_and_plots_including_the_single_agent_baseline(mock, make_cfg, run_dir, labels, tmp_path):
    single = graded_run(mock, make_cfg(mode="single"), tmp_path / "runs" / "single", labels)
    out = tmp_path / "analysis"
    analyze.main([str(run_dir), str(single), "--labels", str(labels), "--out", str(out)])
    with open(out / "configs.csv") as f:
        configs = list(csv.DictReader(f))
    assert configs[0]["format"] == "single" and configs[0]["recall_delivered"] == ""
    assert len(configs) == 1 + len(FORMATS) * 2
    assert (out / "rows.csv").exists() and (out / "paired.csv").exists()
    pytest.importorskip("matplotlib")
    assert (out / "accuracy_vs_tokens.png").stat().st_size > 0 and (out / "tokens_by_phase.png").exists()


def test_without_grades_accuracy_stays_empty(mock, make_cfg, tmp_path, labels):
    run_dir = tmp_path / "runs" / "ungraded"
    run_grid(mock, make_cfg(formats=["quotes"], n_values=[1]), run_dir)
    (config,) = analyze.summarize_configs(analyze.load_rows([run_dir], labels))
    assert config["accuracy"] is None and config["graded"] == 0 and config["output_per_correct"] is None
