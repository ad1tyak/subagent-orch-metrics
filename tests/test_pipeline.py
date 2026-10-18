import gzip
import json

import pytest

from harness import pipeline
from harness.accounting import BUCKETS, PHASES
from harness.check import check
from harness.config import FORMATS
from harness.pipeline import read_gz, write_atomic
from harness.prompts import TEMPLATES
from harness.records import QuestionResult
from harness.run import save_config
from conftest import run_grid
from mock_llm import prefix


def load(path) -> QuestionResult:
    return QuestionResult.model_validate_json(read_gz(path))


def results(run_dir):
    return {p.relative_to(run_dir / "results").as_posix(): load(p) for p in (run_dir / "results").rglob("*.json.gz")}


@pytest.fixture
def grid(mock, make_cfg, tmp_path):
    """One question through all 18 configurations."""
    cfg, run_dir = make_cfg(), tmp_path / "runs" / "test"
    run_grid(mock, cfg, run_dir)
    return cfg, run_dir, results(run_dir)


def test_every_configuration_produces_a_result_and_passes_check(grid):
    cfg, run_dir, found = grid
    assert sorted(found) == sorted(f"n{n}/s0/{fmt}/q1.json.gz" for n in (1, 3, 6) for fmt in FORMATS)
    problems, stats = check(run_dir)
    assert problems == [] and stats["results"] == 18 and stats["failed"] == 0


def test_with_one_subagent_every_format_finds_the_answer(grid):
    _, _, found = grid
    for fmt in FORMATS:
        result = found[f"n1/s0/{fmt}/q1.json.gz"]
        assert result.answer["exact_answer"] == "ZEBRA-42", fmt
        assert "Exact Answer: ZEBRA-42" in result.answer["response"]


def test_stage1_runs_once_per_question_n_and_seed(mock, grid):
    _, run_dir, found = grid
    assert mock.count("plan") == 2  # n=3 and n=6; n=1 needs no plan
    assert len(list((run_dir / "stage1").glob("q1-*.json.gz"))) == 3
    first_round = sum(len(t.calls) for n in (1, 3, 6) for t in found[f"n{n}/s0/quotes/q1.json.gz"].rounds[0].trajectories)
    later_rounds = sum(len(t.calls) for r in found.values() for rnd in r.rounds[1:] for t in rnd.trajectories)
    assert mock.count("search") == first_round + later_rounds


def test_search_never_sees_a_format(mock, grid):
    writing_prompts = (prefix("write_summary"), prefix("write_claims"), prefix("claims_repair"))
    for body in mock.bodies("search"):
        assert not any(m["content"].startswith(writing_prompts) for m in body["messages"])


def test_formats_differ_only_in_the_return(grid):
    _, _, found = grid
    for n in (1, 3, 6):
        group = [found[f"n{n}/s0/{fmt}/q1.json.gz"] for fmt in FORMATS]
        assert len({r.stage1_key for r in group}) == 1
        assert len({json.dumps([t.model_dump() for t in r.rounds[0].trajectories]) for r in group}) == 1
        assert {r.rounds[0].reports[0].format for r in group} == set(FORMATS)


def test_written_formats_spend_writing_tokens_and_copied_ones_do_not(grid):
    _, _, found = grid
    for name, result in found.items():
        written = result.tokens["write"]["output"]
        assert (written > 0) == (result.format in ("short", "medium", "long", "claims")), name


def test_a_finished_run_resumes_without_calls(mock, make_cfg, grid):
    cfg, run_dir, _ = grid
    before = len(mock.requests)
    run_grid(mock, cfg, run_dir)
    assert len(mock.requests) == before


def test_a_changed_config_cannot_reuse_a_run(make_cfg, grid):
    _, run_dir, _ = grid
    with pytest.raises(SystemExit, match="different config"):
        save_config(make_cfg(budgets={"short": 10}), run_dir)
    save_config(make_cfg(max_questions_in_flight=1), run_dir)  # operational settings are fine


def test_redelegation_lands_in_extra_rounds(mock, make_cfg, tmp_path):
    cfg, run_dir = make_cfg(formats=["quotes"], n_values=[3]), tmp_path / "run"
    mock.script["decide"].append({"decision": "redelegate", "subtasks": [{"task": "Find the bridge's registry code"}]})
    run_grid(mock, cfg, run_dir)
    result = load(run_dir / "results/n3/s0/quotes/q1.json.gz")
    assert [r.round for r in result.rounds] == [1, 2] and len(result.rounds[1].trajectories) == 1
    assert TEMPLATES["decide_final"] in mock.bodies("decide")[1]["messages"][-1]["content"]
    calls = result.all_calls()
    round2 = [c for c in calls if c.round == 2]
    assert result.tokens["extra_rounds"]["calls"] == len(round2) > 0
    assert sum(result.tokens[p]["calls"] for p in PHASES) == len(calls) - len(round2)
    for key in ("input", "output", "cached", "reasoning"):
        assert sum(result.tokens[b][key] for b in BUCKETS) == result.tokens["total"][key] == sum(
            getattr(c, {"input": "input_tokens", "output": "output_tokens", "cached": "cached_input_tokens",
                        "reasoning": "reasoning_tokens"}[key]) for c in calls)
    assert check(run_dir)[0] == []


def test_failures_are_recorded_skipped_and_retried(mock, make_cfg, tmp_path):
    cfg, run_dir = make_cfg(formats=["quotes"], n_values=[1]), tmp_path / "run"
    mock.script["decide"].extend([500] * 3)  # one decide call: an attempt plus two retries
    run_grid(mock, cfg, run_dir)
    error = run_dir / "results/n1/s0/quotes/q1.error.json"
    assert error.exists() and "InternalServerError" in json.loads(error.read_text())["error"]
    assert not (run_dir / "results/n1/s0/quotes/q1.json.gz").exists()
    assert any("rerun with --retry-failed" in p for p in check(run_dir)[0])

    before = len(mock.requests)
    run_grid(mock, cfg, run_dir)
    assert len(mock.requests) == before  # failed questions are skipped unless retried

    run_grid(mock, cfg, run_dir, retry_failed=True)
    assert (run_dir / "results/n1/s0/quotes/q1.json.gz").exists() and not error.exists()
    assert check(run_dir)[0] == []


def test_an_interrupted_write_leaves_no_result(tmp_path, monkeypatch):
    def crash(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        write_atomic(tmp_path / "q1.json.gz", b"data")
    assert not (tmp_path / "q1.json.gz").exists()


def test_single_agent_baseline(mock, make_cfg, tmp_path):
    cfg, run_dir = make_cfg(mode="single"), tmp_path / "run"
    run_grid(mock, cfg, run_dir)
    result = load(run_dir / "results/single/s0/q1.json.gz")
    assert result.answer["exact_answer"] == "ZEBRA-42" and result.plan is None and result.rounds[0].reports == []
    assert result.tokens["search"]["calls"] == result.tokens["total"]["calls"] > 0
    assert check(run_dir)[0] == []


def test_check_catches_tampered_token_totals(grid):
    _, run_dir, _ = grid
    path = run_dir / "results/n3/s0/raw/q1.json.gz"
    data = json.loads(read_gz(path))
    data["tokens"]["total"]["input"] += 1
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    problems, _ = check(run_dir)
    assert any("n3/s0/raw/q1.json.gz: token totals" in p for p in problems)
