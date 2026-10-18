import json

from harness import calibrate
from harness.config import FORMATS
from harness.pipeline import results
from harness.run import load_questions
from conftest import run_grid


def test_percentiles_and_rounding():
    assert calibrate.percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert calibrate.percentile([5, 1, 3, 2, 4], 1.0) == 5
    assert calibrate.round_up(1201) == 1300 and calibrate.round_up(1200) == 1200


def test_released_runs_report_searches_output_and_answers(tmp_path):
    hits = [{"docid": d, "score": 1.0, "snippet": "five words of snippet text"} for d in ("d1", "d2")]
    run = {"query_id": "1", "status": "completed", "result": [
        {"type": "tool_call", "tool_name": "search", "output": json.dumps(hits)},
        {"type": "tool_call", "tool_name": "search", "output": json.dumps(hits[:1])},
        {"type": "output_text", "output": "Exact Answer: 1348"},
    ]}
    path = tmp_path / "bm25-gpt5.jsonl"
    path.write_text(json.dumps(run) + "\n" + json.dumps({**run, "query_id": "2", "status": "incomplete"}) + "\n")
    lines = calibrate.released_report(path, "words").splitlines()
    assert "2 questions, 1 completed" in lines[0]
    rows = {line[:36].strip(): line[36:].split() for line in lines[2:]}
    assert rows["searches"][:2] == ["2", "2.0"]
    assert rows["search output tokens"][1] == "15.0"  # three snippets of five words
    assert rows["distinct documents retrieved"][1] == "2.0" and rows["final answer tokens"][1] == "3.0"


def test_run_report_measures_uncapped_sizes_and_reports(mock, make_cfg, tmp_path):
    cfg = make_cfg(formats=list(FORMATS), n_values=[1, 3])
    run_dir = tmp_path / "runs" / "calibrate"
    run_grid(mock, cfg, run_dir, questions=load_questions(cfg))
    report = calibrate.run_report(run_dir)
    subagents = {(r.qid, r.n, r.seed, t.subagent): t for _, r in results(run_dir) for t in r.rounds[0].trajectories}
    assert f"{len(subagents)} first-round subagents (tokens: words)" in report
    seen = [calibrate._passages("words", t, t.seen) for t in subagents.values()]
    assert f"raw {calibrate.round_up(calibrate.percentile(seen, 0.9))} (now {cfg.budgets.raw})" in report
    for fmt in FORMATS:
        assert f"{fmt} report content (cap {getattr(cfg.budgets, fmt)})" in report
    assert "medium writer visible output" in report and "total return per question quotes N=3" in report
    assert "raw writer" not in report


def test_main_points_to_the_download_when_released_runs_are_missing(tmp_path, capsys):
    calibrate.main(["--released", str(tmp_path / "missing")])
    assert "python -m harness.data --released" in capsys.readouterr().out
