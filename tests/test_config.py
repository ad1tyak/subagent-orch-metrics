from pathlib import Path

import pytest
from pydantic import ValidationError

from harness import prompts
from harness.config import load_config, stage1_key

REPO = Path(__file__).resolve().parent.parent


def test_shipped_configs_load(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config("configs/pilot.toml")
    assert cfg.roles.orchestrator in cfg.endpoints and cfg.roles.subagent in cfg.endpoints
    assert len(cfg.formats) * len(cfg.n_values) * len(cfg.seeds) == 18
    assert cfg.budgets.medium == 1000


def test_endpoints_can_be_inline(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        'name = "x"\nquestions = "q.jsonl"\ncorpus = "c.jsonl"\n'
        '[roles]\norchestrator = "a"\nsubagent = "a"\n'
        '[endpoints.a]\nbase_url = "http://h/v1"\nmodel = "m"\ntokenizer = "words"\n'
        '[budgets]\nshort = 150\n'
    )
    cfg = load_config(path)
    assert cfg.endpoints["a"].model == "m"
    assert (cfg.budgets.short, cfg.budgets.long) == (150, 3000)


def test_roles_must_name_endpoints(make_cfg):
    with pytest.raises(ValidationError, match="roles.subagent"):
        make_cfg(roles={"orchestrator": "orch", "subagent": "missing"})


def test_typos_are_rejected(make_cfg):
    with pytest.raises(ValidationError, match="max_step"):
        make_cfg(limits={"max_step": 3})


def test_stage1_key_tracks_what_changes_the_searches(make_cfg, questions, monkeypatch):
    cfg, q = make_cfg(), questions[0]
    base = stage1_key(cfg, q, 3, 0)
    assert stage1_key(cfg, q, 3, 0) == base
    assert stage1_key(cfg, q, 6, 0) != base
    assert stage1_key(cfg, q, 3, 1) != base
    assert stage1_key(cfg, {**q, "clues": q["clues"][:3]}, 3, 0) != base
    other_model = make_cfg(endpoints={"orch": cfg.endpoints["orch"].model_dump(), "sub": {
        **cfg.endpoints["sub"].model_dump(), "model": "another-model"}})
    assert stage1_key(other_model, q, 3, 0) != base
    hotter = make_cfg(endpoints={"orch": cfg.endpoints["orch"].model_dump(), "sub": {
        **cfg.endpoints["sub"].model_dump(), "sampling": {"temperature": 0.2}}})
    assert stage1_key(hotter, q, 3, 0) != base
    assert stage1_key(make_cfg(limits={"max_steps": 6, "search_k": 10}), q, 3, 0) != base
    monkeypatch.setattr(prompts, "VERSION", "changed")
    assert stage1_key(cfg, q, 3, 0) != base


def test_stage1_key_ignores_stage2_settings(make_cfg, questions):
    q = questions[0]
    base = stage1_key(make_cfg(), q, 3, 0)
    assert stage1_key(make_cfg(formats=["raw"], budgets={"raw": 10}), q, 3, 0) == base
    assert stage1_key(make_cfg(limits={"max_steps": 6, "max_extra_rounds": 2}), q, 3, 0) == base
    assert stage1_key(make_cfg(max_questions_in_flight=1), q, 3, 0) == base
