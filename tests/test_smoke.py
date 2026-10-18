import asyncio
import os
import tomllib
from pathlib import Path

import pytest

from harness.config import Endpoint
from harness.llm import ModelClient
from harness.smoke import smoke
from conftest import mock_endpoint
from mock_llm import MockLLM, mock_client


def test_smoke_reports_what_the_endpoint_sends():
    report = asyncio.run(smoke(mock_client(MockLLM(), {"e": Endpoint.model_validate(mock_endpoint())}), "e"))
    assert report["plain_call_tokens"]["input"] > 0 and report["cached_tokens_on_repeat"] > 0
    assert report["reasoning_thinking_on"]["source"] == "reported"
    assert report["seed_repeatable"] is True and report["template_overhead_tokens"] == 0


def test_smoke_shows_a_server_that_reports_little():
    mock = MockLLM(cached=False, reasoning="text")
    report = asyncio.run(smoke(mock_client(mock, {"e": Endpoint.model_validate(mock_endpoint())}), "e"))
    assert report["cached_tokens_on_repeat"] is None
    assert report["reasoning_thinking_on"]["source"] == "estimated"


@pytest.mark.live
def test_live_endpoint_smoke():
    """pytest -m live, with HARNESS_ENDPOINTS=configs/endpoints.toml HARNESS_ENDPOINT=<name>. Spends tokens."""
    path, name = os.environ.get("HARNESS_ENDPOINTS"), os.environ.get("HARNESS_ENDPOINT")
    if not (path and name):
        pytest.skip("set HARNESS_ENDPOINTS and HARNESS_ENDPOINT")
    tables = tomllib.loads(Path(path).read_text())["endpoints"]
    report = asyncio.run(smoke(ModelClient({name: Endpoint.model_validate(tables[name])}), name))
    assert report["plain_call_tokens"]["input"] > 0 and report["plain_call_tokens"]["output"] > 0
