import asyncio

import openai
import pytest
from openai.types import CompletionUsage

from harness.config import Endpoint
from harness.llm import UsageMissing, normalize_usage, request_body
from conftest import mock_endpoint
from mock_llm import MockLLM, mock_client

TAG = {"role": "subagent", "phase": "search", "round": 1, "subagent": 2, "step": 0}
HELLO = [{"role": "user", "content": "say hi"}]


def usage(**fields):
    return CompletionUsage.model_validate({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140, **fields})


def test_openai_shape_reports_everything():
    out = normalize_usage(usage(prompt_tokens_details={"cached_tokens": 64},
                                completion_tokens_details={"reasoning_tokens": 30}), "", "words")
    assert out["input_tokens"] == 100 and out["output_tokens"] == 40
    assert (out["cached_input_tokens"], out["reasoning_tokens"]) == (64, 30)
    assert out["usage_source"] == {"cached": "reported", "reasoning": "reported"}


def test_vllm_shape_without_details_estimates_reasoning_from_text():
    out = normalize_usage(usage(), "one two three four", "words")
    assert out["cached_input_tokens"] is None
    assert out["reasoning_tokens"] == 4
    assert out["usage_source"] == {"cached": "not_reported", "reasoning": "estimated"}


def test_vllm_shape_with_prompt_details():
    out = normalize_usage(usage(prompt_tokens_details={"cached_tokens": 0}), "", "words")
    assert out["cached_input_tokens"] == 0 and out["usage_source"]["cached"] == "reported"


def test_provider_with_no_reasoning_fields_or_text():
    out = normalize_usage(usage(), "", "words")
    assert out["reasoning_tokens"] is None and out["usage_source"]["reasoning"] == "not_reported"


def test_reported_zero_reasoning_with_text_is_estimated():
    out = normalize_usage(usage(completion_tokens_details={"reasoning_tokens": 0}), "a b c", "words")
    assert (out["reasoning_tokens"], out["usage_source"]["reasoning"]) == (3, "estimated")


def test_missing_usage_fails_loudly():
    with pytest.raises(UsageMissing):
        normalize_usage(None, "", "words")


def test_request_body_merges_config_and_flattens_extra_body():
    ep = Endpoint.model_validate(mock_endpoint(
        max_tokens_param="max_completion_tokens",
        sampling={"temperature": 0.6, "extra_body": {"top_k": 20}},
        thinking_on={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        thinking_off={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ))
    assert request_body(ep, True, 500, 7) == {
        "temperature": 0.6, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True},
        "max_completion_tokens": 500, "seed": 7,
    }
    assert request_body(ep, False, 500, None)["chat_template_kwargs"] == {"enable_thinking": False}
    assert "seed" not in request_body(ep, False, 500, None)


def endpoints(**overrides):
    return {"e": Endpoint.model_validate(mock_endpoint(**overrides))}


def test_chat_returns_text_reasoning_and_a_tagged_record(mock):
    client = mock_client(mock, endpoints())
    mock.script["other"].append("hello there")
    reply = asyncio.run(client.chat("e", HELLO, TAG, thinking=True, max_tokens=50, seed=3))
    assert reply.text == "hello there" and reply.reasoning
    record = reply.record
    assert (record.role, record.phase, record.subagent, record.step) == ("subagent", "search", 2, 0)
    assert (record.endpoint, record.model, record.messages_sent) == ("e", "mock-model", 1)
    assert record.input_tokens == 2 and record.reasoning_tokens == 7
    assert record.t_end >= record.t_start
    body = mock.requests[0][1]
    assert body["reasoning_effort"] == "medium" and body["max_tokens"] == 50 and body["seed"] == 3


def test_rate_limit_then_success_retries(mock):
    mock.script["other"].extend([429, "fine"])
    reply = asyncio.run(mock_client(mock, endpoints()).chat("e", HELLO, TAG, thinking=False, max_tokens=5))
    assert reply.text == "fine" and len(mock.requests) == 2


def test_persistent_server_error_raises_after_retries(mock):
    mock.script["other"].extend([500] * 10)
    with pytest.raises(openai.InternalServerError):
        asyncio.run(mock_client(mock, endpoints()).chat("e", HELLO, TAG, thinking=False, max_tokens=5))
    assert len(mock.requests) == 3  # one attempt plus max_retries=2


def test_missing_usage_from_server_raises():
    mock = MockLLM(usage=False)
    with pytest.raises(UsageMissing):
        asyncio.run(mock_client(mock, endpoints()).chat("e", HELLO, TAG, thinking=False, max_tokens=5))


def test_requests_in_flight_never_exceed_max_concurrency():
    mock = MockLLM(delay=0.02)
    client = mock_client(mock, endpoints(max_concurrency=2))

    async def many():
        await asyncio.gather(*(client.chat("e", HELLO, TAG, thinking=False, max_tokens=5) for _ in range(6)))

    asyncio.run(many())
    assert mock.max_active == 2


def test_missing_api_key_env_is_an_error(mock, monkeypatch):
    monkeypatch.delenv("HARNESS_TEST_KEY", raising=False)
    client = mock_client(mock, endpoints(api_key_env="HARNESS_TEST_KEY"))
    with pytest.raises(RuntimeError, match="HARNESS_TEST_KEY"):
        asyncio.run(client.chat("e", HELLO, TAG, thinking=False, max_tokens=5))
