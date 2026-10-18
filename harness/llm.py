"""One client for every OpenAI-compatible endpoint: requests, usage, retries and concurrency."""

import asyncio
import os
import time
from typing import NamedTuple

from openai import AsyncOpenAI

from . import tokens
from .config import Endpoint
from .records import CallRecord


class UsageMissing(RuntimeError):
    """The server sent no usage report, so the call's tokens cannot be counted."""


class Completion(NamedTuple):
    text: str
    reasoning: str
    record: CallRecord


def request_body(ep: Endpoint, thinking: bool, max_tokens: int, seed: int | None) -> dict:
    """Every request field except model and messages. It is sent as extra_body, so
    provider-specific keys pass through untouched; nested extra_body tables in the config are flattened."""
    body = {}
    for part in (ep.sampling, ep.thinking_on if thinking else ep.thinking_off):
        part = dict(part)
        body.update(part.pop("extra_body", {}))
        body.update(part)
    body[ep.max_tokens_param] = max_tokens
    if seed is not None:
        body["seed"] = seed
    return body


def reasoning_text(message) -> str:
    """Thinking text, which servers return under different names (vLLM: reasoning_content)."""
    for field in ("reasoning_content", "reasoning"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value:
            return value
    return ""


def normalize_usage(usage, reasoning: str, tokenizer: str) -> dict:
    if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        raise UsageMissing("the response has no usage report")
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", None)
    reported = getattr(usage.completion_tokens_details, "reasoning_tokens", None)
    if reported is not None and not (reported == 0 and reasoning):
        reasoning_tokens, source = reported, "reported"
    elif reasoning:
        reasoning_tokens, source = tokens.count(tokenizer, reasoning), "estimated"
    else:
        reasoning_tokens, source = None, "not_reported"
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning_tokens,
        "usage_source": {"cached": "reported" if cached is not None else "not_reported", "reasoning": source},
    }


class ModelClient:
    def __init__(self, endpoints: dict[str, Endpoint], http_client=None):
        self.endpoints = endpoints
        self._http_client = http_client
        self._clients: dict[str, AsyncOpenAI] = {}
        self._slots = {name: asyncio.Semaphore(ep.max_concurrency) for name, ep in endpoints.items()}

    def _client(self, name: str) -> AsyncOpenAI:
        if name not in self._clients:
            ep = self.endpoints[name]
            if ep.api_key_env and not os.environ.get(ep.api_key_env):
                raise RuntimeError(f"endpoint {name}: set the {ep.api_key_env} environment variable")
            self._clients[name] = AsyncOpenAI(
                base_url=ep.base_url,
                api_key=os.environ.get(ep.api_key_env) or "not-needed",
                timeout=ep.timeout_s,
                # The SDK retries 408, 409, 429, 5xx and connection errors with backoff and honors Retry-After.
                max_retries=ep.max_retries,
                http_client=self._http_client,
            )
        return self._clients[name]

    async def chat(self, endpoint: str, messages: list[dict], tag: dict, *, thinking: bool,
                   max_tokens: int, seed: int | None = None) -> Completion:
        """One chat call. tag holds role, phase and optionally round, subagent and step."""
        ep = self.endpoints[endpoint]
        client = self._client(endpoint)
        async with self._slots[endpoint]:
            start = time.time()
            response = await client.chat.completions.create(
                model=ep.model, messages=messages, extra_body=request_body(ep, thinking, max_tokens, seed)
            )
            end = time.time()
        choice = response.choices[0]
        text = choice.message.content or ""
        reasoning = reasoning_text(choice.message)
        record = CallRecord(
            **tag,
            endpoint=endpoint,
            model=ep.model,
            messages_sent=len(messages),
            **normalize_usage(response.usage, reasoning, ep.tokenizer),
            finish_reason=choice.finish_reason,
            t_start=start,
            t_end=end,
            response_text=text,
            reasoning_text=reasoning,
        )
        return Completion(text, reasoning, record)
