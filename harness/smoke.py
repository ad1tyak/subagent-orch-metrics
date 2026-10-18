"""Check what an endpoint reports before any run uses it, and save the output with the run.

    python -m harness.smoke configs/endpoints.toml gptoss-20b-vllm > runs/pilot/smoke-gptoss-20b-vllm.json
"""

import argparse
import asyncio
import json

from . import tokens
from .config import load_endpoints
from .llm import ModelClient

_LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 300 + "\nReply with the single word: ok"
_TAG = {"role": "smoke", "phase": "smoke"}


async def smoke(client: ModelClient, name: str) -> dict:
    ep = client.endpoints[name]

    async def ask(text, thinking=False, seed=None, max_tokens=256):
        return await client.chat(name, [{"role": "user", "content": text}], _TAG, thinking=thinking,
                                 max_tokens=max_tokens, seed=seed)

    plain = await ask("Reply with the single word: ready")
    first, repeat = await ask(_LONG_PROMPT), await ask(_LONG_PROMPT)
    question = "What is 17 * 23? Reply with just the number."
    on, off = await ask(question, thinking=True, max_tokens=2048), await ask(question, max_tokens=2048)
    seeded = [await ask("Write one unusual English word.", seed=1234) for _ in range(2)]
    return {
        "endpoint": name,
        "model": ep.model,
        "plain_call_tokens": {"input": plain.record.input_tokens, "output": plain.record.output_tokens},
        "cached_tokens_on_repeat": repeat.record.cached_input_tokens,
        "reasoning_thinking_on": {"tokens": on.record.reasoning_tokens, "source": on.record.usage_source["reasoning"]},
        "reasoning_thinking_off": {"tokens": off.record.reasoning_tokens, "source": off.record.usage_source["reasoning"]},
        "seed_repeatable": seeded[0].text == seeded[1].text,
        "template_overhead_tokens": first.record.input_tokens - tokens.count(ep.tokenizer, _LONG_PROMPT),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoints", help="TOML file with [endpoints.<name>] tables")
    parser.add_argument("endpoint")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(smoke(ModelClient(load_endpoints(args.endpoints)), args.endpoint)), indent=2))


if __name__ == "__main__":
    main()
