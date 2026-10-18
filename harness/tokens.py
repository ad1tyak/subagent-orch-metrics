"""Token counts with an endpoint's tokenizer.

A tokenizer spec is "hf:<model repo>" (Hugging Face tokenizers), "tiktoken:<encoding>",
or "words" (whitespace-separated words, for tests and rough estimates).
"""

import re
from functools import lru_cache

_WORD = re.compile(r"\S+")


@lru_cache(maxsize=None)
def _codec(spec: str):
    kind, _, name = spec.partition(":")
    if kind == "hf":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(name)
        return (lambda text: tok.encode(text, add_special_tokens=False).ids), tok.decode
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return enc.encode_ordinary, enc.decode
    if spec == "words":
        return None
    raise ValueError(f"unknown tokenizer spec {spec!r}")


def count(spec: str, text: str) -> int:
    codec = _codec(spec)
    return len(_WORD.findall(text)) if codec is None else len(codec[0](text))


def truncate(spec: str, text: str, max_tokens: int) -> str:
    """The longest prefix of text that fits in max_tokens."""
    if max_tokens <= 0:
        return ""
    codec = _codec(spec)
    if codec is None:
        words = list(_WORD.finditer(text))
        return text if len(words) <= max_tokens else text[: words[max_tokens - 1].end()]
    ids = codec[0](text)
    return text if len(ids) <= max_tokens else codec[1](ids[:max_tokens])
