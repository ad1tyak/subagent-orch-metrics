"""Reading the JSON that models reply with."""

import json


def first_json(text: str) -> dict:
    """The first JSON object in text, ignoring prose or code fences around it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found")


def confidence(value) -> float | None:
    """A 0-100 confidence from a number or a string like "80%"; None when unreadable."""
    try:
        return max(0.0, min(100.0, float(str(value).strip().rstrip("%"))))
    except ValueError:
        return None
