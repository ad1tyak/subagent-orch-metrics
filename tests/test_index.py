"""The official BM25 index, checked against the searches in the authors' released runs.

    JAVA_HOME=<Java 21 home> .venv/bin/pytest -m index

Needs the bm25 extra, data/bm25 and data/released_runs (python -m harness.data --released).
"""

import json
import random
from pathlib import Path

import pytest

pytestmark = pytest.mark.index
INDEX, RELEASED = Path("data/bm25"), Path("data/released_runs/bm25-gpt5.jsonl")


def released_searches():
    searches = []
    with open(RELEASED, encoding="utf-8") as f:
        for line in f:
            for item in json.loads(line)["result"]:
                if item["type"] == "tool_call" and item["tool_name"] == "search":
                    try:
                        query, hits = json.loads(item["arguments"])["query"], json.loads(item["output"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    searches.append((query, [hit["docid"] for hit in hits]))
    return searches


def test_our_bm25_returns_the_released_runs_top_five():
    pytest.importorskip("pyserini")
    if not (INDEX.exists() and RELEASED.exists()):
        pytest.skip("needs data/bm25 and data/released_runs; run python -m harness.data --released")
    from harness.search import PyseriniIndex

    index = PyseriniIndex(str(INDEX))
    sample = random.Random(0).sample(released_searches(), 200)
    mismatches = [(query, theirs) for query, theirs in sample if index.search(query, 5) != theirs]
    assert mismatches == []
