import asyncio
import json

import httpx2
import pytest

from harness.config import RunConfig
from harness.run import load_questions, run, save_config
from harness.search import Corpus, MemoryIndex, SearchSession
from mock_llm import MockLLM

# Each paragraph is padded past the 80-word passage minimum, so every paragraph becomes its own passage.
FILLER = (
    "The rest of this paragraph is background text that adds length without adding facts, describing ordinary "
    "weather, quiet streets, small shops, market days and the slow change of seasons in the countryside. "
) * 3

DOCS = {
    "d01": ("Ponte de Ferro do Tamel", [
        "The Ponte de Ferro do Tamel is an iron bridge over the river Tamel, completed in 1887. "
        "Its heritage registry code is ZEBRA-42.",
        "The bridge was designed by the engineer Joaquim Seabra, who was born in Lisbon.",
        "After the great flood of 1923 the bridge was restored and its deck was replaced.",
    ]),
    "d02": ("Joaquim Seabra", [
        "Joaquim Seabra was a Portuguese civil engineer born in Lisbon in 1841.",
        "Seabra trained at the army school of military engineering before designing railway works.",
    ]),
    "d03": ("River Tamel", [
        "The river Tamel rises in the northern hills and flows west to the sea.",
        "The Tamel floods often; the worst flood on record struck in 1923.",
    ]),
    "d04": ("Iron bridges of the 1880s", [
        "Many iron bridges were completed in the 1880s, several of them in 1887.",
    ]),
    "d05": ("National heritage registry", [
        "The national heritage registry assigns a code to every protected structure.",
    ]),
    "d06": ("Maria Lopes", [
        "Maria Lopes was an engineer born in Porto who designed steel bridges.",
    ]),
    "d07": ("Stone bridge of Arvel", [
        "The stone bridge of Arvel crosses the river Arvel and was completed in 1702.",
    ]),
    "d08": ("Survey of the northern valleys", [f"Survey section {i} covers one valley of the north." for i in range(10)]),
}

QUESTIONS = [
    {
        "qid": "q1",
        "question": "What heritage registry code was given to the iron bridge over the river Tamel that was "
                    "completed in 1887 and designed by an engineer born in Lisbon?",
        "answer": "ZEBRA-42",
        "clues": [
            "an iron bridge completed in 1887",
            "the bridge crosses the river Tamel",
            "its designer was an engineer born in Lisbon",
            "the designer trained at a school of military engineering",
            "the bridge was restored after a flood in 1923",
            "the bridge has a heritage registry code",
        ],
    },
    {
        "qid": "q2",
        "question": "Which engineer born in Lisbon designed the iron bridge over the river Tamel?",
        "answer": "Joaquim Seabra",
        "clues": ["an engineer born in Lisbon", "an iron bridge over the river Tamel"],
    },
]


LABELS = [
    {"qid": "q1", "answer": "ZEBRA-42", "evidence_docids": ["d01", "d02"], "gold_docids": ["d01"]},
    {"qid": "q2", "answer": "Joaquim Seabra", "evidence_docids": ["d02", "d01"], "gold_docids": ["d02"]},
]


def run_grid(mock, cfg, run_dir, questions=None, retry_failed=False):
    """Run cfg's grid through the mock server; by default only the first question."""
    save_config(cfg, run_dir)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(mock))
    asyncio.run(run(cfg, questions or load_questions(cfg)[:1], run_dir, retry_failed, http_client=http))


def mock_endpoint(**overrides) -> dict:
    return {
        "base_url": "http://mock/v1", "model": "mock-model", "tokenizer": "words", "max_retries": 2,
        "thinking_on": {"reasoning_effort": "medium"}, "thinking_off": {"reasoning_effort": "low"}, **overrides,
    }


@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    with open(path / "corpus.jsonl", "w") as f:
        for docid, (title, paragraphs) in DOCS.items():
            text = "\n\n".join(f"{p} {FILLER}" for p in paragraphs)
            f.write(json.dumps({"docid": docid, "title": title, "text": text}) + "\n")
    with open(path / "questions.jsonl", "w") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q) + "\n")
    with open(path / "labels.jsonl", "w") as f:
        for label in LABELS:
            f.write(json.dumps(label) + "\n")
    return path


@pytest.fixture
def corpus(data_dir):
    return Corpus.load(data_dir / "corpus.jsonl")


@pytest.fixture
def session(corpus):
    return SearchSession(corpus, MemoryIndex(corpus), k=5, snippet_tokens=30, open_tokens=350, tokenizer="words")


@pytest.fixture
def mock():
    return MockLLM()


@pytest.fixture
def make_cfg(data_dir):
    def make(**overrides) -> RunConfig:
        data = {
            "name": "test",
            "endpoints": {"orch": mock_endpoint(), "sub": mock_endpoint()},
            "roles": {"orchestrator": "orch", "subagent": "sub", "single": "orch"},
            "questions": str(data_dir / "questions.jsonl"),
            "corpus": str(data_dir / "corpus.jsonl"),
            "limits": {"max_steps": 6},
        }
        data.update(overrides)
        return RunConfig.model_validate(data)

    return make


@pytest.fixture
def questions():
    return [dict(q) for q in QUESTIONS]
