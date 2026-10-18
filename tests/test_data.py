import base64
import json

import pytest

from harness import data


def encrypt(text: str) -> str:
    raw = text.encode()
    key = data._KEY * (len(raw) // len(data._KEY) + 1)
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()


QUERY_ROWS = [
    {"query_id": qid, "query": encrypt(question), "answer": encrypt(answer),
     "evidence_docs": [{"docid": encrypt(d)} for d in evidence], "gold_docs": [{"docid": encrypt(evidence[0])}]}
    for qid, question, answer, evidence in [
        ("10", "Which bridge, built before 1900, ...?", "Ponte de Ferro", ["d1", "d2"]),
        ("9", "Who wrote the ... in 1921?", "Virginia Woolf", ["d3"]),
    ]
]
CORPUS_ROWS = [
    {"docid": "d1", "text": "---\ntitle: Galata Tower - Wikipedia\ndate: 2003-04-24\n---\nname: Galata Tower", "url": "u1"},
    {"docid": "d2", "text": "No front matter here.", "url": "u2"},
]


@pytest.fixture
def hub(monkeypatch):
    calls = []

    def rows(repo, columns):
        calls.append(repo)
        yield from (QUERY_ROWS if repo == data.QUERIES else CORPUS_ROWS)

    monkeypatch.setattr(data, "rows", rows)
    return calls


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_decrypt_reverses_the_browsecomp_obfuscation():
    text = "Ünïcode and a string longer than one 32-byte key " * 3
    assert data.decrypt(encrypt(text)) == text


def test_title_comes_from_front_matter_or_the_first_line():
    assert data.title(CORPUS_ROWS[0]["text"]) == "Galata Tower - Wikipedia"
    assert data.title("---\ndate: 2010-04-13\n---\nRegistration No. 333\n\nPROSPECTUS") == "Registration No. 333"
    assert data.title("---\n---\nAbhinav Adhikari\nBiotechnologist") == "Abhinav Adhikari"
    assert data.title("---\ntitle: \ndate: \n---\n\nMember login") == "Member login"
    assert data.title("No front matter here.") == "No front matter here."
    assert data.title("") == ""


def test_questions_and_labels_are_split_and_sorted(hub, tmp_path):
    data.fetch_questions(tmp_path)
    questions, labels = read(tmp_path / "questions.jsonl"), read(tmp_path / "labels.jsonl")
    assert questions == [{"qid": "9", "question": "Who wrote the ... in 1921?"},
                         {"qid": "10", "question": "Which bridge, built before 1900, ...?"}]
    assert labels[1] == {"qid": "10", "answer": "Ponte de Ferro", "evidence_docids": ["d1", "d2"], "gold_docids": ["d1"]}
    assert "answer" not in (tmp_path / "questions.jsonl").read_text()


def test_corpus_keeps_text_and_adds_titles(hub, tmp_path):
    data.fetch_corpus(tmp_path)
    docs = read(tmp_path / "corpus.jsonl")
    assert [d["title"] for d in docs] == ["Galata Tower - Wikipedia", "No front matter here."]
    assert docs[0]["text"] == CORPUS_ROWS[0]["text"]


def test_existing_files_are_not_downloaded_again(hub, tmp_path):
    data.fetch_questions(tmp_path)
    data.fetch_corpus(tmp_path)
    data.fetch_questions(tmp_path)
    data.fetch_corpus(tmp_path)
    assert hub == [data.QUERIES, data.CORPUS]


def released_row(qid):
    hits = [{"docid": "d1", "score": 9.5, "snippet": "---\ntitle: Galata Tower\n---\nbuilt in 1348"}]
    return {"query_id": qid, "search_counts": 1, "status": "completed", "result": [
        {"type": "reasoning", "tool_name": None, "arguments": None, "output": [{"text": encrypt("look it up")}]},
        {"type": "tool_call", "tool_name": "search", "arguments": '{"query": "galata tower"}',
         "output": encrypt(json.dumps(hits))},
        {"type": "reasoning", "tool_name": None, "arguments": None, "output": None},
        {"type": "output_text", "tool_name": None, "arguments": None, "output": encrypt("Exact Answer: 1348")},
    ]}


def test_released_runs_are_decrypted_like_the_official_script(monkeypatch, tmp_path):
    calls = []

    def lines(repo, path):
        calls.append(path)
        yield from (json.dumps(released_row(qid)) + "\n" for qid in ("770", "771"))

    monkeypatch.setattr(data, "lines", lines)
    data.fetch_released(tmp_path)
    data.fetch_released(tmp_path)
    assert calls == list(data.RELEASED.values())
    runs = read(tmp_path / "released_runs" / "bm25-gpt5.jsonl")
    assert [r["query_id"] for r in runs] == ["770", "771"]
    reasoning, search, empty, answer = runs[0]["result"]
    assert reasoning["output"] == [{"text": "look it up"}] and empty["output"] is None
    assert search["arguments"] == '{"query": "galata tower"}' and json.loads(search["output"])[0]["docid"] == "d1"
    assert answer["output"] == "Exact Answer: 1348"


def test_an_interrupted_download_leaves_no_file(monkeypatch, tmp_path):
    def broken(repo, columns):
        yield CORPUS_ROWS[0]
        raise ConnectionError("network dropped")

    monkeypatch.setattr(data, "rows", broken)
    with pytest.raises(ConnectionError):
        data.fetch_corpus(tmp_path)
    assert not (tmp_path / "corpus.jsonl").exists()
