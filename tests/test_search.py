import pytest

from harness.search import Corpus, MemoryIndex, bm25, split_passages, terms


def test_short_paragraphs_merge_and_long_ones_split_at_sentences():
    short = "A short paragraph."
    long = " ".join(f"Sentence number {i} has exactly seven words." for i in range(60))  # 420 words
    passages = split_passages(f"{short}\n\n{short}\n\n{long}", min_words=80, max_words=250)
    assert passages[0] == f"{short}\n\n{short}"
    assert all(len(p.split()) <= 250 for p in passages[1:])
    assert all(p.endswith(".") for p in passages)
    assert " ".join(" ".join(p.split()) for p in passages[1:]) == long


def test_passage_ids_are_stable_across_loads(data_dir):
    first, second = Corpus.load(data_dir / "corpus.jsonl"), Corpus.load(data_dir / "corpus.jsonl")
    assert all(first.passages(d) == second.passages(d) for d in first.docs)
    assert len(first.passages("d01")) == 3 and len(first.passages("d08")) == 10


def test_bm25_prefers_matching_text():
    scores = bm25(terms("iron bridge"), [terms("an iron bridge"), terms("a stone wall"), terms("iron")])
    assert scores[0] > scores[2] > scores[1] == 0


def test_memory_index_is_deterministic_and_finds_the_answer_document(corpus):
    index = MemoryIndex(corpus)
    query = "heritage registry code of the iron bridge over the river Tamel completed in 1887"
    assert index.search(query, 5) == index.search(query, 5)
    assert index.search(query, 5)[0] == "d01"
    assert index.search("zzzz", 5) == []


def test_search_records_retrieved_and_lists_hits(session):
    out = session.search("iron bridge Tamel")
    assert out.startswith("1. [d01] Ponte de Ferro do Tamel\n")
    assert session.retrieved[0] == "d01" and session.opened == [] and session.seen == {}
    assert len(out.split("\n")[1].split()) == 30  # snippet capped at snippet_tokens


def test_open_respects_the_cap_and_pages(session):
    first = session.open("d08")
    assert "[d08#p0]" in first and "[d08#p2]" in first and "[d08#p3]" not in first
    assert first.endswith("(passages 0 to 2 of 10; open with offset=3 for more)")
    last = session.open("d08", offset=9)
    assert last.endswith("(passages 9 to 9 of 10)")
    assert session.opened == ["d08"]
    assert list(session.seen) == ["d08#p0", "d08#p1", "d08#p2", "d08#p9"]


def test_open_rejects_unknown_docids_and_bad_offsets(session):
    with pytest.raises(ValueError, match="no document"):
        session.open("nope")
    with pytest.raises(ValueError, match="offset"):
        session.open("d01", offset=3)
    assert session.opened == []
