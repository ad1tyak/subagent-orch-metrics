"""The document collection, BM25 retrieval, and the search session each subagent works through."""

import json
import math
import re
from collections import Counter

from . import tokens

_TERM = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def terms(text: str) -> list[str]:
    return _TERM.findall(text.lower())


def bm25(query: list[str], texts: list[list[str]], k1: float = 0.9, b: float = 0.4) -> list[float]:
    """BM25 score of each text against the query, with Pyserini's default k1 and b."""
    if not texts:
        return []
    avg_len = sum(map(len, texts)) / len(texts) or 1
    df = Counter(term for text in texts for term in set(text))
    scores = []
    for text in texts:
        tf = Counter(text)
        score = 0.0
        for term in set(query):
            if tf[term]:
                idf = math.log(1 + (len(texts) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(text) / avg_len))
        scores.append(score)
    return scores


def split_passages(text: str, min_words: int = 80, max_words: int = 250) -> list[str]:
    """Split on blank lines, cut long paragraphs at sentence ends, and merge short neighbors."""
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if len(paragraph.split()) <= max_words:
            if paragraph:
                pieces.append(paragraph)
            continue
        chunk = []
        for sentence in _SENTENCE_END.split(paragraph):
            if chunk and len(" ".join(chunk + [sentence]).split()) > max_words:
                pieces.append(" ".join(chunk))
                chunk = []
            chunk.append(sentence)
        pieces.append(" ".join(chunk))
    merged = []
    for piece in pieces:
        if merged and len(merged[-1].split()) < min_words and len(merged[-1].split()) + len(piece.split()) <= max_words:
            merged[-1] += "\n\n" + piece
        else:
            merged.append(piece)
    return merged


class Corpus:
    def __init__(self, docs: dict[str, dict]):
        self.docs = docs  # docid -> {"title", "text"}
        self._passages: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path) -> "Corpus":
        """JSONL with docid, text and an optional title per line."""
        # ponytail: the whole corpus stays in memory; read documents from the Lucene index if RAM runs short
        docs = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    docs[str(record["docid"])] = {"title": record.get("title") or "", "text": record["text"]}
        return cls(docs)

    def title(self, docid: str) -> str:
        doc = self.docs[docid]
        return doc["title"] or doc["text"].strip().split("\n", 1)[0][:80]

    def text(self, docid: str) -> str:
        return self.docs[docid]["text"]

    def passages(self, docid: str) -> list[str]:
        if docid not in self._passages:
            self._passages[docid] = split_passages(self.docs[docid]["text"])
        return self._passages[docid]


class MemoryIndex:
    """BM25 over the corpus in memory, for tests and small dev corpora. Real runs use PyseriniIndex."""

    def __init__(self, corpus: Corpus):
        self.docids = list(corpus.docs)
        self.texts = [terms(corpus.title(d) + " " + corpus.text(d)) for d in self.docids]

    def search(self, query: str, k: int) -> list[str]:
        # ponytail: scores every document on every query; fine for thousands of documents, not 100K
        scores = bm25(terms(query), self.texts)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.docids[i] for i in ranked[:k] if scores[i] > 0]


class PyseriniIndex:
    """The official BrowseComp-Plus BM25 index (pip install '.[bm25]'; needs Java 21)."""

    def __init__(self, index_dir: str):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(index_dir)

    def search(self, query: str, k: int) -> list[str]:
        return [hit.docid for hit in self.searcher.search(query, k)]


class SearchSession:
    """One subagent's view of the collection. Records what it retrieved, opened and saw."""

    def __init__(self, corpus: Corpus, index, *, k: int, snippet_tokens: int, open_tokens: int, tokenizer: str):
        self.corpus, self.index = corpus, index
        self.k, self.snippet_tokens, self.open_tokens, self.tokenizer = k, snippet_tokens, open_tokens, tokenizer
        self.retrieved: list[str] = []
        self.opened: list[str] = []
        self.seen: dict[str, str] = {}

    def search(self, query: str) -> str:
        hits = [docid for docid in self.index.search(query, self.k) if docid in self.corpus.docs]
        if not hits:
            return "No documents matched."
        results = []
        for rank, docid in enumerate(hits, 1):
            if docid not in self.retrieved:
                self.retrieved.append(docid)
            snippet = tokens.truncate(self.tokenizer, " ".join(self.corpus.text(docid).split()), self.snippet_tokens)
            results.append(f"{rank}. [{docid}] {self.corpus.title(docid)}\n{snippet}")
        return "\n\n".join(results)

    def open(self, docid: str, offset: int = 0) -> str:
        if docid not in self.corpus.docs:
            raise ValueError(f"no document has docid {docid}")
        passages = self.corpus.passages(docid)
        if not passages:
            return f"[{docid}] {self.corpus.title(docid)}\n\n(empty document)"
        if not 0 <= offset < len(passages):
            raise ValueError(f"offset must be between 0 and {len(passages) - 1}")
        if docid not in self.opened:
            self.opened.append(docid)
        shown, used = [], 0
        for i in range(offset, len(passages)):
            line = f"[{docid}#p{i}] {passages[i]}"
            size = tokens.count(self.tokenizer, line)
            if shown and used + size > self.open_tokens:
                break
            shown.append(line)
            used += size
            self.seen[f"{docid}#p{i}"] = passages[i]
        end = offset + len(shown)
        more = f"; open with offset={end} for more" if end < len(passages) else ""
        return f"[{docid}] {self.corpus.title(docid)}\n\n" + "\n\n".join(shown) + f"\n\n(passages {offset} to {end - 1} of {len(passages)}{more})"
