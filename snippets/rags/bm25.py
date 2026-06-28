"""
Pure Python BM25 keyword search.

BM25 is a strong lexical baseline for RAG retrieval because it handles:
- IDF: rare terms are more important
- TF saturation: repeating a term does not increase score forever
- length normalization: long documents are not always preferred
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    rank: int
    doc_id: int
    score: float
    text: str


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(text) for text in texts]
        self.term_counts = [Counter(tokens) for tokens in self.tokenized_docs]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.idf = self._build_idf()

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_terms = tokenize(query)
        scores = [self._score_doc(query_terms, doc_id) for doc_id in range(len(self.texts))]
        ranked_ids = sorted(range(len(self.texts)), key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
        return [
            SearchResult(rank=rank, doc_id=doc_id, score=scores[doc_id], text=self.texts[doc_id])
            for rank, doc_id in enumerate(ranked_ids, start=1)
            if scores[doc_id] > 0
        ]

    def _build_idf(self) -> dict[str, float]:
        n_docs = len(self.texts)
        doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized_docs:
            doc_freq.update(set(tokens))

        return {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def _score_doc(self, query_terms: list[str], doc_id: int) -> float:
        score = 0.0
        counts = self.term_counts[doc_id]
        doc_length = self.doc_lengths[doc_id]

        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf.get(term, 0.0)
            numerator = freq * (self.k1 + 1.0)
            denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for English/code-ish text and CJK phrases."""
    tokens: list[str] = []
    for part in re.findall(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.append(part)
            for n in (2, 3):
                if len(part) > n:
                    tokens.extend(part[i : i + n] for i in range(0, len(part) - n + 1))
        else:
            tokens.append(part)
    return tokens


def build_bm25(texts: list[str]) -> BM25Index:
    return BM25Index(texts)


def search(query: str, texts: list[str], bm25: BM25Index, top_k: int = 3) -> list[SearchResult]:
    return bm25.search(query, top_k=top_k)


if __name__ == "__main__":
    docs = [
        "Office equipment policy covers monitors, keyboards and headsets.",
        "Office furniture guidelines explain chairs, desks and meeting rooms.",
        "Travel policy explains reimbursement, hotels and flight booking.",
        "RAG retrieval uses keyword search, semantic search and reranking.",
    ]

    index = build_bm25(docs)

    for item in search("furniture policy", docs, index, top_k=3):
        print(f"{item.rank}. score={item.score:.3f} doc={item.doc_id} {item.text}")
