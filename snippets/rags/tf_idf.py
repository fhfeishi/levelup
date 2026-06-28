"""
Pure Python TF-IDF keyword search.

TF-IDF is a transparent lexical baseline:
- TF: a term is important if it appears often in this document
- IDF: a term is important if it appears in fewer documents

For larger corpora, use scikit-learn, Elasticsearch/OpenSearch, Tantivy,
Lucene, or another indexed search engine.
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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]+", text.lower())


def build_tfidf_vectors(texts: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    tokenized_docs = [tokenize(text) for text in texts]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    n_docs = len(texts)
    idf = {
        term: math.log((n_docs + 1) / (freq + 1)) + 1.0
        for term, freq in doc_freq.items()
    }

    vectors = [_tfidf_vector(tokens, idf) for tokens in tokenized_docs]
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    return _tfidf_vector(tokenize(query), idf)


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in common_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def search(query: str, texts: list[str], vectors: list[dict[str, float]], idf: dict[str, float], top_k: int = 3) -> list[SearchResult]:
    query_vector = vectorize_query(query, idf)
    scores = [cosine_similarity(query_vector, vector) for vector in vectors]
    ranked_ids = sorted(range(len(texts)), key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
    return [
        SearchResult(rank=rank, doc_id=doc_id, score=scores[doc_id], text=texts[doc_id])
        for rank, doc_id in enumerate(ranked_ids, start=1)
        if scores[doc_id] > 0
    ]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {
        term: (count / total) * idf.get(term, 0.0)
        for term, count in counts.items()
        if term in idf
    }


if __name__ == "__main__":
    docs = [
        "Office equipment policy covers monitors, keyboards and headsets.",
        "Office furniture guidelines explain chairs, desks and meeting rooms.",
        "Travel policy explains reimbursement, hotels and flight booking.",
        "RAG retrieval uses keyword search, semantic search and reranking.",
    ]

    doc_vectors, idf_values = build_tfidf_vectors(docs)

    for item in search("furniture policy", docs, doc_vectors, idf_values, top_k=3):
        print(f"{item.rank}. score={item.score:.3f} doc={item.doc_id} {item.text}")
