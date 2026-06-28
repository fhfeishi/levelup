"""
Minimal inverted-index keyword search.

This is the most transparent retrieval baseline:
1. tokenize documents
2. build token -> document ids
3. score documents by matched query terms

It is weaker than BM25, but useful for understanding and debugging.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    rank: int
    doc_id: int
    score: float
    matched_terms: tuple[str, ...]
    text: str


def tokenize(text: str) -> list[str]:
    """Tokenize English words and CJK phrases.

    Without a Chinese segmenter, CJK 2/3-grams are a practical fallback:
    "错误码" -> ["错误", "误码", "错误码"].
    """
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


def build_inverted_index(texts: list[str]) -> tuple[dict[str, set[int]], list[Counter[str]]]:
    inverted: dict[str, set[int]] = defaultdict(set)
    term_counts: list[Counter[str]] = []

    for doc_id, text in enumerate(texts):
        counts = Counter(tokenize(text))
        term_counts.append(counts)
        for term in counts:
            inverted[term].add(doc_id)

    return dict(inverted), term_counts


def search(
    query: str,
    texts: list[str],
    inverted: dict[str, set[int]],
    term_counts: list[Counter[str]],
    top_k: int = 5,
) -> list[SearchResult]:
    query_terms = tokenize(query)
    candidate_ids = set()
    for term in query_terms:
        candidate_ids.update(inverted.get(term, set()))

    n_docs = len(texts)
    scored: list[tuple[float, int, tuple[str, ...]]] = []
    for doc_id in candidate_ids:
        matched_terms = tuple(term for term in query_terms if term_counts[doc_id][term] > 0)
        score = 0.0
        for term in matched_terms:
            doc_freq = len(inverted[term])
            idf = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0
            score += term_counts[doc_id][term] * idf
        scored.append((score, doc_id, matched_terms))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        SearchResult(rank=rank, doc_id=doc_id, score=score, matched_terms=matched_terms, text=texts[doc_id])
        for rank, (score, doc_id, matched_terms) in enumerate(scored[:top_k], start=1)
    ]


if __name__ == "__main__":
    docs = [
        "RAG 检索包括关键词搜索、语义搜索、混合检索。",
        "BM25 适合精确词匹配、术语、错误码、函数名。",
        "语义搜索适合同义表达，例如报销制度和差旅政策。",
        "reranker 通常放在召回后，对候选 chunk 重新排序。",
    ]

    index, counts = build_inverted_index(docs)
    for item in search("BM25 错误码 检索", docs, index, counts):
        print(f"{item.rank}. score={item.score:.3f} terms={item.matched_terms} {item.text}")
