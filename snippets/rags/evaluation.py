"""
Small retrieval evaluation metrics.

Build a tiny golden set before tuning chunk_size, top_k, BM25/vector weights,
reranker prompts, or embedding models.
"""

from __future__ import annotations


def hit_rate(results: list[list[str]], expected_doc_ids: list[set[str]], k: int = 5) -> float:
    """Fraction of queries with any relevant document in top-k."""
    hits = 0
    for retrieved, expected in zip(results, expected_doc_ids, strict=True):
        if set(retrieved[:k]) & expected:
            hits += 1
    return hits / len(results) if results else 0.0


def mrr(results: list[list[str]], expected_doc_ids: list[set[str]], k: int = 5) -> float:
    """Mean reciprocal rank for the first relevant document in top-k."""
    total = 0.0
    for retrieved, expected in zip(results, expected_doc_ids, strict=True):
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in expected:
                total += 1.0 / rank
                break
    return total / len(results) if results else 0.0


def precision_at_k(results: list[list[str]], expected_doc_ids: list[set[str]], k: int = 5) -> float:
    """Average precision@k for multi-relevant retrieval tasks."""
    total = 0.0
    for retrieved, expected in zip(results, expected_doc_ids, strict=True):
        if k == 0:
            continue
        total += len(set(retrieved[:k]) & expected) / k
    return total / len(results) if results else 0.0


if __name__ == "__main__":
    retrieved_doc_ids = [
        ["doc-travel", "doc-office", "doc-rag"],
        ["doc-bm25", "doc-vector", "doc-chunking"],
        ["doc-rerank", "doc-rag", "doc-bm25"],
    ]
    golden_doc_ids = [
        {"doc-travel"},
        {"doc-bm25", "doc-keyword"},
        {"doc-rag"},
    ]

    print("hit@2 =", hit_rate(retrieved_doc_ids, golden_doc_ids, k=2))
    print("mrr@3 =", mrr(retrieved_doc_ids, golden_doc_ids, k=3))
    print("precision@2 =", precision_at_k(retrieved_doc_ids, golden_doc_ids, k=2))
