"""
Hybrid retrieval by Reciprocal Rank Fusion (RRF).

Common RAG production pattern:
1. lexical retriever: BM25 / keyword search
2. vector retriever: semantic search
3. fusion: combine result ranks
4. reranker: optional cross-encoder or LLM scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RankedItem:
    doc_id: int
    rank: int
    source: str
    score: float = 0.0


@dataclass(frozen=True)
class FusedResult:
    rank: int
    doc_id: int
    fused_score: float
    sources: tuple[str, ...]
    text: str


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[RankedItem]],
    texts: list[str],
    top_k: int = 5,
    k: int = 60,
) -> list[FusedResult]:
    """Fuse multiple ranked lists using RRF.

    RRF cares about rank position, not raw scores, so it can combine BM25
    scores and embedding cosine scores without normalization headaches.
    """
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + 1.0 / (k + item.rank)
            sources.setdefault(item.doc_id, set()).add(item.source)

    ranked_doc_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        FusedResult(
            rank=rank,
            doc_id=doc_id,
            fused_score=scores[doc_id],
            sources=tuple(sorted(sources[doc_id])),
            text=texts[doc_id],
        )
        for rank, doc_id in enumerate(ranked_doc_ids, start=1)
    ]


if __name__ == "__main__":
    docs = [
        "BM25 适合精确关键词、错误码和 API 名称。",
        "语义检索适合同义问题，比如差旅报销和出差费用。",
        "混合检索常把 BM25 和向量召回合并，再进入 reranker。",
        "chunk 太大会稀释相关信息，太小会丢上下文。",
    ]

    bm25_results = [
        RankedItem(doc_id=0, rank=1, source="bm25", score=7.2),
        RankedItem(doc_id=2, rank=2, source="bm25", score=4.1),
    ]
    vector_results = [
        RankedItem(doc_id=1, rank=1, source="vector", score=0.82),
        RankedItem(doc_id=2, rank=2, source="vector", score=0.76),
    ]

    for item in reciprocal_rank_fusion([bm25_results, vector_results], docs, top_k=3):
        print(f"{item.rank}. score={item.fused_score:.4f} sources={item.sources} {item.text}")
