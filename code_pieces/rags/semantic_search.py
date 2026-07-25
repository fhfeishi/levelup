"""
Semantic search with sentence embeddings.

Keyword search matches surface forms. Semantic search maps query and documents
to vectors, so related expressions can be close even when words differ.

Optional install for real embeddings:
    pip install sentence-transformers

Model choices:
- Chinese/general: BAAI/bge-small-zh-v1.5
- Multilingual: intfloat/multilingual-e5-small
- English/general: sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Vector = Sequence[float]


@dataclass(frozen=True)
class SearchResult:
    rank: int
    doc_id: int
    score: float
    text: str


class SemanticRetriever:
    def __init__(self, texts: list[str], embeddings: Sequence[Vector]):
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        self.texts = texts
        self.embeddings = [normalize(vector) for vector in embeddings]

    def search(self, query_embedding: Vector | Sequence[Vector], top_k: int = 3) -> list[SearchResult]:
        query_vector = _first_vector(query_embedding)
        query_vector = normalize(query_vector)
        scores = [cosine_similarity(query_vector, embedding) for embedding in self.embeddings]
        ranked_ids = sorted(range(len(self.texts)), key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
        return [
            SearchResult(rank=rank, doc_id=doc_id, score=scores[doc_id], text=self.texts[doc_id])
            for rank, doc_id in enumerate(ranked_ids, start=1)
        ]


def normalize(vector: Vector) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def cosine_similarity(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def load_sentence_transformer(model_name: str = "BAAI/bge-small-zh-v1.5"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _first_vector(vector_or_vectors: Vector | Sequence[Vector]) -> Vector:
    if len(vector_or_vectors) == 0:
        return []
    first = vector_or_vectors[0]
    try:
        iter(first)  # type: ignore[arg-type]
    except TypeError:
        return vector_or_vectors  # type: ignore[return-value]
    return first  # type: ignore[return-value]


if __name__ == "__main__":
    docs = [
        "差旅政策规定了机票、酒店和报销流程。",
        "办公家具采购包括桌椅、显示器支架和会议室设备。",
        "RAG 系统通常包括文档切分、向量召回、重排序和答案生成。",
        "BM25 对错误码、接口名、专有名词这类精确关键词很有效。",
    ]
    query = "员工出差费用怎么报销"

    try:
        model = load_sentence_transformer()
    except ImportError:
        raise SystemExit("Missing dependency: pip install sentence-transformers")

    doc_embeddings = model.encode(docs, normalize_embeddings=True).tolist()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    retriever = SemanticRetriever(docs, doc_embeddings)
    for item in retriever.search(query_embedding, top_k=3):
        print(f"{item.rank}. score={item.score:.3f} doc={item.doc_id} {item.text}")
