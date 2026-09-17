"""
Text chunking helpers for RAG.

Chunking is retrieval architecture, not just preprocessing. The chunk decides
what the retriever can return and what context the generator can see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start: int
    end: int


def recursive_split(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split text by preferred boundaries, then add character overlap.

    Boundary priority:
    1. markdown/code paragraphs
    2. lines
    3. sentences
    4. spaces
    5. characters
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    pieces = _split_by_boundaries(text.strip(), chunk_size)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current}\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece

    if current:
        chunks.append(current)

    if overlap == 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-overlap:]
        overlapped.append(f"{prefix}\n{chunk}")
    return overlapped


def make_chunks(doc_id: str, text: str, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    raw_chunks = recursive_split(text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    cursor = 0
    for index, chunk_text in enumerate(raw_chunks):
        start = text.find(chunk_text.strip(), cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk_text)
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}:{index}", text=chunk_text, start=start, end=end))
        cursor = max(cursor, end)
    return chunks


def _split_by_boundaries(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    separators = [r"\n\n+", r"\n+", r"(?<=[。！？.!?])", r"\s+"]
    parts = [text]
    for pattern in separators:
        next_parts: list[str] = []
        for part in parts:
            if len(part) <= chunk_size:
                next_parts.append(part)
                continue
            split_parts = [item.strip() for item in re.split(pattern, part) if item.strip()]
            next_parts.extend(split_parts if split_parts else [part])
        parts = next_parts

    final_parts: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            final_parts.append(part)
        else:
            final_parts.extend(part[i : i + chunk_size] for i in range(0, len(part), chunk_size))
    return final_parts


if __name__ == "__main__":
    sample = """
# RAG chunking

chunk 太大会召回一整片噪声，答案生成阶段容易被无关内容干扰。

chunk 太小会丢上下文，例如标题、表格字段含义、代码函数调用关系。
通常先从 500-1000 tokens 和 10%-20% overlap 开始，再用评估集调参。
"""

    for chunk in make_chunks("note-001", sample, chunk_size=80, overlap=12):
        print(f"\n[{chunk.chunk_id}] {chunk.start}:{chunk.end}\n{chunk.text}")
