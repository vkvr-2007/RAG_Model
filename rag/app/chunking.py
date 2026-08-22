from __future__ import annotations

import re
from collections.abc import Iterable

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+")


def _words(text: str) -> list[str]:
    return text.split()


def passage_chunks(passage: str, **_: int) -> list[str]:
    return [passage.strip()] if passage and passage.strip() else []


def sentence_chunks(passage: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(passage) if s.strip()]
    if not sentences:
        return []
    output: list[str] = []
    bucket: list[str] = []
    size = 0
    for sentence in sentences:
        n = len(_words(sentence))
        if bucket and size + n > chunk_size:
            output.append(" ".join(bucket))
            tail: list[str] = []
            tail_size = 0
            for old in reversed(bucket):
                tail.insert(0, old)
                tail_size += len(_words(old))
                if tail_size >= overlap:
                    break
            bucket, size = tail, tail_size
        bucket.append(sentence)
        size += n
    if bucket:
        output.append(" ".join(bucket))
    return output


def recursive_chunks(passage: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    """Token windows, preferring paragraph/sentence boundaries where possible."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", passage):
        units.extend(s.strip() for s in SENTENCE_BOUNDARY.split(paragraph) if s.strip())
    if not units:
        return []
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = _words(unit)
        while words:
            remaining = chunk_size - count
            if remaining <= 0:
                chunks.append(" ".join(current))
                overlap_words = _words(" ".join(current))[-overlap:]
                current, count = overlap_words, len(overlap_words)
                remaining = chunk_size - count
            take = words[:remaining]
            current.extend(take)
            count += len(take)
            words = words[remaining:]
            if words or count >= chunk_size:
                chunks.append(" ".join(current))
                overlap_words = _words(" ".join(current))[-overlap:]
                current, count = overlap_words, len(overlap_words)
    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, strategy: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    strategies = {"passage": passage_chunks, "sentence": sentence_chunks, "recursive": recursive_chunks}
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return strategies[strategy](text, chunk_size=chunk_size, overlap=overlap)
