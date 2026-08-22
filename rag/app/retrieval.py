from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class RetrievedChunk:
    metadata: dict
    score: float


class Retriever:
    def __init__(self, index_dir: Path, embedder, use_bm25: bool = True):
        import faiss
        self.index = faiss.read_index(str(index_dir / "vectors.faiss"))
        self.metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.embedder = embedder
        self.bm25 = None
        bm25_file = index_dir / "bm25.pkl"
        if use_bm25 and bm25_file.exists():
            with bm25_file.open("rb") as handle:
                self.bm25 = pickle.load(handle)

    def search_dense(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = self.embedder.encode([query])
        scores, ids = self.index.search(vector, min(k, len(self.metadata)))
        return [RetrievedChunk(self.metadata[i], float(score)) for score, i in zip(scores[0], ids[0]) if i >= 0]

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        dense = self.search_dense(query, max(k * 3, k))
        if not self.bm25:
            return dense[:k]
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_ids = np.argsort(sparse_scores)[::-1][: max(k * 3, k)]
        ranks: dict[int, float] = {}
        by_id = {id(item.metadata): item for item in dense}
        # Reciprocal-rank fusion avoids comparing incomparable score scales.
        for rank, item in enumerate(dense, 1): ranks[item.metadata["_row"]] = ranks.get(item.metadata["_row"], 0.0) + 1 / (60 + rank)
        for rank, row in enumerate(sparse_ids, 1): ranks[int(row)] = ranks.get(int(row), 0.0) + 1 / (60 + rank)
        return [RetrievedChunk(self.metadata[row], score) for row, score in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:k]]

    def search_with_timings(self, query: str, k: int) -> tuple[list[RetrievedChunk], float, dict[str, float]]:
        import time
        embedding_started = time.perf_counter()
        vector = self.embedder.encode([query])
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        vector_started = time.perf_counter()
        scores, ids = self.index.search(vector, min(max(k * 3, k), len(self.metadata)))
        dense = [RetrievedChunk(self.metadata[i], float(score)) for score, i in zip(scores[0], ids[0]) if i >= 0]
        vector_ms = (time.perf_counter() - vector_started) * 1000
        if not self.bm25:
            return dense[:k], (dense[0].score if dense else 0.0), {"embedding_ms": embedding_ms, "vector_retrieval_ms": vector_ms, "bm25_ms": 0.0, "fusion_ms": 0.0}
        bm25_started = time.perf_counter()
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_ids = np.argsort(sparse_scores)[::-1][: max(k * 3, k)]
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        fusion_started = time.perf_counter()
        ranks: dict[int, float] = {}
        for rank, item in enumerate(dense, 1): ranks[item.metadata["_row"]] = ranks.get(item.metadata["_row"], 0.0) + 1 / (60 + rank)
        for rank, row in enumerate(sparse_ids, 1): ranks[int(row)] = ranks.get(int(row), 0.0) + 1 / (60 + rank)
        fused = [RetrievedChunk(self.metadata[row], score) for row, score in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:k]]
        return fused, (dense[0].score if dense else 0.0), {"embedding_ms": embedding_ms, "vector_retrieval_ms": vector_ms, "bm25_ms": bm25_ms, "fusion_ms": (time.perf_counter() - fusion_started) * 1000}
