from __future__ import annotations

import logging
import re
import time

from app.generation import Generator
from app.schemas import QueryResponse, Source


_STOPWORDS = {
    "क्या", "कौन", "किस", "कौनसा", "कौनसी", "जिस", "जिसका", "जिसकी", "यह", "वह", "और", "या", "है",
    "के", "की", "में", "पर", "से", "का", "कि", "को", "ने", "हो", "कर", "है", "what", "when", "where",
    "which", "who", "why", "how", "the", "a", "an", "of", "is", "are", "to", "in", "on", "for",
    "at", "with", "from", "by", "it", "this", "that"
}


def _is_capital_query(query: str) -> bool:
    q = query.lower()
    return any(token in q for token in ["राजधानी", "capital", "मुख्यालय", "main city", "primary city"])


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w\s\u0900-\u097F]", " ", value.lower())


def _extract_terms(value: str) -> list[str]:
    text = _normalize_text(value)
    return [token for token in text.split() if token and token not in _STOPWORDS and len(token) > 1]


def _extract_subject_terms(query: str) -> list[str]:
    terms = _extract_terms(query)
    generic_terms = {
        "राजधानी", "capital", "मुख्यालय", "main", "city", "primary", "क्या", "कौन", "किस",
        "कौनसा", "कौनसी", "गणराज्य", "देश", "राज्य", "नगर", "शहर", "प्रांत", "स्थान",
        "country", "state", "city", "capital", "republic"
    }
    return [term for term in terms if term not in generic_terms]


def _chunk_matches_query(query: str, chunk) -> bool:
    subject_terms = _extract_subject_terms(query)
    evidence = _normalize_text(chunk.metadata["text"])
    if not subject_terms:
        return any(token in evidence for token in ["राजधानी", "capital", "मुख्यालय", "main city", "primary city"])
    return any(term in evidence for term in subject_terms)


def _has_query_evidence(query: str, chunks: list) -> bool:
    if not chunks:
        return False
    return _chunk_matches_query(query, chunks[0])


def _has_capital_evidence(query: str, chunks: list) -> bool:
    if not chunks:
        return False
    text = _normalize_text(chunks[0].metadata["text"])
    if not any(token in text for token in ["राजधानी", "capital", "मुख्यालय", "main city", "primary city"]):
        return False
    subject_terms = _extract_subject_terms(query)
    if not subject_terms:
        return True
    return any(term in text for term in subject_terms)


class RAGService:
    def __init__(self, retriever, generator: Generator, top_k: int, min_score: float, strict_extractive: bool = True):
        self.retriever, self.generator, self.top_k, self.min_score, self.strict_extractive = retriever, generator, top_k, min_score, strict_extractive

    async def query(self, query: str, generate: bool | None = None) -> QueryResponse:
        if generate is None:
            generate = not self.strict_extractive
        normalized = " ".join((query or "").split())
        if not normalized:
            return QueryResponse(answer="संदर्भ में पर्याप्त जानकारी नहीं है", grounded=False, confidence=0.0, sources=[], latency_ms=0.0)
        started = time.perf_counter()
        preprocess_ms = (time.perf_counter() - started) * 1000
        chunks, dense_score, timings = self.retriever.search_with_timings(normalized, self.top_k)
        sources = [Source(query_id=c.metadata["query_id"], chunk_id=c.metadata["chunk_id"], text=c.metadata["text"], language=c.metadata["language"], chunking_strategy=c.metadata["chunking_strategy"], score=round(c.score, 4)) for c in chunks]
        if not chunks or dense_score < self.min_score:
            total = (time.perf_counter()-started)*1000
            logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=0 total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], total)
            return QueryResponse(answer="संदर्भ में पर्याप्त जानकारी नहीं है", grounded=False, confidence=0.0, sources=[], latency_ms=round(total, 2))

        if not _has_query_evidence(normalized, chunks):
            total = (time.perf_counter()-started)*1000
            logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=0 total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], total)
            return QueryResponse(answer="संदर्भ में पर्याप्त जानकारी नहीं है", grounded=False, confidence=0.0, sources=[], latency_ms=round(total, 2))

        if _is_capital_query(normalized) and not _has_capital_evidence(normalized, chunks):
            total = (time.perf_counter()-started)*1000
            logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=0 total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], total)
            return QueryResponse(answer="संदर्भ में पर्याप्त जानकारी नहीं है", grounded=False, confidence=0.0, sources=[], latency_ms=round(total, 2))

        if not generate:
            total = (time.perf_counter()-started)*1000
            logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=0 total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], total)
            answer = sources[0].text.strip()
            return QueryResponse(answer=answer, grounded=True, confidence=round(min(1.0, max(0.0, dense_score)), 4), sources=sources, latency_ms=round(total, 2))

        if not self.generator or not getattr(self.generator, "base_url", None) or not getattr(self.generator, "api_key", None) or not getattr(self.generator, "model", None):
            total = (time.perf_counter()-started)*1000
            logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=0 total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], total)
            return QueryResponse(answer="संदर्भ में पर्याप्त जानकारी नहीं है", grounded=False, confidence=0.0, sources=[], latency_ms=round(total, 2))

        generation_started = time.perf_counter()
        context = "\n\n".join(f"[{i + 1}] {item.text}" for i, item in enumerate(sources))
        generated = await self.generator.answer(normalized, context)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        total = (time.perf_counter()-started)*1000
        logging.getLogger(__name__).info("rag timings preprocess=%.2f embedding=%.2f vector=%.2f bm25=%.2f fusion=%.2f generation=%.2f total=%.2f", preprocess_ms, timings["embedding_ms"], timings["vector_retrieval_ms"], timings["bm25_ms"], timings["fusion_ms"], generation_ms, total)
        return QueryResponse(answer=generated.answer, grounded=generated.grounded, confidence=round(min(1.0, max(0.0, dense_score)) if generated.grounded else 0.0, 4), sources=sources if generated.grounded else [], latency_ms=round(total, 2))
