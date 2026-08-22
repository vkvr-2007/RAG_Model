"""Local end-to-end check without downloading a model or calling an LLM."""
import asyncio
import tempfile
from pathlib import Path

from app.embeddings import HashEmbedder
from app.generation import Generator
from app.indexer import write_index
from app.retrieval import Retriever
from app.service import RAGService


async def main():
    with tempfile.TemporaryDirectory() as directory:
        records = [{"query_id": "demo-1", "chunk_id": "demo-1-0", "text": "दिल्ली भारत की राजधानी है।", "source_passage": "दिल्ली भारत की राजधानी है।", "language": "hi", "chunking_strategy": "passage"}]
        write_index(records, Path(directory), HashEmbedder())
        retriever = Retriever(Path(directory), HashEmbedder())
        chunks, score, _ = retriever.search_with_timings("भारत की राजधानी क्या है", 1)
        assert chunks and chunks[0].metadata["chunk_id"] == "demo-1-0"
        service = RAGService(retriever, Generator(None, None, None, 1), 1, 0.1)
        result = await service.query("भारत की राजधानी क्या है")
        assert not result.grounded  # no LLM configured: safe refusal
        print({"retrieved_chunk": chunks[0].metadata["chunk_id"], "dense_score": round(score, 4), "safe_fallback": not result.grounded, "latency_ms": result.latency_ms})


if __name__ == "__main__":
    asyncio.run(main())
