from app.embeddings import HashEmbedder
from app.indexer import build_records, write_index
from app.retrieval import Retriever


def test_dataset_processing_and_retrieval(tmp_path):
    rows = [{"query_id": "q1", "passages": {"is_selected": [1], "Translated_passages": ["दिल्ली भारत की राजधानी है।"]}}]
    records = build_records(rows, "passage", 100, 10, 10)
    assert records[0]["query_id"] == "q1"
    write_index(records, tmp_path, HashEmbedder())
    found, score, timings = Retriever(tmp_path, HashEmbedder()).search_with_timings("भारत की राजधानी", 1)
    assert found[0].metadata["chunk_id"] == records[0]["chunk_id"]
    assert score > 0 and timings["embedding_ms"] >= 0
