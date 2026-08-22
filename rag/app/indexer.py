from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import chunk_text
from app.embeddings import build_embedder


def load_hindi_dataset():
    """Stream Hindi rows, working around missing config metadata on the Hub repo."""
    from datasets import load_dataset
    try:
        return load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train", streaming=True)
    except ValueError as error:
        if "BuilderConfig 'hi' not found" not in str(error):
            raise
        print("Dataset config 'hi' is unavailable; streaming train/hintrain.parquet directly.")
        return load_dataset("parquet", data_files="hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet", split="train", streaming=True)


def load_local_parquet(path: Path):
    """Yield rows locally in bounded record batches; never load the 3.7 GB file at once."""
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    needed = ["query_id", "passages"]
    for batch in parquet.iter_batches(batch_size=512, columns=needed):
        yield from batch.to_pylist()


def selected_passages(row: dict) -> list[str]:
    passages = row.get("passages", {})
    translated = passages.get("Translated_passages", []) or []
    selected = passages.get("is_selected", []) or []
    cleaned = []
    for text, flag in zip(translated, selected):
        if flag and text and text.strip():
            cleaned.append(text.strip())
    if not cleaned:
        cleaned = [text.strip() for text in translated if text and text.strip()][:1]
    return cleaned


def _sanitize_text(text: str, limit: int = 1500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    trimmed = cleaned[:limit].rsplit(" ", 1)[0] if " " in cleaned[:limit] else cleaned[:limit]
    return trimmed.strip()


def build_records(rows, strategy: str, chunk_size: int, overlap: int, limit: int) -> list[dict]:
    records = []
    for example_no, row in enumerate(rows):
        if example_no >= limit:
            break
        query_id = str(row.get("query_id", example_no))
        for passage_no, passage in enumerate(selected_passages(row)):
            passage = _sanitize_text(passage)
            for piece_no, text in enumerate(chunk_text(passage, strategy, chunk_size, overlap)):
                text = _sanitize_text(text, 800)
                if not text:
                    continue
                records.append({
                    "query_id": query_id,
                    "chunk_id": f"{example_no}-{passage_no}-{piece_no}",
                    "text": text,
                    "source_passage": passage,
                    "language": "hi",
                    "chunking_strategy": strategy,
                })
    return records


def write_index(records: list[dict], output_dir: Path, embedder, use_bm25: bool = True, batch_size: int = 64) -> None:
    import faiss
    if not records:
        raise ValueError("No chunks produced; check dataset fields and subset size")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.vstack([embedder.encode([r["text"] for r in records[i:i + batch_size]]) for i in range(0, len(records), batch_size)])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "vectors.faiss"))
    for row, record in enumerate(records):
        record["_row"] = row
    (output_dir / "metadata.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    if use_bm25:
        with (output_dir / "bm25.pkl").open("wb") as handle:
            pickle.dump(BM25Okapi([r["text"].lower().split() for r in records]), handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and index a bounded MSMARCO-XI subset.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--input-parquet", type=Path, help="Local Hindi MSMARCO-XI Parquet shard; preferred for reliable offline indexing")
    parser.add_argument("--output-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--strategy", choices=["passage", "sentence", "recursive"], default="sentence")
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--embedding-backend", choices=["sentence_transformers", "hash"], default="sentence_transformers")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--no-bm25", action="store_true")
    args = parser.parse_args()
    if args.input_parquet:
        if not args.input_parquet.is_file():
            raise FileNotFoundError(f"Local Parquet file not found: {args.input_parquet}")
        dataset = load_local_parquet(args.input_parquet)
    else:
        dataset = load_hindi_dataset()
    records = build_records(dataset, args.strategy, args.chunk_size, args.overlap, args.limit)
    write_index(records, args.output_dir, build_embedder(args.embedding_backend, args.embedding_model), not args.no_bm25)
    print(f"Wrote {len(records)} chunks from at most {args.limit} streamed examples to {args.output_dir}")


if __name__ == "__main__":
    main()
