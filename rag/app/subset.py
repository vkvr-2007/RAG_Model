"""Create a compact, real MSMARCO-XI subset from the downloaded Hindi shard."""
from __future__ import annotations

import argparse
from pathlib import Path


def create_subset(source: Path, output: Path, limit: int) -> None:
    """Use DuckDB's Parquet reader so LIMIT does not materialize the huge row group."""
    import duckdb
    if not source.is_file():
        raise FileNotFoundError(source)
    if limit < 1:
        raise ValueError("limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    # Only fields needed by indexing are preserved. The resulting file is small,
    # self-contained, and contains genuine Hindi MSMARCO-XI passages.
    source_sql = str(source.resolve()).replace("'", "''")
    output_sql = str(output.resolve()).replace("'", "''")
    connection.execute(
        f"COPY (SELECT query_id, passages FROM read_parquet('{source_sql}') LIMIT {limit}) "
        f"TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 512)"
    )
    count = connection.execute(f"SELECT count(*) FROM read_parquet('{output_sql}')").fetchone()[0]
    if count != limit:
        raise RuntimeError(f"Subset validation failed: expected {limit} rows, wrote {count}")
    print(f"Wrote {count} real Hindi MSMARCO-XI rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a bounded local Hindi MSMARCO-XI subset.")
    parser.add_argument("--source", type=Path, default=Path("data/hintrain.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/hindi_subset.parquet"))
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    create_subset(args.source, args.output, args.limit)


if __name__ == "__main__":
    main()
