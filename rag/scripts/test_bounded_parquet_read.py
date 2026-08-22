"""Confirm DuckDB can read a tiny real Hindi slice via HTTP range requests."""
import duckdb

URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet?download=true"
connection = duckdb.connect()
rows = connection.execute(f"SELECT query_id, query, passages FROM read_parquet('{URL}') LIMIT 3").fetchall()
print("rows_read:", len(rows))
print("first_query_id:", rows[0][0] if rows else None)
print("first_query:", rows[0][1] if rows else None)
