"""Inspect the currently published MSMARCO-XI streaming schema without downloading it all."""
from itertools import islice
from app.indexer import load_hindi_dataset

dataset = load_hindi_dataset()
row = next(islice(dataset, 1))
print("columns:", sorted(row.keys()))
print("sample:", {key: row[key] for key in row if key not in {"passages"}})
print("passage keys:", sorted((row.get("passages") or {}).keys()))
