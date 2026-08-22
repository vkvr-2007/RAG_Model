import duckdb

rows = duckdb.sql("SELECT query_id, query FROM 'data/hintrain.parquet' LIMIT 3").fetchall()
print("rows_read:", len(rows))
print("first:", rows[0])
