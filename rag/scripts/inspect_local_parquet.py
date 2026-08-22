import pyarrow.parquet as pq

metadata = pq.ParquetFile("data/hintrain.parquet").metadata
print("rows", metadata.num_rows)
print("row_groups", metadata.num_row_groups)
for index in range(min(5, metadata.num_row_groups)):
    row_group = metadata.row_group(index)
    print(index, row_group.num_rows, row_group.total_byte_size)
