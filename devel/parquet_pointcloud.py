import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

data = {
    "galaxy_id": [1, 2],
    "stars_pos": [
        [[1.2, 3.4, 5.1], [2.2, 4.5, 6.7]],
        [[7.7, 8.8, 9.9]],
    ],
    "stars_mass": [[1.5, 0.9], [2.3]],
    "stars_id": [[101, 102], [201]],
}
df = pd.DataFrame(data)

schema = pa.schema(
    [
        ("galaxy_id", pa.int64()),
        ("stars_pos", pa.list_(pa.list_(pa.float32(), list_size=3))),
        ("stars_mass", pa.list_(pa.float32())),
    ]
)

table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
pq.write_table(table, "pointcloud.parquet")
