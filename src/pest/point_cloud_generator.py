import os
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq

from pest.generator import Generator


class PointCloudGenerator(Generator):
    def __init__(
        self,
        columns: list[str],
        chunk_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize the PointCloudGenerator."""
        super().__init__()
        self.columns = columns
        self.chunk_size = chunk_size
        self.compression = compression

    def __call__(self, input_data: Dict[str, Any], output_directory: str):
        """Generate Parquet dataset from input data."""

        os.makedirs(output_directory, exist_ok=True)

        batch = []
        file_idx = 0
        galaxy_ids = [galaxy["id"] for galaxy in input_data["galaxies"]]
        table = pa.table({"galaxy_id": pa.array(galaxy_ids)})

        # Iterate over all input directories
        for galaxy in input_data["galaxies"]:
            table

            if "stars_position" in self.columns:
                for star in galaxy["particles"]["stars"]:
                    star_arr = pa.array([star["position"]], type=pa.list_(pa.float32(), list_size=3))
                    table = pa.concat_tables([table, pa.Table.from_arrays([star_arr], names=["stars_pos"])])

            # Write batch if chunk_size reached
            batch.append(table)
            if self.chunk_size and len(batch) >= self.chunk_size:
                pq.write_table(
                    pa.concat_tables(batch),
                    f"{output_directory}/{file_idx}.parquet",
                    compression=self.compression,
                )
                file_idx += 1
                batch = []

        # Write any remaining data
        if batch:
            pq.write_table(
                pa.concat_tables(batch),
                f"{output_directory}/{file_idx}.parquet",
                compression=self.compression,
            )
