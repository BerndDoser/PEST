import os
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq

from pest.generator import Generator


class PointCloudGenerator(Generator):
    def __init__(
        self,
        chunk_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize the PointCloudGenerator."""
        super().__init__()
        self.chunk_size = chunk_size
        self.compression = compression

    def __call__(self, input_data: Dict[str, Any], output_directory: str):
        """Generate Parquet dataset from input data."""

        os.makedirs(output_directory, exist_ok=True)

        batch = []
        file_idx = 0

        # Iterate over all input directories
        for galaxy in input_data["galaxies"]:
            stars_pos = galaxy["particles"]["stars"][0]["position"]
            arr = pa.array([stars_pos], type=pa.list_(pa.float32()))
            table = pa.Table.from_arrays([arr], names=["stars_pos"])

            batch.append(table)
            # Write batch if chunk_size reached
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
