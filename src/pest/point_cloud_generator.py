import os

import pyarrow as pa
import pyarrow.parquet as pq

from pest.galaxy import Galaxy
from pest.generator import Generator


class PointCloudGenerator(Generator):
    def __init__(
        self,
        output_directory: str,
        columns: list[str],
        chunk_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize the PointCloudGenerator."""

        super().__init__()
        self.columns = columns
        self.chunk_size = chunk_size
        self.compression = compression
        self.output_directory = output_directory

        self.idx = 0
        self.file_idx = 0

        os.makedirs(output_directory, exist_ok=True)

    def add_galaxy(self, galaxy: Galaxy):
        """Add a galaxy to the dataset and write it immediately as a new row to the Parquet file."""
        galaxy_data = {}

        for column in self.columns:
            parts = column.split("_", 1)
            if len(parts) == 2:
                particle_type, attribute = parts
                values = []
                for particle in getattr(galaxy, particle_type):
                    if hasattr(particle, attribute):
                        values.append(getattr(particle, attribute))
                galaxy_data[column] = values
            else:
                if hasattr(galaxy, column):
                    galaxy_data[column] = getattr(galaxy, column)

        # Write this galaxy as a single row to the Parquet file
        table = pa.Table.from_pylist([galaxy_data])
        pq.write_table(
            table,
            os.path.join(self.output_directory, f"{self.file_idx}.parquet"),
            compression=self.compression,
        )

        # Increment the index and file index
        self.idx += 1
        if self.idx >= self.chunk_size:
            self.idx = 0
            self.file_idx += 1
