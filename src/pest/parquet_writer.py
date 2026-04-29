"""Write pipeline items to Parquet files with configurable chunk size."""

import importlib
import os
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pest.generator import Generator


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


class ParquetWriter(Generator):
    """Write pipeline items to Parquet files, splitting output by chunk_size.

    Items received via ``process()`` are accumulated into a batch. When the
    batch reaches ``chunk_size`` rows it is flushed to a numbered output file
    (``part-0.parquet``, ``part-1.parquet``, …).  Any remaining rows are
    written on ``close()``.

    Each column entry in ``columns`` may optionally carry a ``generator``
    sub-config.  That sub-generator must expose a ``generate(item)`` method
    that returns the column value for a given item.
    """

    def __init__(
        self,
        output_directory: str = "output",
        chunk_size: int = 1000,
        compression: str = "snappy",
        columns: list[dict] | None = None,
    ):
        """Initialize the ParquetWriter.

        Args:
            output_directory: Directory where Parquet files will be written.
            chunk_size: Maximum number of rows per output file (default: 1000).
            compression: Parquet compression codec (default: "snappy").
            columns: Optional list of column configs. Each entry is a dict with
                a ``name`` key and an optional ``generator`` sub-config dict
                containing ``class_path`` and ``init_args``.  When omitted all
                fields of each incoming item are written as-is.
        """
        self.output_directory = output_directory
        self.chunk_size = chunk_size
        self.compression = compression
        self.columns = columns or []

        # Instantiate per-column sub-generators where provided
        self._column_generators: dict[str, Any] = {}
        for col in self.columns:
            gen_cfg = col.get("generator")
            if gen_cfg:
                self._column_generators[col["name"]] = _instantiate(gen_cfg["class_path"], gen_cfg.get("init_args", {}))

        os.makedirs(output_directory, exist_ok=True)

        self._batch: list[dict] = []
        self._file_idx: int = 0

    def process(self, item: Any) -> None:
        """Accumulate one item and flush to disk when chunk_size is reached.

        Args:
            item: A ``dict`` or an object whose attributes match the configured
                column names.  When no columns are configured all keys/attributes
                of the item are written.
        """
        if self.columns:
            row: dict = {}
            for col in self.columns:
                name = col["name"]
                gen = self._column_generators.get(name)
                if gen is not None:
                    row[name] = gen.generate(item)
                elif isinstance(item, dict):
                    row[name] = item[name]
                else:
                    row[name] = getattr(item, name)
        else:
            row = dict(item) if isinstance(item, dict) else vars(item)

        self._batch.append(row)
        if len(self._batch) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        """Write the current batch to a new Parquet file and reset the buffer."""
        if not self._batch:
            return
        table = pa.Table.from_pylist(self._batch)
        path = os.path.join(self.output_directory, f"part-{self._file_idx}.parquet")
        pq.write_table(table, path, compression=self.compression)
        self._file_idx += 1
        self._batch = []

    def close(self) -> None:
        """Flush any remaining rows to disk."""
        self._flush()
