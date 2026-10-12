import numpy as np
import pytest

from pest import ParquetWriter, validate_parquet

RECORDS = [{"subhalo_id": i, "image": np.zeros((3, 4, 4), dtype=np.float32)} for i in range(3)]


def test_parquet_writer_creates_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "out.parquet"

    ParquetWriter(str(output_path))(RECORDS)

    assert output_path.exists()


def test_validate_parquet_summary(tmp_path):
    output_path = tmp_path / "out.parquet"
    ParquetWriter(str(output_path))(RECORDS)

    summary = validate_parquet(output_path, min_rows=3, required_columns=["subhalo_id", "image"])

    assert summary["path"] == str(output_path)
    assert summary["num_rows"] == 3
    assert summary["columns"] == ["subhalo_id", "image"]
    assert summary["size_bytes"] > 0


def test_validate_parquet_too_few_rows(tmp_path):
    output_path = tmp_path / "out.parquet"
    ParquetWriter(str(output_path))(RECORDS)

    with pytest.raises(ValueError, match="rows"):
        validate_parquet(output_path, min_rows=4)


def test_validate_parquet_missing_columns(tmp_path):
    output_path = tmp_path / "out.parquet"
    ParquetWriter(str(output_path))(RECORDS)

    with pytest.raises(ValueError, match="missing columns"):
        validate_parquet(output_path, required_columns=["subhalo_id", "pos"])


def test_validate_parquet_empty_file(tmp_path):
    output_path = tmp_path / "out.parquet"
    ParquetWriter(str(output_path))([])

    assert validate_parquet(output_path, min_rows=0)["num_rows"] == 0
    with pytest.raises(ValueError):
        validate_parquet(output_path)
