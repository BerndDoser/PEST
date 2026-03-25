"""Tests for the Pipeline class."""

from unittest.mock import MagicMock, call, patch

import pyarrow.parquet as pq
import pytest

from pest.galaxy import Galaxy, Gas, Star
from pest.pipeline import Pipeline
from pest.point_cloud_generator import PointCloudGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_star(i=0):
    return Star(id=i, mass=1.0, position=[float(i), 0.0, 0.0], velocity=[0.0, 0.0, 0.0], luminosity=2.5)


def _make_gas(i=100):
    return Gas(id=i, mass=0.5, position=[float(i), 1.0, 0.0], velocity=[0.0, 0.0, 0.0], temperature=1e4)


def _make_galaxy(gid, central=True):
    return Galaxy(
        id=gid,
        central=central,
        mass=1e10,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        stars=[_make_star(gid)],
        gas=[_make_gas(gid + 100)],
    )


def _mock_extractor(galaxies):
    """Return a mock Extractor whose extract() returns the given galaxy list."""
    extractor = MagicMock()
    extractor.extract.return_value = galaxies
    return extractor


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_stores_extractor_and_generators(self):
        extractor = _mock_extractor([])
        gen = MagicMock()
        pipeline = Pipeline(extractor, [gen])
        assert pipeline.extractor is extractor
        assert pipeline.generators == [gen]


class TestPipelineRun:
    def test_calls_extract_once(self):
        extractor = _mock_extractor([])
        gen = MagicMock()
        Pipeline(extractor, [gen]).run()
        extractor.extract.assert_called_once()

    def test_add_galaxy_called_for_each_galaxy(self):
        galaxies = [_make_galaxy(1), _make_galaxy(2), _make_galaxy(3)]
        extractor = _mock_extractor(galaxies)
        gen = MagicMock()
        Pipeline(extractor, [gen]).run()
        assert gen.add_galaxy.call_count == 3
        gen.add_galaxy.assert_has_calls([call(g) for g in galaxies])

    def test_multiple_generators_each_receive_all_galaxies(self):
        galaxies = [_make_galaxy(1), _make_galaxy(2)]
        extractor = _mock_extractor(galaxies)
        gen_a, gen_b = MagicMock(), MagicMock()
        Pipeline(extractor, [gen_a, gen_b]).run()
        assert gen_a.add_galaxy.call_count == 2
        assert gen_b.add_galaxy.call_count == 2

    def test_close_called_on_every_generator(self):
        extractor = _mock_extractor([_make_galaxy(1)])
        gen_a, gen_b = MagicMock(), MagicMock()
        Pipeline(extractor, [gen_a, gen_b]).run()
        gen_a.close.assert_called_once()
        gen_b.close.assert_called_once()

    def test_close_called_even_when_add_galaxy_raises(self):
        galaxies = [_make_galaxy(1)]
        extractor = _mock_extractor(galaxies)
        gen = MagicMock()
        gen.add_galaxy.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            Pipeline(extractor, [gen]).run()
        gen.close.assert_called_once()

    def test_no_galaxies_no_add_galaxy_calls(self):
        extractor = _mock_extractor([])
        gen = MagicMock()
        Pipeline(extractor, [gen]).run()
        gen.add_galaxy.assert_not_called()
        gen.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration test with PointCloudGenerator
# ---------------------------------------------------------------------------

class TestPipelineWithPointCloudGenerator:
    COLUMNS = ["id", "stars_position", "stars_mass", "stars_luminosity", "gas_position", "gas_mass", "gas_temperature"]

    def test_parquet_file_created(self, tmp_path):
        galaxies = [_make_galaxy(1), _make_galaxy(2)]
        extractor = _mock_extractor(galaxies)
        generator = PointCloudGenerator(
            output_directory=str(tmp_path),
            columns=self.COLUMNS,
        )
        Pipeline(extractor, [generator]).run()
        assert (tmp_path / "part-0.parquet").exists()

    def test_parquet_row_count_matches_galaxies(self, tmp_path):
        galaxies = [_make_galaxy(i) for i in range(5)]
        extractor = _mock_extractor(galaxies)
        generator = PointCloudGenerator(
            output_directory=str(tmp_path),
            columns=self.COLUMNS,
        )
        Pipeline(extractor, [generator]).run()
        table = pq.read_table(str(tmp_path / "part-0.parquet"))
        assert table.num_rows == 5

    def test_parquet_chunking_creates_multiple_files(self, tmp_path):
        galaxies = [_make_galaxy(i) for i in range(6)]
        extractor = _mock_extractor(galaxies)
        generator = PointCloudGenerator(
            output_directory=str(tmp_path),
            columns=self.COLUMNS,
            chunk_size=4,
        )
        Pipeline(extractor, [generator]).run()
        # 6 galaxies with chunk_size=4 → part-0 (4 rows) + part-1 (2 rows)
        assert (tmp_path / "part-0.parquet").exists()
        assert (tmp_path / "part-1.parquet").exists()
        total_rows = sum(
            pq.read_table(str(f)).num_rows for f in tmp_path.glob("*.parquet")
        )
        assert total_rows == 6

    def test_parquet_star_luminosity_values(self, tmp_path):
        star = Star(id=1, mass=1.0, position=[1.0, 2.0, 3.0], velocity=[0.0, 0.0, 0.0], luminosity=42.0)
        galaxy = Galaxy(id=1, central=True, mass=1e10, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], stars=[star])
        extractor = _mock_extractor([galaxy])
        generator = PointCloudGenerator(
            output_directory=str(tmp_path),
            columns=["id", "stars_luminosity"],
        )
        Pipeline(extractor, [generator]).run()
        table = pq.read_table(str(tmp_path / "part-0.parquet"))
        assert table["stars_luminosity"][0].as_py() == [42.0]
