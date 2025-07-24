"""Test the IllustrisExtractor implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pest.illustris_extractor import IllustrisExtractor


class TestIllustrisExtractor:
    """Test cases for IllustrisExtractor."""

    def test_initialization_valid_config(self):
        """Test successful initialization with valid configuration."""
        with patch.object(Path, "exists", return_value=True):
            extractor = IllustrisExtractor(
                simulation_path="/fake/path",
                simulation="TNG50",
                snapshot=99,
                objects="centrals",
                component=[
                    {
                        "name": "stars",
                        "fields": ["masses", "positions"],
                        "selector": {"type": "stellar mass", "min": 1e10, "max": 1e12},
                    }
                ],
            )

            assert extractor.simulation == "TNG50"
            assert extractor.snapshot == 99
            assert extractor.objects == "centrals"
            assert len(extractor.components) == 1

    def test_initialization_invalid_path(self):
        """Test initialization failure with invalid path."""
        with pytest.raises(ValueError):
            IllustrisExtractor(
                simulation_path="/nonexistent/path", simulation="TNG50", snapshot=99, objects="centrals", component=[]
            )

    def test_get_available_fields(self):
        """Test getting available fields for components."""
        with patch.object(Path, "exists", return_value=True):
            extractor = IllustrisExtractor(simulation_path="/fake/path", component=[])

            star_fields = extractor.get_available_fields("stars")
            assert "masses" in star_fields
            assert "positions" in star_fields
            assert "velocities" in star_fields

            gas_fields = extractor.get_available_fields("gas")
            assert "masses" in gas_fields
            assert "densities" in gas_fields

    def test_validate_configuration_invalid_component(self):
        """Test configuration validation with invalid component."""
        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError):
                IllustrisExtractor(
                    simulation_path="/fake/path", component=[{"name": "invalid_component", "fields": ["masses"]}]
                )

    def test_validate_configuration_invalid_field(self):
        """Test configuration validation with invalid field."""
        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError):
                IllustrisExtractor(
                    simulation_path="/fake/path", component=[{"name": "stars", "fields": ["invalid_field"]}]
                )

    def test_validate_configuration_invalid_selector(self):
        """Test configuration validation with invalid selector."""
        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError):
                IllustrisExtractor(
                    simulation_path="/fake/path",
                    component=[
                        {
                            "name": "stars",
                            "fields": ["masses"],
                            "selector": {"type": "invalid_selector", "min": 1e10, "max": 1e12},
                        }
                    ],
                )

    def test_repr(self):
        """Test string representation."""
        with patch.object(Path, "exists", return_value=True):
            extractor = IllustrisExtractor(
                simulation_path="/fake/path",
                simulation="TNG100",
                snapshot=50,
                objects="satellites",
                component=[{"name": "stars", "fields": ["masses"]}],
            )

            repr_str = repr(extractor)
            assert "IllustrisExtractor" in repr_str
            assert "TNG100" in repr_str
            assert "50" in repr_str
            assert "satellites" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
