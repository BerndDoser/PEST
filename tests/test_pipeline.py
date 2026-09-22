from pathlib import Path

import pandas as pd
import yaml

from pest import Pipeline


def test_illustris_skirt_pipeline_1(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_1.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)

    Pipeline(config).run()

    assert output_path.exists(), "Parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert set(df.columns) == {"image", "simulation", "snapshot", "subhalo_id"}
    assert df.iloc[0]["simulation"] == "TNG50"
    assert df.iloc[0]["snapshot"] == 95

    image = df.iloc[0]["image"]
    assert len(image) == 3
    assert len(image[0]) == 32
    assert len(image[0][0]) == 32


def test_illustris_skirt_pipeline_2(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_2.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)

    Pipeline(config).run()

    assert output_path.exists(), "Parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert set(df.columns) == {"simulation", "snapshot", "subhalo_id"}
    assert df.iloc[0]["simulation"] == "TNG50"
    assert df.iloc[0]["snapshot"] == 95
