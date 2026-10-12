from pathlib import Path

import pytest

from pest import apply_overrides, load_config, save_config

PIPELINES_DIR = Path(__file__).parent.parent / "pipelines"

CONFIG = {
    "num_workers": 16,
    "extract": {"class_path": "pest.PynbodyDataset", "init_args": {"snapshot_path": "/old/snap"}},
    "load": [{"class_path": "pest.ParquetWriter", "init_args": {"output_path": "output/old.parquet"}}],
}


def test_apply_overrides_sets_nested_keys_and_list_items():
    config = apply_overrides(
        CONFIG,
        {
            "num_workers": 1,
            "extract.init_args.snapshot_path": "/new/snap",
            "load.0.init_args.output_path": "/out/new.parquet",
        },
    )

    assert config["num_workers"] == 1
    assert config["extract"]["init_args"]["snapshot_path"] == "/new/snap"
    assert config["load"][0]["init_args"]["output_path"] == "/out/new.parquet"


def test_apply_overrides_does_not_mutate_input():
    apply_overrides(CONFIG, {"extract.init_args.snapshot_path": "/new/snap"})

    assert CONFIG["extract"]["init_args"]["snapshot_path"] == "/old/snap"


def test_apply_overrides_adds_new_leaf_key():
    config = apply_overrides(CONFIG, {"extract.init_args.halos_path": "/halos"})

    assert config["extract"]["init_args"]["halos_path"] == "/halos"


def test_apply_overrides_missing_intermediate_key():
    with pytest.raises(KeyError):
        apply_overrides(CONFIG, {"extract.init_arg.snapshot_path": "/new/snap"})


def test_apply_overrides_bad_list_index():
    with pytest.raises(IndexError):
        apply_overrides(CONFIG, {"load.1.init_args.output_path": "/out/new.parquet"})


def test_save_and_load_config_round_trip(tmp_path):
    path = save_config(CONFIG, tmp_path / "nested" / "config.yaml")

    assert path.exists()
    assert load_config(path) == CONFIG


@pytest.mark.parametrize("config_path", sorted(PIPELINES_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_pipeline_configs_write_parquet_first(config_path):
    """The Airflow DAGs override `load.0.init_args.output_path`, so it must be a ParquetWriter."""
    config = load_config(config_path)

    assert config["load"][0]["class_path"] == "pest.ParquetWriter"
    assert "output_path" in config["load"][0]["init_args"]
