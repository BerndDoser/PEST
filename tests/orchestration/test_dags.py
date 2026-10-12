from pathlib import Path

import pandas as pd
import pytest

from pest.illustris_skirt_downloader import SKIRT_URLS

pytest.importorskip("airflow.sdk")

from airflow.sdk import DagRunState, TaskInstanceState

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TNG_SNAPSHOT = TEST_DATA_DIR / "illustris_tng" / "TNG50-1" / "snapdir_099" / "snap_099"


def _task_states(dag_run) -> dict[str, list]:
    states: dict[str, list] = {}
    for ti in dag_run.get_task_instances():
        states.setdefault(ti.task_id, []).append(ti.state)
    return states


def _output_files(output_dir, dag_id: str, name: str) -> list:
    return sorted((output_dir / dag_id).glob(f"*/{name}"))


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert {"pest_illustris_skirt", "pest_pynbody_tng"} <= set(dagbag.dag_ids)


@pytest.mark.parametrize(
    "dag_id, extra_tasks",
    [
        ("pest_illustris_skirt", {"list_skirt_urls", "download_skirt"}),
        ("pest_pynbody_tng", set()),
    ],
)
def test_dag_structure(dagbag, dag_id, extra_tasks):
    dag = dagbag.dags[dag_id]
    common_tasks = {
        "prepare_run",
        "resolve_config",
        "check_inputs",
        "run_pest",
        "validate_parquet",
        "hf_push_requested",
        "push_to_hub",
    }

    assert dag.schedule is None
    assert dag.max_active_runs == 1
    assert set(dag.task_ids) == common_tasks | extra_tasks
    assert {"check_inputs", "resolve_config"} <= dag.get_task("run_pest").upstream_task_ids
    assert "run_pest" in dag.get_task("validate_parquet").upstream_task_ids
    assert "hf_push_requested" in dag.get_task("push_to_hub").upstream_task_ids


def test_skirt_dag_urls_match_pest(dagbag):
    assert dagbag.dags["pest_illustris_skirt"].params["skirt_urls"] == SKIRT_URLS


def test_skirt_dag_runs_on_local_data(airflow_db, dagbag, tmp_path, monkeypatch):
    monkeypatch.setenv("PEST_OUTPUT_DIR", str(tmp_path))
    dag = dagbag.dags["pest_illustris_skirt"]

    dag_run = dag.test(
        run_conf={
            "config_template": str(TEST_DATA_DIR / "illustris_skirt_pipeline_1.yaml"),
            "data_dir": str(TEST_DATA_DIR / "illustris_tng_skirt"),
            "skirt_urls": [],
            "num_workers": 1,
        }
    )

    states = _task_states(dag_run)
    assert dag_run.state == DagRunState.SUCCESS, states
    assert states["download_skirt"] == [TaskInstanceState.SKIPPED]
    assert states["push_to_hub"] == [TaskInstanceState.SKIPPED]

    [output_path] = _output_files(tmp_path, "pest_illustris_skirt", "illustris_skirt.parquet")
    assert (output_path.parent / "config.yaml").exists()
    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert df.iloc[0]["simulation"] == "TNG50"
    image = df.iloc[0]["image"]
    assert (len(image), len(image[0]), len(image[0][0])) == (3, 32, 32)


def test_skirt_dag_maps_one_download_per_url(airflow_db, dagbag, tmp_path, monkeypatch):
    monkeypatch.setenv("PEST_OUTPUT_DIR", str(tmp_path))
    dag = dagbag.dags["pest_illustris_skirt"]

    dag_run = dag.test(
        run_conf={
            "config_template": str(TEST_DATA_DIR / "illustris_skirt_pipeline_1.yaml"),
            "data_dir": str(TEST_DATA_DIR / "illustris_tng_skirt"),
            "skirt_urls": SKIRT_URLS[:2],
            "num_workers": 1,
        },
        mark_success_pattern="download_skirt",
    )

    states = _task_states(dag_run)
    assert dag_run.state == DagRunState.SUCCESS, states
    assert states["download_skirt"] == [TaskInstanceState.SUCCESS] * 2


def test_pynbody_dag_runs_on_synthetic_tng(airflow_db, dagbag, tmp_path, monkeypatch):
    monkeypatch.setenv("PEST_OUTPUT_DIR", str(tmp_path))
    dag = dagbag.dags["pest_pynbody_tng"]

    dag_run = dag.test(
        run_conf={
            "snapshot_path": str(TNG_SNAPSHOT),
            "num_workers": 1,
            # The production stellar mass window selects none of the synthetic subhalos.
            "overrides": {"transform.0.init_args.min": 1.0e9, "transform.0.init_args.max": 1.0e12},
        }
    )

    states = _task_states(dag_run)
    assert dag_run.state == DagRunState.SUCCESS, states

    [output_path] = _output_files(tmp_path, "pest_pynbody_tng", "pynbody_tng.parquet")
    df = pd.read_parquet(output_path)
    assert len(df) >= 1
    assert {"subhalo_id", "snapshot", "SubhaloSFR", "pos", "vel", "mass"} <= set(df.columns)
    assert (df["snapshot"] == 99).all()


def test_pynbody_dag_fails_on_missing_snapshot(airflow_db, dagbag, tmp_path, monkeypatch):
    monkeypatch.setenv("PEST_OUTPUT_DIR", str(tmp_path))
    dag = dagbag.dags["pest_pynbody_tng"]

    dag_run = dag.test(run_conf={"snapshot_path": str(tmp_path / "missing" / "snap_099")})

    states = _task_states(dag_run)
    assert dag_run.state == DagRunState.FAILED
    assert states["check_inputs"] == [TaskInstanceState.FAILED]
    assert states["run_pest"] == [TaskInstanceState.UPSTREAM_FAILED]
