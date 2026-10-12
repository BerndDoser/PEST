"""
# PEST: pynbody TNG point clouds

Runs the `pynbody_tng.yaml` PEST pipeline on an Illustris/TNG snapshot and its
subhalo catalogue: selects subhalos by stellar mass and flag, loads their star
particles, writes them to a Parquet file, checks it and optionally pushes it to
the Hugging Face Hub.

Trigger it manually with the `snapshot_path` of the data. Each run writes its
resolved `config.yaml` and the Parquet file to
`PEST_OUTPUT_DIR/pest_pynbody_tng/<run_id>/`.
"""

import glob

from airflow.sdk import Param, dag, get_current_context, task
from pest_common import (
    check_inputs,
    common_params,
    data_path,
    hf_push_requested,
    pipeline_overrides,
    push_to_hub,
    resolve_config,
    run_layout,
    run_pest,
    validate_parquet,
)


@dag(
    dag_id="pest_pynbody_tng",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["pest"],
    doc_md=__doc__,
    params={
        **common_params("pynbody_tng.yaml", ["subhalo_id", "snapshot", "pos", "vel", "mass"]),
        "snapshot_path": Param(
            "TNG50-1/snapdir_099/snap_099",
            type="string",
            description="Snapshot path as passed to pynbody.load, absolute or relative to PEST_DATA_DIR.",
        ),
        "halos_path": Param(
            None,
            type=["null", "string"],
            description="Halo catalogue path; null lets pynbody find it next to the snapshot.",
        ),
    },
)
def pest_pynbody_tng():
    @task(multiple_outputs=True)
    def prepare_run() -> dict:
        context = get_current_context()
        params = context["params"]
        layout = run_layout(context, params["config_template"], "pynbody_tng.parquet")

        snapshot_path = data_path(params["snapshot_path"])
        overrides = {"extract.init_args.snapshot_path": snapshot_path}
        # Snapshots and catalogues may be split into several files, e.g. snap_099.0.hdf5.
        input_globs = [glob.escape(snapshot_path) + "*"]
        if params["halos_path"]:
            halos_path = data_path(params["halos_path"])
            overrides["extract.init_args.halos_path"] = halos_path
            input_globs.append(glob.escape(halos_path) + "*")

        return {
            **layout,
            "overrides": pipeline_overrides(params, layout, overrides),
            "input_globs": input_globs,
            "min_rows": params["min_rows"],
            "required_columns": params["required_columns"],
            "hf_repo_id": params["hf_repo_id"],
        }

    run = prepare_run()
    config_path = resolve_config(run["template"], run["overrides"], run["config_path"])
    summary = validate_parquet(run["output_path"], run["min_rows"], run["required_columns"])

    config_path >> check_inputs(run["input_globs"]) >> run_pest(run["run_dir"], config_path) >> summary
    summary >> hf_push_requested(run["hf_repo_id"]) >> push_to_hub(run["output_path"], run["hf_repo_id"])


pest_pynbody_tng()
