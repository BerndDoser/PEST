"""
# PEST: Illustris/TNG SKIRT images

Downloads the SKIRT synthetic image tarballs (one task per URL; a tarball that
was already extracted is skipped), runs the `illustris_skirt.yaml` PEST pipeline
on them, checks the resulting Parquet file and optionally pushes it to the
Hugging Face Hub.

Trigger it manually and adjust the params, e.g. `skirt_urls: []` to only process
data that is already in `data_dir`. Each run writes its resolved `config.yaml`
and the Parquet file to `PEST_OUTPUT_DIR/pest_illustris_skirt/<run_id>/`.
"""

import glob
import os

from airflow.sdk import Param, TriggerRule, dag, get_current_context, task
from pest_common import (
    PEST_PYTHON,
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

# Copy of `pest.illustris_skirt_downloader.SKIRT_URLS` (a test keeps them in sync),
# so that parsing this DAG does not need to import pest.
SKIRT_URLS = [
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.131.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.135.tar",
]


@task.external_python(
    python=PEST_PYTHON,
    expect_airflow=False,
    retries=3,
    retry_exponential_backoff=True,
    max_active_tis_per_dag=2,
)
def download_skirt(url: str, data_dir: str) -> str:
    """Download and extract one SKIRT tarball; the API key is taken from ILLUSTRIS_API_KEY."""
    from pest import download_and_extract

    return str(download_and_extract(url, data_dir))


@dag(
    dag_id="pest_illustris_skirt",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["pest"],
    doc_md=__doc__,
    params={
        **common_params("illustris_skirt.yaml", ["image", "simulation", "snapshot", "subhalo_id"]),
        "data_dir": Param(
            "illustris_skirt",
            type="string",
            description="Directory the SKIRT images are downloaded to and read from, absolute or relative to "
            "PEST_DATA_DIR.",
        ),
        "skirt_urls": Param(
            SKIRT_URLS,
            type="array",
            items={"type": "string"},
            description="SKIRT tarballs to download; empty to only use the data already in data_dir.",
        ),
    },
)
def pest_illustris_skirt():
    @task(multiple_outputs=True)
    def prepare_run() -> dict:
        context = get_current_context()
        params = context["params"]
        layout = run_layout(context, params["config_template"], "illustris_skirt.parquet")
        data_dir = data_path(params["data_dir"])
        return {
            **layout,
            "overrides": pipeline_overrides(params, layout, {"extract.init_args.path": data_dir}),
            "data_dir": data_dir,
            "input_globs": [os.path.join(glob.escape(data_dir), "**", "*.fits")],
            "min_rows": params["min_rows"],
            "required_columns": params["required_columns"],
            "hf_repo_id": params["hf_repo_id"],
        }

    # Mapping needs the plain return value of a task, not one key of a multi-output task.
    @task
    def list_skirt_urls() -> list[str]:
        return get_current_context()["params"]["skirt_urls"]

    run = prepare_run()
    # The config is resolved before downloading, so that bad params fail before a long download.
    config_path = resolve_config(run["template"], run["overrides"], run["config_path"])
    downloads = download_skirt.partial(data_dir=run["data_dir"]).expand(url=list_skirt_urls())
    # With no URLs the download task is skipped; carry on with the data already present.
    inputs = check_inputs.override(trigger_rule=TriggerRule.NONE_FAILED)(run["input_globs"])
    summary = validate_parquet(run["output_path"], run["min_rows"], run["required_columns"])

    config_path >> downloads >> inputs >> run_pest(run["run_dir"], config_path) >> summary
    summary >> hf_push_requested(run["hf_repo_id"]) >> push_to_hub(run["output_path"], run["hf_repo_id"])


pest_illustris_skirt()
