"""Settings, params and tasks shared by the PEST Airflow DAGs.

PEST runs in its own Python environment (`PEST_PYTHON`), separate from Airflow's.
Every task that needs PEST therefore runs as a subprocess, either via
`@task.bash` (the `pest` CLI) or via `@task.external_python`. This module and the
DAG files must never import `pest` themselves, and the functions decorated with
`@task.external_python` must be self-contained: only their source is shipped to
the PEST interpreter, so they have to import everything they use.

Environment variables (read when the DAGs are parsed):
    PEST_PYTHON: Python interpreter of the PEST environment (default: this interpreter).
    PEST_BIN: The `pest` CLI (default: `pest` next to `PEST_PYTHON`).
    PEST_POOL: Airflow pool for the pipeline runs (default: `default_pool`).

Environment variables (read when a task runs):
    PEST_PIPELINES_DIR: Where relative `config_template` params are looked up.
    PEST_DATA_DIR: Where relative input paths in the params are looked up.
    PEST_OUTPUT_DIR: Root of the per-run output directories.
"""

import glob
import os
import re
import shlex
import sys
from datetime import timedelta
from pathlib import Path

from airflow.sdk import Param, task

PEST_PYTHON = os.environ.get("PEST_PYTHON", sys.executable)
PEST_BIN = os.environ.get("PEST_BIN", str(Path(PEST_PYTHON).with_name("pest")))
PEST_POOL = os.environ.get("PEST_POOL", "default_pool")


def _env_dir(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def data_path(path: str) -> str:
    """Resolve an input path from the params; relative paths are taken relative to `PEST_DATA_DIR`."""
    return str(_env_dir("PEST_DATA_DIR", "/opt/airflow/data") / path)


def run_layout(context, template: str, parquet_name: str) -> dict:
    """Create the output directory of a DAG run and return its paths.

    Every run gets `PEST_OUTPUT_DIR/<dag_id>/<run_id>/` holding the resolved
    `config.yaml` next to the Parquet file it produced.
    """
    template_path = _env_dir("PEST_PIPELINES_DIR", "/opt/pest/pipelines") / template
    if not template_path.is_file():
        raise FileNotFoundError(f"Config template {template_path} not found.")

    # Run ids such as "manual__2026-10-12T10:00:00+00:00" contain characters that
    # are awkward in paths.
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", context["run_id"])
    run_dir = _env_dir("PEST_OUTPUT_DIR", "/opt/airflow/output") / context["ti"].dag_id / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": str(run_dir),
        "template": str(template_path),
        "config_path": str(run_dir / "config.yaml"),
        "output_path": str(run_dir / parquet_name),
    }


def pipeline_overrides(params, layout: dict, overrides: dict) -> dict:
    """Combine the config overrides of a run.

    The output path and `num_workers` come first, then the DAG's own
    `overrides`, and the user's `overrides` param last so it can change anything.
    """
    combined = {"load.0.init_args.output_path": layout["output_path"]}
    if params["num_workers"] is not None:
        combined["num_workers"] = params["num_workers"]
    return {**combined, **overrides, **params["overrides"]}


def common_params(template: str, required_columns: list[str]) -> dict:
    """Params shared by all PEST DAGs."""
    return {
        "config_template": Param(
            template,
            type="string",
            description="Pipeline YAML config, absolute or relative to PEST_PIPELINES_DIR.",
        ),
        "num_workers": Param(
            None,
            type=["null", "integer"],
            minimum=1,
            description="Number of PEST worker processes; null keeps the value from the config.",
        ),
        "overrides": Param(
            {},
            type="object",
            description='Extra config overrides as dotted keys, e.g. {"transform.0.init_args.min": 1e9}.',
        ),
        "min_rows": Param(1, type="integer", minimum=0, description="Fail if the output has fewer rows."),
        "required_columns": Param(
            required_columns,
            type="array",
            items={"type": "string"},
            description="Fail if the output lacks any of these columns.",
        ),
        "hf_repo_id": Param(
            None,
            type=["null", "string"],
            description="If set, push the output to this Hugging Face dataset repo (needs HF_TOKEN).",
        ),
    }


@task.external_python(python=PEST_PYTHON, expect_airflow=False)
def resolve_config(template: str, overrides: dict, config_path: str) -> str:
    """Write the config of this run: the template with the run's overrides applied."""
    import yaml

    from pest import apply_overrides, load_config, save_config

    config = apply_overrides(load_config(template), overrides)
    print(yaml.safe_dump(config, sort_keys=False))
    return str(save_config(config, config_path))


@task
def check_inputs(patterns: list[str]) -> None:
    """Fail early if an input glob matches nothing, e.g. because the data is not mounted."""
    missing = [pattern for pattern in patterns if next(glob.iglob(pattern, recursive=True), None) is None]
    if missing:
        raise FileNotFoundError(
            f"No input files match {missing}. Check the input paths in the params and the data volume mounts."
        )


@task.bash(pool=PEST_POOL, skip_on_exit_code=None, do_xcom_push=False, execution_timeout=timedelta(hours=48))
def run_pest(run_dir: str, config_path: str) -> str:
    """Run the PEST pipeline on the resolved config."""
    return f"cd {shlex.quote(run_dir)} && PYTHONUNBUFFERED=1 exec {shlex.quote(PEST_BIN)} {shlex.quote(config_path)}"


@task.external_python(python=PEST_PYTHON, expect_airflow=False)
def validate_parquet(path: str, min_rows: int, required_columns: list[str]) -> dict:
    """Check the row count and columns of the pipeline output."""
    from pest import validate_parquet as validate

    summary = validate(path, min_rows=min_rows, required_columns=required_columns)
    print(summary)
    return summary


@task.short_circuit
def hf_push_requested(hf_repo_id: str | None) -> bool:
    """Skip the Hugging Face upload unless a repo id was given."""
    return bool(hf_repo_id)


@task.external_python(python=PEST_PYTHON, expect_airflow=False, retries=2)
def push_to_hub(parquet_path: str, repo_id: str) -> None:
    """Push the pipeline output to the Hugging Face Hub; the token is taken from HF_TOKEN."""
    from pest import HuggingFaceWriter

    HuggingFaceWriter(parquet_path, repo_id)([])
