# Orchestration with Airflow

The `orchestration/` directory holds [Apache Airflow](https://airflow.apache.org/) DAGs that run
PEST pipelines as ETL workflows, plus a Podman Compose setup that runs Airflow locally. Airflow
adds parametrized runs, a separate output directory per run, retries for downloads, a check of
every output, and an optional upload to the Hugging Face Hub, all visible in the Airflow UI.

## DAGs

Both DAGs are triggered manually (they have no schedule), and each DAG runs one run at a time.

### `pest_illustris_skirt`

Downloads the Illustris/TNG SKIRT image tarballs (one task per URL), runs the
`pipelines/illustris_skirt.yaml` pipeline on them, checks the Parquet output and optionally pushes it
to the Hugging Face Hub.

```
prepare_run → resolve_config → download_skirt[url, ...] → check_inputs → run_pest
            → validate_parquet → hf_push_requested → push_to_hub
```

### `pest_pynbody_tng`

Runs the `pipelines/pynbody_tng.yaml` pipeline on an Illustris/TNG snapshot and its subhalo
catalogue.

```
prepare_run → resolve_config → check_inputs → run_pest → validate_parquet
            → hf_push_requested → push_to_hub
```

### Tasks

| Task | What it does |
|---|---|
| `prepare_run` | Creates the run directory and collects the config overrides from the params. |
| `resolve_config` | Applies the overrides to the config template and writes `config.yaml` into the run directory. |
| `download_skirt` | Downloads and extracts one SKIRT tarball into `data_dir`; does nothing if it was already extracted. |
| `check_inputs` | Fails early if the input files don't exist, e.g. because the data isn't mounted. |
| `run_pest` | Runs `pest config.yaml`, i.e. the whole extract, transform and load pipeline. |
| `validate_parquet` | Fails if the output has fewer than `min_rows` rows or lacks a `required_columns` column. |
| `hf_push_requested`, `push_to_hub` | Pushes the output to the Hugging Face Hub if `hf_repo_id` is set; skipped otherwise. |

Each run writes its resolved config next to its output:

```
output/<dag_id>/<run_id>/
├── config.yaml
└── illustris_skirt.parquet   (or pynbody_tng.parquet)
```

## Params

Both DAGs take these params:

| Param | Default | Meaning |
|---|---|---|
| `config_template` | `illustris_skirt.yaml` / `pynbody_tng.yaml` | Pipeline config, absolute or relative to `pipelines/`. |
| `num_workers` | `null` | PEST worker processes; `null` keeps the value from the config. |
| `overrides` | `{}` | Any other config change, as dotted keys; list items are indexed by number, e.g. `{"transform.0.init_args.min": 1e9}`. |
| `min_rows` | `1` | Minimum number of rows in the output. |
| `required_columns` | depends on the DAG | Columns the output must have. |
| `hf_repo_id` | `null` | Hugging Face dataset repo to push the output to. |

`pest_illustris_skirt` also takes:

| Param | Default | Meaning |
|---|---|---|
| `data_dir` | `illustris_skirt` | Where the SKIRT images are downloaded to and read from. |
| `skirt_urls` | the six TNG50, TNG100 and Illustris-1 SDSS tarballs | Tarballs to download; `[]` uses only the data already in `data_dir`. |

`pest_pynbody_tng` also takes:

| Param | Default | Meaning |
|---|---|---|
| `snapshot_path` | `TNG50-1/snapdir_099/snap_099` | Snapshot as passed to `pynbody.load`. |
| `halos_path` | `null` | Halo catalogue; `null` lets pynbody find it next to the snapshot. |

Relative input paths (`data_dir`, `snapshot_path`, `halos_path`) are relative to the data
directory.

## Running Airflow with Podman

The image is the official Airflow image with PEST installed in a separate virtual environment, so
PEST's dependencies don't have to fit Airflow's. It uses the LocalExecutor: the tasks, and so the
PEST pipelines, run inside the `airflow-scheduler` container, which therefore needs enough CPUs
and memory for the pipelines.

The setup is meant for rootless [Podman](https://podman.io/) with
[podman-compose](https://github.com/containers/podman-compose), e.g. `apt install podman
podman-compose` on Ubuntu 24.04. The Airflow containers map the image's `airflow` user to your own
user, so the files they write to the mounted directories belong to you (their group is one of your
sub-GIDs).

1. Create the settings file and fill it in:

   ```bash
   cd orchestration
   cp .env.example .env
   ```

   - `FERNET_KEY` and `AIRFLOW_JWT_SECRET`: generate them with the commands given in `.env.example`.
   - `PEST_HOST_DATA_DIR` and `PEST_HOST_OUTPUT_DIR`: host directories for the input data and the
     outputs (default: `data/` and `output/` in the repository).
   - `ILLUSTRIS_API_KEY`: needed to download the SKIRT images.
   - `HF_TOKEN`: needed to push to the Hugging Face Hub.

2. Build the image, initialize Airflow and start it:

   ```bash
   podman compose build
   podman compose run --rm airflow-init
   podman compose up -d
   ```

   podman-compose starts the containers without waiting for each other, so the database has to be
   initialized with the separate `run` step before the first `up`. podman-compose 1.0.6 then reports
   that the name of the Postgres container is already in use and starts the existing container
   instead; this is harmless.

3. Open the Airflow UI at <http://localhost:8080> and log in (default `airflow` / `airflow`, set in
   `.env`). Use *Trigger* on a DAG to edit its params and start a run.

Runs can also be triggered from the command line:

```bash
podman compose exec airflow-scheduler airflow dags trigger pest_pynbody_tng \
  --conf '{"snapshot_path": "TNG50-1/snapdir_099/snap_099", "num_workers": 16}'

podman compose exec airflow-scheduler airflow dags trigger pest_illustris_skirt \
  --conf '{"skirt_urls": [], "hf_repo_id": "my-org/illustris-skirt"}'
```

Stop Airflow with `podman compose down`; add `-v` to also delete its database.

### What is mounted

| Host | Container | Purpose |
|---|---|---|
| `orchestration/dags/` | `/opt/airflow/dags` | DAGs; changes are picked up without rebuilding. |
| `pipelines/` | `/opt/pest/pipelines` | Config templates; changes are picked up without rebuilding. |
| `PEST_HOST_DATA_DIR` | `/opt/airflow/data` | Input data and SKIRT downloads. |
| `PEST_HOST_OUTPUT_DIR` | `/opt/airflow/output` | Run outputs. |

Changes to the PEST code in `src/` need a new image and new containers:

```bash
podman compose build
podman compose down
podman compose up -d
```

### Settings

`PEST_POOL_SLOTS` (default 1) is the number of PEST pipelines that may run at the same time, across
all DAGs. PEST holds all records in memory until it writes them, so raise it only if the scheduler
container has the memory for several pipelines. `AIRFLOW_PARALLELISM` (default 8) is the number of
tasks Airflow runs at the same time.

The API keys are only passed to the `airflow-scheduler` container, as environment variables. They
are never part of the image, the DAGs or the run params.

## Developing the DAGs

The DAG tests need Airflow, which is in the `airflow` dependency group:

```bash
uv run --extra dev --extra hf --group airflow pytest tests/orchestration
```

The tests run both DAGs end to end on the test data in `tests/data`, using
[`dag.test()`](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/debug.html). Without
the `airflow` group, these tests are skipped.

The DAGs never import `pest` themselves. Tasks that need PEST run in the PEST environment, either
as a `pest` command (`@task.bash`) or with `@task.external_python`. Outside the container this environment
is the current Python interpreter. These environment variables configure the DAGs:

| Variable | Default | Meaning |
|---|---|---|
| `PEST_PYTHON` | the Airflow interpreter | Python interpreter of the PEST environment. |
| `PEST_BIN` | `pest` next to `PEST_PYTHON` | The `pest` command. |
| `PEST_POOL` | `default_pool` | Airflow pool of the `run_pest` tasks. |
| `PEST_PIPELINES_DIR` | `/opt/pest/pipelines` | Where relative `config_template` params are looked up. |
| `PEST_DATA_DIR` | `/opt/airflow/data` | Where relative input paths are looked up. |
| `PEST_OUTPUT_DIR` | `/opt/airflow/output` | Root of the run directories. |
