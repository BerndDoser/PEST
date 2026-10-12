"""Airflow test environment for the DAGs in orchestration/dags.

Airflow reads its configuration on import, so the environment is set up here,
before any test module imports it. Without the `airflow` dependency group
installed, the tests in this directory are skipped.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
DAGS_DIR = REPO_DIR / "orchestration" / "dags"

_airflow_home = tempfile.mkdtemp(prefix="pest-airflow-")
os.environ["AIRFLOW_HOME"] = _airflow_home
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(DAGS_DIR)
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = f"sqlite:///{_airflow_home}/airflow.db"
# Run PEST in the test interpreter, with the configs from the repo.
os.environ["PEST_PYTHON"] = sys.executable
os.environ["PEST_PIPELINES_DIR"] = str(REPO_DIR / "pipelines")


@pytest.fixture(scope="session")
def airflow_db():
    from airflow.utils.db import initdb

    initdb()


@pytest.fixture(scope="session")
def dagbag():
    from airflow.dag_processing.dagbag import BundleDagBag

    return BundleDagBag(dag_folder=DAGS_DIR, bundle_path=DAGS_DIR)
