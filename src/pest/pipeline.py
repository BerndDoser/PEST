import argparse
import importlib
import importlib.metadata
import multiprocessing
import os
import subprocess
import time

import numpy as np
import yaml


def _git_commit() -> str:
    """Return the short git commit hash of the running source checkout, if available."""
    pkg_dir = os.path.dirname(__file__)
    try:
        # A non-editable install just copies this file into site-packages,
        # which may happen to sit inside some unrelated project's git
        # checkout. Only trust the result if this file is actually tracked
        # by the repo found there.
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", os.path.basename(__file__)],
            cwd=pkg_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pkg_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def print_header() -> None:
    """Print a header with the PEST version and git commit."""
    version = importlib.metadata.version("astro-pest")
    commit = _git_commit()
    title = f"PEST pipeline v{version} ({commit})"
    width = len(title) + 4
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


class _TransformStep:
    """A single transform (or filter) step.

    With a `column`, the transform receives and returns the value of that
    column. Without a column (`None`), it operates on the whole record: a
    filter receives the record and returns whether to keep it, any other
    transform receives the record and returns the (updated) record.
    """

    def __init__(self, column: str | None, transform):
        self.column = column
        self.transform = transform
        self.is_filter = getattr(transform, "is_filter", False)

    @property
    def name(self) -> str:
        return self.transform.__class__.__name__


def _build_steps(transform_cfgs: list[dict]) -> list[_TransformStep]:
    """Instantiate the transform chain.

    Each config entry is either a column-scoped group
    `{column: ..., transformations: [...]}` or a single record-level step
    `{class_path: ..., init_args: ...}`.
    """
    steps = []
    for cfg in transform_cfgs:
        if "class_path" in cfg:
            steps.append(_TransformStep(None, _instantiate(cfg["class_path"], cfg.get("init_args", {}))))
            continue
        for transform_cfg in cfg.get("transformations", []):
            transform = _instantiate(transform_cfg["class_path"], transform_cfg.get("init_args", {}))
            steps.append(_TransformStep(cfg["column"], transform))
    return steps


def _prefilter(dataset, steps: list[_TransformStep], dropped_counts: dict[str, int]) -> tuple[np.ndarray, int]:
    """Apply the leading vectorizable filters on the dataset table before extraction.

    Returns the indices of the surviving records and the number of leading
    steps that were applied here and must be skipped per record.
    """
    indices = np.arange(len(dataset))
    if not hasattr(dataset, "table"):
        return indices, 0

    table = None
    num_applied = 0
    for step in steps:
        if not (step.is_filter and step.column is None and hasattr(step.transform, "mask")):
            break
        if table is None:
            table = dataset.table()
        keep = np.asarray(step.transform.mask(table), dtype=bool)[indices]
        if not keep.all():
            dropped_counts[step.name] = dropped_counts.get(step.name, 0) + int((~keep).sum())
        indices = indices[keep]
        num_applied += 1
    return indices, num_applied


# Populated once per worker process by `_init_worker`, so the dataset and the
# transform chain are each built only once and then reused for every record
# that worker handles.
_worker_state: dict = {}


def _init_worker(extract_cfg: dict, transform_cfgs: list[dict], start_step: int = 0) -> None:
    """Pool initializer: build one dataset instance and one transform chain per worker.

    The first `start_step` steps were already applied by `_prefilter` and are skipped.
    """
    dataset = _instantiate(extract_cfg["class_path"], extract_cfg.get("init_args", {}))
    steps = _build_steps(transform_cfgs)[start_step:]
    for step in steps:
        if hasattr(step.transform, "bind"):
            step.transform.bind(dataset)
    _worker_state["dataset"] = dataset
    _worker_state["steps"] = steps


def _process_record(index: int) -> tuple[dict | None, str | None]:
    """Extract one record and run the full transform chain on it in a single pass.

    Returns ``(record, None)`` when the record survives, or ``(None, filter_name)``
    naming the filter step that dropped it.
    """
    dataset = _worker_state["dataset"]
    steps = _worker_state["steps"]
    record = dataset[index]

    for step in steps:
        try:
            if step.is_filter:
                if not step.transform(record):
                    return None, step.name
            elif step.column is None:
                record = step.transform(record)
            else:
                record[step.column] = step.transform(np.array(record[step.column]))
        except Exception as e:
            print(
                f"Transform {step.name} failed for "
                f"simulation={record.get('simulation')}, "
                f"snapshot={record.get('snapshot')}, "
                f"subhalo_id={record.get('subhalo_id')}: {e}"
            )
            raise
    return record, None


class Pipeline:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self.num_workers = config.get("num_workers", 1)
        self.batch_size = config.get("batch_size", 16)
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed", 42)

    def run(self) -> None:
        """Run the pipeline: extract, transform, and load data.

        Extraction and transformation are fused into a single pass per record:
        each worker loads one record and immediately runs the whole transform
        chain on it, instead of rewriting the full dataset once per step.
        """
        extract_cfg = self.config["extract"]
        transform_cfgs = self.config.get("transform", [])

        # Instantiated once here to get the record count and to apply the
        # leading vectorizable filters on the dataset table; each worker below
        # builds its own instance to avoid pickling/sharing file handles.
        dataset = _instantiate(extract_cfg["class_path"], extract_cfg.get("init_args", {}))
        num_records = len(dataset)
        dropped_counts: dict[str, int] = {}
        indices, start_step = _prefilter(dataset, _build_steps(transform_cfgs), dropped_counts)
        del dataset

        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)

        transform_start = time.perf_counter()
        if self.num_workers > 1:
            with multiprocessing.Pool(
                self.num_workers,
                initializer=_init_worker,
                initargs=(extract_cfg, transform_cfgs, start_step),
            ) as pool:
                results = list(pool.imap(_process_record, indices, chunksize=max(1, self.batch_size)))
        else:
            _init_worker(extract_cfg, transform_cfgs, start_step)
            results = [_process_record(i) for i in indices]

        records = []
        for record, dropped_by in results:
            if record is not None:
                records.append(record)
            else:
                dropped_counts[dropped_by] = dropped_counts.get(dropped_by, 0) + 1
        del results

        print(
            f"Extract+Transform: kept {len(records)}/{num_records} records "
            f"in {time.perf_counter() - transform_start:.2f}s"
        )
        for name, count in dropped_counts.items():
            print(f"  dropped by {name}: {count}")

        # Load
        load_start = time.perf_counter()
        load_cfgs = self.config.get("load", [])
        loads = [_instantiate(cfg["class_path"], cfg.get("init_args", {})) for cfg in load_cfgs]

        for load in loads:
            load(records)
        print(f"Load: {len(records)} records in {time.perf_counter() - load_start:.2f}s")


def main() -> None:
    """CLI entry point: read a YAML config file and run the pipeline."""

    print_header()

    parser = argparse.ArgumentParser(
        prog="pest",
        description="Preprocessing Engine for Spherinator Training",
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Pipeline(config).run()


if __name__ == "__main__":
    main()
