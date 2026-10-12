"""Load, override and save pipeline configs."""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Read a pipeline config from a YAML file."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _child(node, key: str):
    """Return the child of a dict or list node, indexing lists by integer key."""
    if isinstance(node, list):
        return node[int(key)]
    return node[key]


def apply_overrides(config: dict, overrides: Mapping[str, Any]) -> dict:
    """Return a copy of `config` with dotted-key overrides applied.

    Integer key segments index into lists, e.g. ``"load.0.init_args.output_path"``.
    All intermediate keys must already exist (`KeyError`/`IndexError` otherwise),
    so a typo can't silently add an unused setting. Only the last segment may add
    a new key to a dict, e.g. ``"extract.init_args.halos_path"``.
    """
    config = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        *parents, leaf = dotted_key.split(".")
        node = config
        for key in parents:
            node = _child(node, key)
        if isinstance(node, list):
            node[int(leaf)] = value
        elif isinstance(node, dict):
            node[leaf] = value
        else:
            raise KeyError(f"Cannot set {dotted_key!r}: {'.'.join(parents)!r} is not a dict or list.")
    return config


def save_config(config: dict, path: str | Path) -> Path:
    """Write a pipeline config to a YAML file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return path
