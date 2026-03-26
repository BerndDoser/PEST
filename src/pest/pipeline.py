"""Pipeline orchestrating extract → convert → generate for multiple generators."""

import argparse
import importlib
from typing import List

import yaml

from .extractor import Extractor
from .generator import Generator


class Pipeline:
    """Orchestrates extraction, conversion, and generation for one or more generators.

    The same extraction pass feeds every registered generator, so large simulations
    are only read from disk once regardless of how many output formats are requested.
    """

    def __init__(self, extractor: Extractor, generators: List[Generator]):
        """Initialize the Pipeline.

        Args:
            extractor: An Extractor instance (e.g. IllustrisExtractor) used to
                produce raw particle data.
            generators: One or more Generator instances (e.g. PointCloudGenerator)
                that will receive each converted Galaxy object.
        """
        self.extractor = extractor
        self.generators = generators

    def run(self) -> None:
        """Run the full extract → convert → generate pipeline.

        Calls ``extractor.extract()``, converts the result to Galaxy objects via
        ``build_galaxies()``, and feeds each Galaxy to every registered generator.
        Each generator's ``close()`` is called in a ``finally`` block so output
        files are properly finalised even if an error occurs mid-run.
        """
        try:
            for galaxy in self.extractor.extract():
                for generator in self.generators:
                    generator.add_galaxy(galaxy)
        finally:
            for generator in self.generators:
                generator.close()


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


def main() -> None:
    """CLI entry point: read a YAML config file and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="pest",
        description="Preprocessing Engine for Spherinator Training",
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    source_cfg = config["source"]
    extractor = _instantiate(source_cfg["class_path"], source_cfg.get("init_args", {}))

    target_cfg = config["target"]
    generator = _instantiate(target_cfg["class_path"], target_cfg.get("init_args", {}))

    Pipeline(extractor, [generator]).run()


if __name__ == "__main__":
    main()
