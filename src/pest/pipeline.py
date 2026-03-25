"""Pipeline orchestrating extract → convert → generate for multiple generators."""

from typing import List

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
