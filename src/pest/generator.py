from abc import ABC, abstractmethod

from pest.galaxy import Galaxy


class Generator(ABC):
    """Base class for data generators."""

    def __init__(self):
        """Initialize the Converter."""

    @abstractmethod
    def add_galaxy(
        self,
        galaxy: Galaxy,
    ):
        """Add a galaxy to the dataset.

        Args:
            galaxy (Galaxy): The galaxy to add.
        """
