from abc import ABC, abstractmethod


class Generator(ABC):
    """Base class for data generators."""

    def __init__(self):
        """Initialize the Converter."""

    @abstractmethod
    def __call__(
        self,
        input_file: str,
        output_file: str,
    ):
        """Convert a single file to Parquet format.

        Args:
            input_file (str): Path to the input file.
            output_file (str): Path to the output file.
        """
