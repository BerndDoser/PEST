from abc import ABC, abstractmethod
from typing import Any, Dict


class Generator(ABC):
    """Base class for data generators."""

    def __init__(self):
        """Initialize the Converter."""

    @abstractmethod
    def __call__(
        self,
        input_data: Dict[str, Any],
        output_directory: str,
    ):
        """Generate Parquet dataset from input data.

        Args:
            input_data (Dict[str, Any]): Input data to convert.
            output_directory (str): Directory to save the output files.
        """
