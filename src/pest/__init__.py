import importlib.metadata

from .count import Count
from .fits_converter import FitsConverter
from .fits_dataset import FitsDataset
from .gaia_converter import GaiaConverter
from .illustris_downloader import IllustrisDownloader, PropertyType, Selector
from .illustris_extractor import IllustrisExtractor
from .illustris_preprocess_api import data_preprocess_api
from .illustris_preprocess_local import data_preprocess_local
from .illustris_skirt_reader import IllustrisSkirtReader
from .orientation import (
    align_image_horizontally,
    crop_quadratic,
    estimate_geometry_weighted,
    reflectional_invariance,
    visualize_results,
)
from .pipeline import Pipeline
from .point_cloud_generator import PointCloudGenerator

__version__ = importlib.metadata.version("astro-pest")
__all__ = [
    "align_image_horizontally",
    "Count",
    "crop_quadratic",
    "data_preprocess_api",
    "data_preprocess_local",
    "estimate_geometry_weighted",
    "FitsConverter",
    "FitsDataset",
    "GaiaConverter",
    "IllustrisDownloader",
    "IllustrisExtractor",
    "IllustrisSkirtReader",
    "Pipeline",
    "PointCloudGenerator",
    "PropertyType",
    "reflectional_invariance",
    "Selector",
    "visualize_results",
]
