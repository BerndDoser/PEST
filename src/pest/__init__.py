import importlib.metadata

from .extractor import Extractor
from .fits_converter import FitsConverter
from .gaia_converter import GaiaConverter
from .illustris_downloader import IllustrisDownloader, PropertyType, Selector
from .illustris_extractor import IllustrisExtractor
from .illustris_preprocess_api import data_preprocess_api
from .illustris_preprocess_local import data_preprocess_local

__version__ = importlib.metadata.version("astro-pest")
__all__ = [
    "data_preprocess_api",
    "data_preprocess_local",
    "Extractor",
    "FitsConverter",
    "GaiaConverter",
    "IllustrisDownloader",
    "IllustrisExtractor",
    "PropertyType",
    "Selector",
]
