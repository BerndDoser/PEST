from pathlib import Path

import numpy as np
from astropy.io import fits

from pest.create_normalized_rgb_colors import CreateNormalizedRGBColors


class FitsDatasetIter:
    """Iterator over FITS files in the Illustris SKIRT dataset.

    Lazily loads and normalizes each file, skipping invalid images
    (NaN, Inf, or constant).

    Args:
        path (str): Path to the directory containing FITS files.
    """

    def __init__(self, path: str):
        self.path = Path(path)

        self.normalize_rgb = CreateNormalizedRGBColors(
            stretch=0.9,
            range=5,
            lower_limit=0.001,
            channel_combinations=[[2, 3], [1, 0], [0]],
            scalers=[0.7, 0.5, 1.3],
        )

    def __iter__(self) -> np.ndarray:
        for fits_file in sorted(self.path.rglob("*.fits")):
            data = fits.getdata(fits_file, 0)
            data = np.array(data, dtype=np.float32)
            data = self.normalize_rgb(data)
            if np.isnan(data).any() or np.isinf(data).any() or np.all(data == data.flat[0]):
                continue

            splits = fits_file[: -len(".fits")].split("/")
            simulation = splits[-5]
            snapshot = np.int32(splits[-3].split("_")[1])
            subhalo_id = np.int32(splits[-1].split("_")[1])

            yield data
