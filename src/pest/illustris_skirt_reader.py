"""Reader for the Illustris SKIRT dataset in FITS format."""

from pathlib import Path
from typing import Generator

import numpy as np
from astropy.io import fits

from pest.create_normalized_rgb_colors import CreateNormalizedRGBColors


class IllustrisSkirtReader:
    """Reader for the Illustris SKIRT dataset in FITS format."""

    def __init__(
        self,
        path: str,
    ):
        self.path = Path(path)

        self.normalize_rgb = CreateNormalizedRGBColors(
            stretch=0.9,
            range=5,
            lower_limit=0.001,
            channel_combinations=[[2, 3], [1, 0], [0]],
            scalers=[0.7, 0.5, 1.3],
        )

    def extract(self):
        """Iterate over all FITS files in the path and yield normalized image data.

        Yields:
            np.ndarray: Normalized RGB image array of shape (3, H, W) with values in [0, 1].
        """
        for fits_file in sorted(self.path.rglob("*.fits")):
            data = fits.getdata(fits_file, 0)
            data = np.array(data, dtype=np.float32)
            data = self.normalize_rgb(data)

            if np.isnan(data).any() or np.isinf(data).any() or np.all(data == data.flat[0]):
                continue

            yield data
