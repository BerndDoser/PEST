"""PyTorch Dataset for the Illustris SKIRT dataset in FITS format."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from pest.create_normalized_rgb_colors import CreateNormalizedRGBColors


class FitsDataset:
    """PyTorch Dataset for the Illustris SKIRT dataset in FITS format.

    Scans the given directory for FITS files at construction time, filters out
    invalid images (NaN, Inf, or constant), and provides indexed access.

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

        self.files = sorted(self.path.rglob("*.fits"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray | None:
        fits_file = self.files[index]
        data = fits.getdata(fits_file, 0)
        data = np.array(data, dtype=np.float32)
        data = self.normalize_rgb(data)
        if np.isnan(data).any() or np.isinf(data).any() or np.all(data == data.flat[0]):
            return None
        return data
