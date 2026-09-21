from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

MORPHS_FILENAME = "morphs_r.hdf5"


class FitsDataset:
    """PyTorch Dataset for the Illustris SKIRT dataset in FITS format.

    Args:
        path (str): Path to the directory containing FITS files.
        columns (list[str] | None): List of columns to extract from the FITS files.

    The `sersic_n` column is read from the morphology catalog `morphs_r.hdf5`
    located in the snapshot directory of the FITS file, e.g.
    `TNG50/sdss/snapshot_095/morphs_r.hdf5` for
    `TNG50/sdss/snapshot_095/data/broadband_90.fits`.
    """

    def __init__(
        self,
        path: str,
        columns: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = sorted(self.path.rglob("*.fits"))
        self.columns = columns

        self._sersic_n_by_subhalo: dict[Path, dict[int, np.float32]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _sersic_n(self, fits_file: Path, subhalo_id: np.int32) -> np.float32:
        morphs_path = fits_file.parents[1] / MORPHS_FILENAME
        if morphs_path not in self._sersic_n_by_subhalo:
            with h5py.File(morphs_path, "r") as f:
                self._sersic_n_by_subhalo[morphs_path] = dict(zip(f["subfind_id"][:], f["sersic_n"][:]))
        return np.float32(self._sersic_n_by_subhalo[morphs_path][subhalo_id])

    def __getitem__(self, index: int) -> dict:
        fits_file = self.files[index]
        image = fits.getdata(fits_file, 0)
        image = np.array(image, dtype=np.float32)

        data: dict = {"image": image}
        if self.columns:
            splits = fits_file.parts
            subhalo_id = None
            if "subhalo_id" in self.columns or "sersic_n" in self.columns:
                subhalo_id = np.int32(splits[-1][: -len(".fits")].split("_")[1])
            for col in self.columns:
                if col == "simulation":
                    data["simulation"] = splits[-5]
                elif col == "snapshot":
                    data["snapshot"] = np.int32(splits[-3].split("_")[1])
                elif col == "subhalo_id":
                    data["subhalo_id"] = subhalo_id
                elif col == "sersic_n":
                    data["sersic_n"] = self._sersic_n(fits_file, subhalo_id)

        return data
