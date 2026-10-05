import re
from pathlib import Path

import numpy as np
import pynbody

from .units import in_units

COMPONENTS = {"stars": "st", "gas": "g", "dm": "dm"}
# pynbody tries the generic Arepo catalogue first, which reads the wrong multi-file header for TNG.
TNG_CATALOGUE = ["TNGSubfindHDFCatalogue"]
DEFAULT_COLUMNS = ["subhalo_id"]
PARTICLE_FIELDS = {"pos", "vel", "mass"}


class PynbodyDataset:
    """Dataset of subhalos read with pynbody (e.g. IllustrisTNG snapshots).

    One record per subhalo of the halo catalogue, with the record index being
    the subhalo id. Only halo catalogue data is read here, so records are
    cheap. The selection of subhalos (e.g. a mass range) is done with filters
    in the transform stage, which can use `table()` to drop subhalos before
    any record is extracted. Particle data is added afterwards by a
    `pest.LoadParticles` transform step, using `particles()`.

    Args:
        snapshot_path (str): Path to the snapshot, e.g. `.../snapshot_099/snap_099`.
        halos_path (str | None): Path to the halo catalogue. If `None`, pynbody
            locates it next to the snapshot.
        columns (list[str] | None): Columns to extract: "subhalo_id", "snapshot",
            or any halo catalogue property (e.g. "SubhaloSFR").
            Defaults to `["subhalo_id"]`.
    """

    def __init__(
        self,
        snapshot_path: str,
        halos_path: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self.columns = DEFAULT_COLUMNS if columns is None else columns
        particle_columns = PARTICLE_FIELDS.intersection(self.columns)
        if particle_columns:
            raise ValueError(
                f"Particle fields {sorted(particle_columns)} are not dataset columns, "
                "load them with a pest.LoadParticles transform step."
            )

        self.snapshot = pynbody.load(snapshot_path)
        self.snapshot.physical_units()
        if halos_path is None:
            self.halos = self.snapshot.halos(subhalos=True, priority=TNG_CATALOGUE)
        else:
            self.halos = self.snapshot.halos(filename=halos_path, subhalos=True, priority=TNG_CATALOGUE)
        self.halos.physical_units()

        # Only the catalogue arrays are read here, no particle data.
        self.properties = self.halos.get_properties_all_halos()
        self.num_subhalos = len(next(iter(self.properties.values())))

        match = re.search(r"(\d+)$", Path(snapshot_path).name)
        self.snapshot_number = np.int32(match.group(1)) if match else None

    def __len__(self) -> int:
        return self.num_subhalos

    def table(self) -> dict:
        """Halo catalogue properties of all subhalos, indexed like the records."""
        return self.properties

    def __getitem__(self, index: int) -> dict:
        data: dict = {}
        for col in self.columns:
            if col == "subhalo_id":
                data["subhalo_id"] = np.int32(index)
            elif col == "snapshot":
                data["snapshot"] = self.snapshot_number
            else:
                data[col] = np.asarray(self.properties[col][index])
        return data

    def halo(self, subhalo_id: int | np.integer):
        """The pynbody halo object of one subhalo, giving access to all its particles."""
        return self.halos[int(subhalo_id)]

    def particles(self, subhalo_id: int, component: str, fields: list[str]) -> dict[str, np.ndarray]:
        """Read particle `fields` of one subhalo.

        "pos" and "vel" are returned relative to the subhalo center in kpc and
        km/s, any other particle field (e.g. "mass") as a float32 array.
        """
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {list(COMPONENTS)}, got {component!r}")
        particles = getattr(self.halo(subhalo_id), COMPONENTS[component])
        return {field: self._particle_field(particles, field, subhalo_id) for field in fields}

    def _particle_field(self, particles, field: str, subhalo_id: int) -> np.ndarray:
        values = particles[field]
        if field in ("pos", "vel"):
            center_key = "SubhaloPos" if field == "pos" else "SubhaloVel"
            center = in_units(self.properties[center_key], getattr(values, "units", None))[subhalo_id]
            return (np.asarray(values) - center).astype(np.float32)
        return np.asarray(values, dtype=np.float32)
