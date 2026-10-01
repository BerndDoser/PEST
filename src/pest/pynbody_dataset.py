import re
from pathlib import Path

import numpy as np
import pynbody

COMPONENTS = {"stars": "st", "gas": "g", "dm": "dm"}
MASS_TYPES = {"stellar", "total"}
DEFAULT_COLUMNS = ["subhalo_id", "pos", "vel", "mass"]


def _in_units(array, units) -> np.ndarray:
    """Convert a unit-carrying SimArray to `units` and return a plain numpy array."""
    has_units = not isinstance(getattr(array, "units", pynbody.units.NoUnit()), pynbody.units.NoUnit)
    if has_units and units is not None and not isinstance(units, pynbody.units.NoUnit):
        array = array.in_units(units)
    return np.asarray(array)


def _find_catalogue(snapshot_path: str) -> str | None:
    """Locate an IllustrisTNG-style `groups_NNN/fof_subhalo_tab_NNN` catalogue for a snapshot.

    Handles snapshot paths like `.../snapdir_099/snap_099`, `.../snapdir_099/snap_099.0.hdf5`
    and `.../snap_099.hdf5`, which pynbody cannot map to the catalogue on its own.
    Returns the catalogue path without the `.N.hdf5` suffix, or `None` if not found.
    """
    path = Path(snapshot_path)
    match = re.search(r"_(\d+)(?:\.\d+)?(?:\.hdf5)?$", path.name)
    if match is None:
        return None
    num = match.group(1)
    for root in (path.parent.parent, path.parent):
        stem = root / f"groups_{num}" / f"fof_subhalo_tab_{num}"
        if stem.with_name(stem.name + ".hdf5").exists() or stem.with_name(stem.name + ".0.hdf5").exists():
            return str(stem)
    return None


class PynbodyDataset:
    """Dataset of subhalo particle data read with pynbody (e.g. IllustrisTNG snapshots).

    Subhalos are selected using only the halo catalogue metadata (masses and
    `SubhaloFlag`), so no particle data is read in `__init__`. The particles
    of a selected subhalo are loaded lazily in `__getitem__`.

    Args:
        snapshot_path (str): Path to the snapshot, e.g. `.../snapshot_099/snap_099`.
        halos_path (str | None): Path to the halo catalogue, e.g.
            `.../groups_099/fof_subhalo_tab_099`. If `None`, a sibling `groups_NNN`
            directory is searched, falling back to pynbody's own lookup.
        component (str): Particle family to return: "stars", "gas" or "dm".
        mass_type (str): Mass used for the selection: "stellar" (`SubhaloMassType[:, 4]`)
            or "total" (`SubhaloMass`).
        min_mass (float): Lower mass limit in Msun (exclusive).
        max_mass (float): Upper mass limit in Msun (exclusive).
        columns (list[str] | None): Columns to extract. Besides "subhalo_id" and
            "snapshot", "pos" and "vel" return the particle positions [kpc] and
            velocities [km/s] relative to the subhalo center, any other particle
            field (e.g. "mass") is returned as a float32 array, and any halo
            catalogue property (e.g. "SubhaloSFR") is returned for the subhalo.
            Defaults to `["subhalo_id", "pos", "vel", "mass"]`.
    """

    def __init__(
        self,
        snapshot_path: str,
        halos_path: str | None = None,
        component: str = "stars",
        mass_type: str = "stellar",
        min_mass: float = 0.0,
        max_mass: float = np.inf,
        columns: list[str] | None = None,
    ) -> None:
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {list(COMPONENTS)}, got {component!r}")
        if mass_type not in MASS_TYPES:
            raise ValueError(f"mass_type must be one of {sorted(MASS_TYPES)}, got {mass_type!r}")

        self.component = component
        self.columns = DEFAULT_COLUMNS if columns is None else columns

        self.snapshot = pynbody.load(snapshot_path)
        self.snapshot.physical_units()
        if halos_path is None:
            halos_path = _find_catalogue(snapshot_path)
        if halos_path is None:
            self.halos = self.snapshot.halos(subhalos=True)
        else:
            self.halos = self.snapshot.halos(filename=halos_path, subhalos=True)
        self.halos.physical_units()

        # Only the catalogue arrays are read here, no particle data.
        self.properties = self.halos.get_properties_all_halos()
        if mass_type == "stellar":
            mass = _in_units(self.properties["SubhaloMassType"], "Msol")[:, 4]
        else:
            mass = _in_units(self.properties["SubhaloMass"], "Msol")
        mask = (mass > min_mass) & (mass < max_mass)
        if "SubhaloFlag" in self.properties:
            mask &= np.asarray(self.properties["SubhaloFlag"]) == 1
        self.subhalo_ids = np.flatnonzero(mask)

        match = re.search(r"(\d+)$", Path(snapshot_path).name)
        self.snapshot_number = np.int32(match.group(1)) if match else None

    def __len__(self) -> int:
        return len(self.subhalo_ids)

    def __getitem__(self, index: int) -> dict:
        subhalo_id = self.subhalo_ids[index]

        data: dict = {}
        particles = None
        for col in self.columns:
            if col == "subhalo_id":
                data["subhalo_id"] = np.int32(subhalo_id)
            elif col == "snapshot":
                data["snapshot"] = self.snapshot_number
            elif col in self.properties:
                data[col] = np.asarray(self.properties[col][subhalo_id])
            else:
                if particles is None:
                    particles = getattr(self.halos[subhalo_id], COMPONENTS[self.component])
                data[col] = self._particle_field(particles, col, subhalo_id)
        return data

    def _particle_field(self, particles, field: str, subhalo_id: int) -> np.ndarray:
        values = particles[field]
        if field in ("pos", "vel"):
            center_key = "SubhaloPos" if field == "pos" else "SubhaloVel"
            center = _in_units(self.properties[center_key], getattr(values, "units", None))[subhalo_id]
            return (np.asarray(values) - center).astype(np.float32)
        return np.asarray(values, dtype=np.float32)
