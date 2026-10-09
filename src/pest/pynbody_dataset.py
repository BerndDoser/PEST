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
# Catalogue properties holding the (position, velocity) of the subhalo center, per catalogue format.
CENTER_PROPERTIES = [("SubhaloPos", "SubhaloVel"), ("sub_pos", "sub_vel")]


class PynbodyDataset:
    """Dataset of (sub)halos of any snapshot and halo catalogue that pynbody can read.

    One record per halo of the catalogue, e.g. IllustrisTNG, Arepo or Gadget-4
    SUBFIND subhalos, EAGLE FoF groups, AHF or AdaptaHOP halos. Catalogues that
    can list all subhalos (SUBFIND) are loaded as subhalo catalogues, all others
    with their default halos. The record index is the position in the
    catalogue, and the "subhalo_id" column is pynbody's halo number, which can
    differ from it (e.g. EAGLE group numbers start at 1).

    Only halo catalogue data is read here, so records are cheap. The selection
    of subhalos (e.g. a mass range) is done with filters in the transform stage,
    which can use `table()` to drop subhalos before any record is extracted.
    Particle data is added afterwards by a `pest.LoadParticles` transform step,
    using `particles()`.

    Args:
        snapshot_path (str): Path to the snapshot, e.g. `.../snapshot_099/snap_099`.
        halos_path (str | None): Path to the halo catalogue. If `None`, pynbody
            locates it next to the snapshot.
        columns (list[str] | None): Columns to extract: "subhalo_id", "snapshot",
            or any halo catalogue property (e.g. "SubhaloSFR"). Which properties
            exist depends on the catalogue format; some (e.g. EAGLE, AdaptaHOP)
            provide none. Defaults to `["subhalo_id"]`.
        halos_args (dict | None): Keyword arguments for pynbody's `halos()`,
            replacing the automatic catalogue selection, e.g.
            `{"priority": ["AHFCatalogue"]}`.
    """

    def __init__(
        self,
        snapshot_path: str,
        halos_path: str | None = None,
        columns: list[str] | None = None,
        halos_args: dict | None = None,
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
        self.halos = self._load_halos(halos_path, halos_args)
        self.halos.physical_units()
        self.halo_numbers = np.asarray(self.halos.keys())

        # Only the catalogue arrays are read here, no particle data.
        self.properties = self.halos.get_properties_all_halos()
        missing = [col for col in self.columns if col not in ("subhalo_id", "snapshot", *self.properties)]
        if missing:
            raise KeyError(f"Columns {missing} are not in the halo catalogue, available: {sorted(self.properties)}")
        self.center_properties = next(
            (keys for keys in CENTER_PROPERTIES if all(key in self.properties for key in keys)), None
        )
        # Formats without partial loading (e.g. Gadget binary, Ramses) fall back to copying out of the snapshot.
        self.partial_loading = True

        name = Path(snapshot_path).name
        match = re.search(r"snap(?:shot)?_(\d+)", name) or re.search(r"(\d+)$", name.removesuffix(".hdf5"))
        self.snapshot_number = np.int32(match.group(1)) if match else None

    def _load_halos(self, halos_path: str | None, halos_args: dict | None):
        kwargs = {} if halos_path is None else {"filename": halos_path}
        if halos_args is not None:
            return self.snapshot.halos(**kwargs, **halos_args)
        try:
            return self.snapshot.halos(subhalos=True, priority=TNG_CATALOGUE, **kwargs)
        except (TypeError, RuntimeError):
            # Catalogues without a flat list of subhalos (e.g. EAGLE, AHF, AdaptaHOP) reject `subhalos`.
            return self.snapshot.halos(**kwargs)

    def __len__(self) -> int:
        return len(self.halo_numbers)

    def table(self) -> dict:
        """Halo catalogue properties of all subhalos, indexed like the records."""
        return self.properties

    def __getitem__(self, index: int) -> dict:
        data: dict = {}
        for col in self.columns:
            if col == "subhalo_id":
                data["subhalo_id"] = np.int32(self.halo_numbers[index])
            elif col == "snapshot":
                data["snapshot"] = self.snapshot_number
            else:
                data[col] = np.asarray(self.properties[col][index])
        return data

    def halo(self, subhalo_id: int | np.integer):
        """A standalone pynbody snapshot holding only the particles of one subhalo.

        The particles are read from disk with partial loading where the format
        supports it, otherwise copied out of the snapshot. Indexing the
        catalogue instead gives a view on the full snapshot, where loading any
        array reads it for all particles of the simulation, and where moving
        the particles (e.g. centering) moves the whole snapshot.
        """
        subhalo_id = int(subhalo_id)
        if self.partial_loading:
            try:
                halo = self.halos.load_copy(subhalo_id)
            except (TypeError, NotImplementedError):
                self.partial_loading = False
        if not self.partial_loading:
            halo = self.halos[subhalo_id].get_copy_on_access_simsnap()
        halo.physical_units()
        return halo

    def particles(self, subhalo_id: int, component: str, fields: list[str]) -> dict[str, np.ndarray]:
        """Read particle `fields` of one subhalo.

        "pos" and "vel" are returned relative to the subhalo center in kpc and
        km/s, any other particle field (e.g. "mass") as a float32 array. The
        center is taken from the catalogue, or computed with
        `pynbody.analysis.center` if the catalogue has none.
        """
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {list(COMPONENTS)}, got {component!r}")
        halo = self.halo(subhalo_id)
        if self.center_properties is None:
            pynbody.analysis.center(halo)
        particles = getattr(halo, COMPONENTS[component])
        return {field: self._particle_field(particles, field, subhalo_id) for field in fields}

    def _particle_field(self, particles, field: str, subhalo_id: int) -> np.ndarray:
        values = particles[field]
        if field in ("pos", "vel") and self.center_properties is not None:
            center_key = self.center_properties[0 if field == "pos" else 1]
            index = self.halos.number_mapper.number_to_index(int(subhalo_id))
            center = in_units(self.properties[center_key], getattr(values, "units", None))[index]
            offset = np.asarray(values) - center
            boxsize = self._boxsize(values) if field == "pos" else None
            if boxsize is not None:
                # Halos crossing the boundary of a periodic box.
                offset = (offset + boxsize / 2) % boxsize - boxsize / 2
            return offset.astype(np.float32)
        return np.asarray(values, dtype=np.float32)

    def _boxsize(self, values) -> float | None:
        boxsize = self.snapshot.properties.get("boxsize")
        units = getattr(values, "units", None)
        if boxsize is None or units is None:
            return None
        return float(boxsize.in_units(units, **self.snapshot.conversion_context()))
