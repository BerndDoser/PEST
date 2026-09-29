import numpy as np
import pynbody
import pytest

import pest.pynbody_dataset
from pest import PynbodyDataset

# 5 subhalos: stellar masses [Msun] and flags
STELLAR_MASS = np.array([1e8, 5e9, 2e10, 3e10, 1e12])
SUBHALO_FLAG = np.array([1, 1, 0, 1, 1])


class FakeParticles(dict):
    pass


class FakeHalo:
    def __init__(self, subhalo_id):
        n = subhalo_id + 1
        pos = np.full((n, 3), 10.0 * subhalo_id + 1.0)
        self.st = FakeParticles(pos=pos, vel=pos * 2, mass=np.ones(n))


class FakeHalos:
    def __init__(self):
        self.accessed = []

    def physical_units(self):
        pass

    def get_properties_all_halos(self):
        n = len(STELLAR_MASS)
        mass_type = np.zeros((n, 6))
        mass_type[:, 4] = STELLAR_MASS
        pos = 10.0 * np.arange(n)[:, None] * np.ones((1, 3))
        return {
            "SubhaloMassType": mass_type,
            "SubhaloMass": 10 * STELLAR_MASS,
            "SubhaloFlag": SUBHALO_FLAG,
            "SubhaloPos": pos,
            "SubhaloVel": 2 * pos,
            "SubhaloSFR": np.arange(n, dtype=float),
        }

    def __getitem__(self, subhalo_id):
        self.accessed.append(int(subhalo_id))
        return FakeHalo(int(subhalo_id))


class FakeSnapshot:
    def __init__(self):
        self.fake_halos = FakeHalos()

    def physical_units(self):
        pass

    def halos(self, **kwargs):
        return self.fake_halos


@pytest.fixture
def fake_pynbody(monkeypatch):
    snapshot = FakeSnapshot()
    monkeypatch.setattr(pest.pynbody_dataset.pynbody, "load", lambda path: snapshot)
    return snapshot


def test_selection_uses_metadata_only(fake_pynbody):
    dataset = PynbodyDataset("TNG50-1/snapshot_099/snap_099", min_mass=1e9, max_mass=1e11)

    assert list(dataset.subhalo_ids) == [1, 3]
    assert len(dataset) == 2
    assert fake_pynbody.fake_halos.accessed == []


def test_total_mass_selection(fake_pynbody):
    dataset = PynbodyDataset("snap_099", mass_type="total", min_mass=1e11)

    assert list(dataset.subhalo_ids) == [3, 4]


def test_getitem(fake_pynbody):
    dataset = PynbodyDataset(
        "TNG50-1/snapshot_099/snap_099",
        min_mass=1e9,
        max_mass=1e11,
        columns=["subhalo_id", "snapshot", "pos", "vel", "mass", "SubhaloSFR"],
    )

    item = dataset[1]
    assert item["subhalo_id"] == 3
    assert item["snapshot"] == 99
    assert item["pos"].shape == (4, 3)
    assert item["pos"].dtype == np.float32
    np.testing.assert_allclose(item["pos"], 1.0)
    np.testing.assert_allclose(item["vel"], 2.0)
    np.testing.assert_allclose(item["mass"], 1.0)
    assert item["SubhaloSFR"] == 3.0
    assert fake_pynbody.fake_halos.accessed == [3]


def test_catalog_columns_do_not_read_particles(fake_pynbody):
    dataset = PynbodyDataset("snap_099", columns=["subhalo_id", "SubhaloSFR"])

    for item in dataset:
        assert set(item) == {"subhalo_id", "SubhaloSFR"}
    assert fake_pynbody.fake_halos.accessed == []


def test_invalid_component(fake_pynbody):
    with pytest.raises(ValueError):
        PynbodyDataset("snap_099", component="bh")


def test_mass_selection_converts_units(fake_pynbody, monkeypatch):
    properties = fake_pynbody.fake_halos.get_properties_all_halos()
    properties["SubhaloMassType"] = pynbody.array.SimArray(properties["SubhaloMassType"] / 1e10, "1e10 Msol")
    monkeypatch.setattr(fake_pynbody.fake_halos, "get_properties_all_halos", lambda: properties)

    dataset = PynbodyDataset("snap_099", min_mass=1e9, max_mass=1e11)

    assert list(dataset.subhalo_ids) == [1, 3]
