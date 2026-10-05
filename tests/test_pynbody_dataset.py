import numpy as np
import pandas as pd
import pynbody
import pytest

import pest.pynbody_dataset
from pest import FilterRange, LoadParticles, Pipeline, PynbodyDataset, RenderStars

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


MASS_FILTER = {
    "class_path": "pest.FilterRange",
    "init_args": {"column": "SubhaloMassType", "index": 4, "min": 1e9, "max": 1e11, "units": "Msol"},
}
FLAG_FILTER = {"class_path": "pest.FilterRange", "init_args": {"column": "SubhaloFlag", "min": 1}}
LOAD_PARTICLES = {"class_path": "pest.LoadParticles", "init_args": {"fields": ["pos", "vel", "mass"]}}


def _run_pipeline(tmp_path, transform, columns=("subhalo_id", "snapshot", "SubhaloSFR")):
    output_path = tmp_path / "out.parquet"
    config = {
        "num_workers": 1,
        "shuffle": False,
        "extract": {
            "class_path": "pest.PynbodyDataset",
            "init_args": {"snapshot_path": "TNG50-1/snapshot_099/snap_099", "columns": list(columns)},
        },
        "transform": transform,
        "load": [{"class_path": "pest.ParquetWriter", "init_args": {"output_path": str(output_path)}}],
    }
    Pipeline(config).run()
    return pd.read_parquet(output_path)


def test_dataset_reads_catalogue_only(fake_pynbody):
    dataset = PynbodyDataset("TNG50-1/snapshot_099/snap_099", columns=["subhalo_id", "snapshot", "SubhaloSFR"])

    assert len(dataset) == 5
    item = dataset[3]
    assert item == {"subhalo_id": 3, "snapshot": 99, "SubhaloSFR": 3.0}
    assert fake_pynbody.fake_halos.accessed == []


def test_particle_columns_rejected(fake_pynbody):
    with pytest.raises(ValueError, match="LoadParticles"):
        PynbodyDataset("snap_099", columns=["subhalo_id", "pos"])


def test_particles(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    particles = dataset.particles(3, "stars", ["pos", "vel", "mass"])
    assert particles["pos"].shape == (4, 3)
    assert particles["pos"].dtype == np.float32
    np.testing.assert_allclose(particles["pos"], 1.0)
    np.testing.assert_allclose(particles["vel"], 2.0)
    np.testing.assert_allclose(particles["mass"], 1.0)
    assert fake_pynbody.fake_halos.accessed == [3]


def test_halo(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    halo = dataset.halo(np.int32(3))
    assert isinstance(halo, FakeHalo)
    assert len(halo.st["mass"]) == 4
    assert fake_pynbody.fake_halos.accessed == [3]


def test_invalid_component(fake_pynbody):
    dataset = PynbodyDataset("snap_099")
    with pytest.raises(ValueError):
        dataset.particles(0, "bh", ["pos"])


def test_mass_filter_on_table(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    stellar = FilterRange("SubhaloMassType", index=4, min=1e9, max=1e11).mask(dataset.table())
    total = FilterRange("SubhaloMass", min=1e11).mask(dataset.table())
    assert list(np.flatnonzero(stellar)) == [1, 2, 3]
    assert list(np.flatnonzero(total)) == [2, 3, 4]


def test_mass_filter_converts_units(fake_pynbody, monkeypatch):
    properties = fake_pynbody.fake_halos.get_properties_all_halos()
    properties["SubhaloMassType"] = pynbody.array.SimArray(properties["SubhaloMassType"] / 1e10, "1e10 Msol")
    monkeypatch.setattr(fake_pynbody.fake_halos, "get_properties_all_halos", lambda: properties)
    dataset = PynbodyDataset("snap_099")

    mask = FilterRange("SubhaloMassType", index=4, min=1e9, max=1e11, units="Msol").mask(dataset.table())
    assert list(np.flatnonzero(mask)) == [1, 2, 3]


def test_pipeline_loads_particles_only_for_selected(fake_pynbody, tmp_path):
    df = _run_pipeline(tmp_path, [MASS_FILTER, FLAG_FILTER, LOAD_PARTICLES])

    assert list(df["subhalo_id"]) == [1, 3]
    assert list(df["SubhaloSFR"]) == [1.0, 3.0]
    assert set(df.columns) == {"subhalo_id", "snapshot", "SubhaloSFR", "pos", "vel", "mass"}
    np.testing.assert_allclose(np.stack(df.iloc[1]["pos"]), np.ones((4, 3)))
    assert fake_pynbody.fake_halos.accessed == [1, 3]


def test_pipeline_filter_after_loading(fake_pynbody, tmp_path):
    late_filter = {"class_path": "pest.FilterRange", "init_args": {"column": "SubhaloSFR", "max": 2.0}}
    df = _run_pipeline(tmp_path, [MASS_FILTER, LOAD_PARTICLES, late_filter])

    # SubhaloFlag is not applied here, so subhalo 2 survives the mass filter.
    assert list(df["subhalo_id"]) == [1, 2]
    assert fake_pynbody.fake_halos.accessed == [1, 2, 3]


def test_load_particles_requires_binding():
    with pytest.raises(RuntimeError):
        LoadParticles(fields=["pos"])({"subhalo_id": 0})


class FakeCentering:
    def __init__(self, calls, halo, kwargs):
        self.calls = calls
        calls.append(("center", halo, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append(("revert",))


@pytest.fixture
def fake_render(monkeypatch):
    calls = []

    def render(halo, **kwargs):
        calls.append(("render", halo, kwargs))
        image = np.zeros((4, 4, 3))
        image[0] = 1.0  # first row = lowest y in pynbody
        return image

    monkeypatch.setattr(pynbody.analysis, "center", lambda halo, **kw: FakeCentering(calls, halo, kw))
    monkeypatch.setattr(pynbody.plot.stars, "render", render)
    return calls


def test_render_stars(fake_pynbody, fake_render):
    step = RenderStars(width="30 kpc", resolution=4, center_mode="ssc", render_args={"dynamic_range": 3.0})
    step.bind(PynbodyDataset("snap_099"))

    record = step({"subhalo_id": np.int32(3)})

    assert record["image"].shape == (4, 4, 3)
    assert record["image"].dtype == np.float32
    np.testing.assert_allclose(record["image"][-1], 1.0)
    assert [call[0] for call in fake_render] == ["center", "render", "revert"]
    _, centered_halo, center_kwargs = fake_render[0]
    _, rendered_halo, render_kwargs = fake_render[1]
    assert centered_halo is rendered_halo
    assert center_kwargs == {"mode": "ssc", "move_all": False}
    assert render_kwargs == {
        "width": "30 kpc",
        "resolution": 4,
        "noplot": True,
        "return_image": True,
        "dynamic_range": 3.0,
    }


def test_pipeline_renders_stars(fake_pynbody, fake_render, tmp_path):
    render = {"class_path": "pest.RenderStars", "init_args": {"resolution": 4}}
    df = _run_pipeline(tmp_path, [MASS_FILTER, FLAG_FILTER, render])

    assert list(df["subhalo_id"]) == [1, 3]
    assert np.stack([np.stack(row) for row in df.iloc[0]["image"]]).shape == (4, 4, 3)


def test_render_stars_requires_binding():
    with pytest.raises(RuntimeError):
        RenderStars()({"subhalo_id": 0})
