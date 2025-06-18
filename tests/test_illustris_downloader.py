import pytest

from pest import IllustrisDownloader, PropertyType, Selector


def test_get_subhalos_all():
    illustris_downloader = IllustrisDownloader()
    result = illustris_downloader.get_subhalos()

    assert result["count"] == 4371211
    assert len(result["results"]) == 100


def test_get_subhalos_limit():
    illustris_downloader = IllustrisDownloader()
    result = illustris_downloader.get_subhalos(limit=10)

    assert result["count"] == 4371211
    assert len(result["results"]) == 10


def test_selector_property_type():
    selector = Selector(property=PropertyType.MASS, min_value=1e10, max_value=1e11)
    assert selector.property == PropertyType.MASS


def test_get_subhalos_selector():
    illustris_downloader = IllustrisDownloader()
    mass_min = 10**11.99 / 1e10 * 0.704
    mass_max = 10**12.01 / 1e10 * 0.704
    result = illustris_downloader.get_subhalos(
        selector=Selector(property=PropertyType.MASS, min_value=mass_min, max_value=mass_max)
    )

    assert result["count"] == 94


def test_download_json():
    illustris_downloader = IllustrisDownloader()
    result = illustris_downloader.get(0)

    assert result is not None
    assert result["id"] == 0
    assert result["snap"] == 99
    assert result["len"] == 88772413


@pytest.mark.skip(reason="Skipping test case as per user request")
def test_download_hdf5(tmp_path):
    illustris_downloader = IllustrisDownloader(download_path=tmp_path)
    result = illustris_downloader.get_hdf5(354934)

    assert result == "cutout_354934.hdf5"
