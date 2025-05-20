from pest import IllustrisDownloader


def test_download_json():
    illustris_downloader = IllustrisDownloader()
    result = illustris_downloader.get(0)

    assert result is not None
    assert result["id"] == 0
    assert result["snap"] == 99
    assert result["len"] == 88772413


def test_download_hdf5(tmp_path):
    illustris_downloader = IllustrisDownloader(download_path=tmp_path)
    result = illustris_downloader.get_hdf5(354934)

    assert result == "cutout_354934.hdf5"
