from pest import FitsDataset

EXPECTED_SERSIC_N = {90: 2.070126}


def test_fits_dataset_sersic_n():
    dataset = FitsDataset(
        path="tests/data/illustris_tng_skirt",
        columns=["image", "simulation", "snapshot", "subhalo_id", ("sersic_n", "r")],
    )

    assert len(dataset) == 1

    for item in dataset:
        assert item["simulation"] == "TNG50"
        assert item["snapshot"] == 95
        assert item[("sersic_n", "r")] == EXPECTED_SERSIC_N[int(item["subhalo_id"])]
