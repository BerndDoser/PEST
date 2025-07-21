from pest import FitsConverter

small_set_200 = "/home/doserbd/data/illustris/small_set_200/fits/TNG100/sdss/snapnum_099/data"
full_set = [
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_099/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_095/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_099/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_095/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_135/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_131/data",
]

for datatype, flatten in [("png", False), ("uint8", True), ("float32", True)]:
    for compression in ["none", "snappy", "gzip"]:
        FitsConverter(
            image_size=128,
            datatype=datatype,
            flatten=flatten,
            compression=compression,
        ).convert_all(
            small_set_200,
            small_set_200 + f"/../../../../../parquet-v4-128-{datatype}-{compression}",
        )
