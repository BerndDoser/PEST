from pest import FitsConverter

FitsConverter(
    image_size=128,
    datatype="png",
    chunk_size=10000,
).convert_all(
    [
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_099/data",
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_095/data",
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_099/data",
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_095/data",
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_135/data",
        "/hits/basement/its/doserbd/data/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_131/data",
    ],
    "/urz/gpuscratch/its/doserbd/data/SKIRT_synthetic_images/parquet-v5-128-png",
)
