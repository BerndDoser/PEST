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

FitsConverter(
    image_size=128,
    datatype="png",
    flatten=False,
).convert_all(
    small_set_200,
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images_small/parquet-v4-128-png",
)
