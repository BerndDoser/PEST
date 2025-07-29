from pest import FitsConverter

FitsConverter(
    image_size=128,
    datatype="png",
    flatten=False,
).convert_all(
    "/home/doserbd/data/illustris/small_set_200/fits/TNG100/sdss/snapnum_099/data",
    "/hits/basement/its/doserbd/data/SKIRT_synthetic_images_small/parquet-v4-128-png",
)
