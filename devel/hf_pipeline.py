import numpy as np
from datasets import Dataset

import pest

NUM_PROC = 10


# Load FITS files and create a Hugging Face Dataset
def load_fits_records(fits_dir: str):
    dataset = pest.FitsDataset(fits_dir)
    for item in dataset:
        yield item


ds = Dataset.from_generator(
    load_fits_records,
    gen_kwargs={"fits_dir": "/home/bernd/data/TNG50-1/test"},
    num_proc=NUM_PROC,
)

transformations = [
    pest.CreateNormalizedRGBColors(),
    pest.FilterUnhealthyData(),
]

for transform in transformations:
    if getattr(transform, "is_filter", False):
        ds = ds.filter(transform, batched=False, num_proc=NUM_PROC)
    else:

        def apply(batch, t=transform):
            batch["image"] = [t(np.array(img)) for img in batch["image"]]
            return batch

        ds = ds.map(apply, batched=True, num_proc=NUM_PROC)

# Write to disk
ds.to_parquet("output/illustris_skirt/data.parquet")
