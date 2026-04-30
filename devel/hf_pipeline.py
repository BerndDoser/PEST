from pathlib import Path

import numpy as np
from astropy.io import fits
from datasets import Dataset

import pest

NUM_PROC = 10


# Load FITS files and create a Hugging Face Dataset
def load_fits_records(fits_dir: str):
    files = sorted(Path(fits_dir).rglob("*.fits"))
    for f in files:
        # parts = f.parts
        yield {
            "image": np.array(fits.getdata(f, 0), dtype=np.float32),
            # "simulation": parts[-5],
            # "snapshot": int(parts[-3].split("_")[1]),
            # "subhalo_id": int(parts[-1][: -len(".fits")].split("_")[1]),
        }


ds = Dataset.from_generator(
    load_fits_records,
    gen_kwargs={"fits_dir": "/home/bernd/data/TNG50-1/test"},
    num_proc=NUM_PROC,
)

# Normalize RGB colors
transform = pest.CreateNormalizedRGBColors()


def apply_normalized_rgb(batch):
    batch["image"] = [transform(np.array(img)) for img in batch["image"]]
    return batch


ds = ds.map(apply_normalized_rgb, batched=True, batch_size=10, num_proc=NUM_PROC)


def filter_unhealthy(batch):
    return not (
        np.isnan(batch["image"]).any()
        or np.isinf(batch["image"]).any()
        or np.all(np.array(batch["image"]) == np.array(batch["image"]).flat[0])
    )


# Filter out unhealthy data (NaN, Inf, or all pixels the same)
ds = ds.filter(filter_unhealthy, batched=False, num_proc=NUM_PROC)

# 3. Write to disk
ds.to_parquet("output/illustris_skirt/data.parquet")
