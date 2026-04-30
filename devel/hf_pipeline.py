from pathlib import Path

import numpy as np
from astropy.io import fits
from datasets import Dataset


# 1. LOADER — replaces FitsDataset + extractor
def load_fits_records(fits_dir: str):
    files = sorted(Path(fits_dir).rglob("*.fits"))
    for f in files:
        parts = f.parts
        yield {
            "image": np.array(fits.getdata(f, 0), dtype=np.float32),
            "simulation": parts[-5],
            "snapshot": int(parts[-3].split("_")[1]),
            "subhalo_id": int(parts[-1][: -len(".fits")].split("_")[1]),
        }


ds = Dataset.from_generator(
    load_fits_records,
    gen_kwargs={"fits_dir": "/urz/gpuscratch/its/doserbd/data/SKIRT_synthetic_images/fits"},
)


# 2. TRANSFORMS — replaces the transformations list in the YAML
def normalize_rgb(batch):
    # your CreateNormalizedRGBColors logic here
    batch["image"] = [img / img.max() for img in batch["image"]]
    return batch


def remove_invalid(batch):
    valid = [img.sum() > 0 for img in batch["image"]]
    return {k: [v for v, ok in zip(vals, valid) if ok] for k, vals in batch.items()}


ds = ds.map(normalize_rgb, batched=True)
ds = ds.filter(lambda x: x["image"].sum() > 0)  # RemoveInvalidImages
# ... chain more .map() / .filter() calls

# 3. WRITER — replaces ParquetWriter
ds.to_parquet("output/illustris_skirt/data.parquet")
