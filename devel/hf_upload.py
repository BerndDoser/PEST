import numpy as np
from datasets import Array3D, Dataset, Features, Image, Value
from PIL import Image as PILImage

features = Features(
    {
        "image": Array3D(shape=(3, 128, 128), dtype="float32"),
        "simulation": Value("string"),
        "snapshot": Value("int32"),
        "subhalo_id": Value("int32"),
    }
)

dataset = Dataset.from_parquet("output/illustris_skirt.parquet", features=features)


def to_image(example):
    arr = np.array(example["image"])  # (3, 128, 128) float32
    arr = np.transpose(arr, (1, 2, 0))  # (128, 128, 3)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return {"image": PILImage.fromarray(arr)}


dataset = dataset.map(to_image, num_proc=32).cast_column("image", Image())
dataset.push_to_hub(
    "HITS-AIN/Illustris_TNG_SKIRT_SDSS",
    max_shard_size="500MB",
    private=False,
    token=open(".hf_hits_ain_write.txt").read().strip(),
)
