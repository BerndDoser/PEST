from datasets import Array3D, Dataset, Features, Value

features = Features(
    {
        "image": Array3D(shape=(3, 128, 128), dtype="float32"),
        "simulation": Value("string"),
        "snapshot": Value("int32"),
        "subhalo_id": Value("int32"),
    }
)

dataset = Dataset.from_parquet("output/illustris_skirt_small.parquet", features=features)
dataset.push_to_hub("bernddoser/Illustris_TNG_SKIRT_SDSS", max_shard_size="500MB", private=True)
