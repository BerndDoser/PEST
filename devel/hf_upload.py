from datasets import Dataset

dataset = Dataset.from_parquet("/urz/gpuscratch/its/doserbd/data/SKIRT_synthetic_images/parquet-v4-128/0.parquet")
dataset.push_to_hub("bernddoser/illustris-skirt", max_shard_size="500MB")
