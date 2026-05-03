import importlib

from datasets import Dataset


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


class ParquetWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def __call__(self, dataset: Dataset):
        dataset.to_parquet(self.output_path)
