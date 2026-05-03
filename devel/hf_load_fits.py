import importlib

from datasets import Array3D, Dataset, Features, Value

class_path = "pest.FitsDataset"
init_args = {
    # "path": "tests/data/fits",
    "path": "data/fits",
    "columns": ["image", "simulation", "snapshot", "subhalo_id"],
}

features = Features(
    {
        "image": Array3D(shape=(4, 679, 679), dtype="float32"),
        "simulation": Value("string"),
        "snapshot": Value("int32"),
        "subhalo_id": Value("int32"),
    }
)


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


def load_records(class_path, init_args):
    dataset = _instantiate(class_path, init_args)
    for item in dataset:
        yield item


ds = Dataset.from_generator(
    load_records,
    gen_kwargs={
        "class_path": class_path,
        "init_args": init_args,
    },
    # features=features,
    num_proc=1,
)
