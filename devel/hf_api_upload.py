from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/urz/gpuscratch/its/doserbd/data/SKIRT_synthetic_images/parquet-v4-128",
    repo_id="bernddoser/illustris-skirt",
    repo_type="dataset",
)
