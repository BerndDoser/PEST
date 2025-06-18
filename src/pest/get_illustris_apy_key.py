import os


def get_illustris_api_key():
    """Return the API key for the Illustris API.
    The key can be set as an environment variable or in a file."""
    api_file = os.path.join(os.path.dirname(__file__), "..", "..", ".illustris_api_key.txt")
    if "ILLUSTRIS_API_KEY" in os.environ:
        return os.environ["ILLUSTRIS_API_KEY"]
    elif os.path.isfile(api_file):
        with open(api_file, "r") as file:
            return file.read().rstrip()
    raise ValueError(
        "No API key found. Please set the ILLUSTRIS_API_KEY environment variable"
        "or create a file named '.illustris_api_key.txt' with the API key"
    )
