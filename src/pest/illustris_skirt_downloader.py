"""Downloader for Illustris/TNG SKIRT synthetic image tarballs."""

import argparse
import os
import re
from pathlib import Path

import requests

SKIRT_URLS = [
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.131.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.135.tar",
]


def get_illustris_api_key() -> str:
    """Return the API key for the Illustris/TNG API.

    The key can be set via the ILLUSTRIS_API_KEY environment variable or in a
    '.illustris_api_key.txt' file in the project root directory.
    """
    if "ILLUSTRIS_API_KEY" in os.environ:
        return os.environ["ILLUSTRIS_API_KEY"]

    api_file = Path(__file__).resolve().parents[2] / ".illustris_api_key.txt"
    if api_file.is_file():
        return api_file.read_text().strip()

    raise ValueError(
        "No API key found. Please set the ILLUSTRIS_API_KEY environment variable "
        "or create a file named '.illustris_api_key.txt' with the API key."
    )


def download_file(url: str, output_path: Path, api_key: str, chunk_size: int = 1 << 20) -> Path:
    """Download a single file, honoring the server's suggested filename.

    Equivalent to `wget -nc --content-disposition`: the download is skipped if
    a file with the resolved filename already exists in output_path.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers={"api-key": api_key}, stream=True, timeout=60) as response:
        response.raise_for_status()

        filename = None
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            match = re.search(r'filename="?([^";]+)"?', content_disposition)
            if match:
                filename = match.group(1)
        if filename is None:
            filename = url.rsplit("/", 1)[-1]

        destination = output_path / filename
        if destination.exists():
            print(f"Skipping {filename} (already exists)")
            return destination

        print(f"Downloading {filename} ...")
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)

    return destination


def download_files(urls: list[str], output_path: str | Path, api_key: str | None = None) -> list[Path]:
    """Download a list of URLs into output_path, skipping files that already exist."""
    api_key = api_key or get_illustris_api_key()
    output_path = Path(output_path)
    return [download_file(url, output_path, api_key) for url in urls]


def main() -> None:
    """CLI entry point: download the SKIRT tarballs to a given path."""
    parser = argparse.ArgumentParser(
        prog="pest-download-skirt",
        description="Download Illustris/TNG SKIRT synthetic image tarballs.",
    )
    parser.add_argument("path", help="Directory to download the files into.")
    parser.add_argument(
        "--urls-file",
        help="Optional text file with one URL per line. Defaults to the built-in SKIRT tarball list.",
    )
    args = parser.parse_args()

    if args.urls_file:
        with open(args.urls_file) as fh:
            urls = [line.strip() for line in fh if line.strip()]
    else:
        urls = SKIRT_URLS

    download_files(urls, args.path)


if __name__ == "__main__":
    main()
