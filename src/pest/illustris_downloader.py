import os

import requests

from pest.get_illustris_apy_key import get_illustris_api_key


class IllustrisDownloader(object):

    def __init__(
        self,
        download_path: str = "data",
        base_path: str = "http://www.tng-project.org/api",
        simulation: str = "TNG100-1",
        snapshot: int = 99,
        timeout: int = 100,
    ):
        """Downloads data from the Illustris API.

        Args:
            base_path (str, optional): Defaults to "http://www.tng-project.org/api/TNG100-1/".
            snap_num (int, optional): Defaults to 99.

        """
        self.download_path = download_path
        self.headers = {"api-key": get_illustris_api_key()}
        self.url = os.path.join(base_path, simulation, "snapshots", str(snapshot))
        self.params = {
            "stars": "Coordinates,Masses",
            "gas": "Coordinates,Potential",
        }
        self.timeout = timeout

    def get(self, subhalo_id: int):
        # make HTTP GET request to path
        r = requests.get(
            url=os.path.join(self.url, "subhalos", str(subhalo_id)),
            headers=self.headers,
            timeout=self.timeout,
        )

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers["content-type"] != "application/json":
            raise RuntimeError("Response content is not JSON")

        return r.json()

    def get_hdf5(self, subhalo_id: int):
        # make HTTP GET request to path
        r = requests.get(
            url=os.path.join(self.url, "subhalos", str(subhalo_id), "cutout.hdf5"),
            headers=self.headers,
            params=self.params,
            timeout=self.timeout,
        )

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if "content-disposition" not in r.headers:
            raise RuntimeError("No content-disposition header found")

        filename = r.headers["content-disposition"].split("filename=")[1]
        with open(os.path.join(self.download_path, filename), "wb") as f:
            f.write(r.content)
        return filename
