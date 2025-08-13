from pest import PointCloudGenerator


def test_point_cloud_generation(tmp_path):
    data = {
        "galaxies": [
            {
                "id": 1,
                "central": True,
                "mass": 1.0,
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "particles": {
                    "stars": [
                        {
                            "id": 1,
                            "mass": 1.0,
                            "position": [0.0, 0.0, 0.0],
                            "velocity": [0.0, 0.0, 0.0],
                        }
                    ]
                },
            }
        ]
    }
    generator = PointCloudGenerator(["stars_position", "stars_mass"])
    generator(data, tmp_path)

    assert (tmp_path / "0.parquet").exists(), "Parquet file was not created."
