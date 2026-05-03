import numpy as np

from .orientation import align_image_horizontally, estimate_geometry_weighted


class AlignImageHorizontally:
    """Rotate a (C, H, W) image so that the galaxy major axis is horizontal."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pipeline format is (C, H, W); orientation functions expect (H, W, C)
        image = image.transpose(1, 2, 0)
        stats = estimate_geometry_weighted(image)
        image = align_image_horizontally(image, stats["pa_rad"])
        return image.transpose(2, 0, 1)
