import numpy as np

from .orientation import align_image_horizontally, estimate_geometry_weighted


class AlignImageHorizontally:
    """Rotate a (C, H, W) image so that the galaxy major axis is horizontal."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pipeline format is (C, H, W); orientation functions expect (H, W, C)
        img_hwc = np.moveaxis(image, 0, -1)
        stats = estimate_geometry_weighted(img_hwc.copy())
        rotated_hwc = align_image_horizontally(img_hwc, stats["pa_rad"])
        return np.moveaxis(rotated_hwc, -1, 0).astype(image.dtype)
