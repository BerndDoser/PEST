import numpy as np

from .orientation import estimate_geometry_weighted


class CropQuadratic:
    """Crop a square region around the galaxy centroid detected via weighted moments.

    The half-size of the crop is ``scale`` times the estimated major axis length.

    Args:
        scale (float): Multiplier applied to the major axis length to determine the
            half-size of the square crop (default 1.5).
    """

    def __init__(self, scale: float = 1.5):
        self.scale = scale

    def crop_(self, img, center, half_size):
        """Crop a square region around the center."""
        x_c, y_c = center
        H, W = img.shape[:2]

        col_min = int(max(0, x_c - half_size))
        col_max = int(min(W, x_c + half_size))
        row_min = int(max(0, y_c - half_size))
        row_max = int(min(H, y_c + half_size))

        return img[row_min:row_max, col_min:col_max]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pipeline format is (C, H, W); orientation functions expect (H, W, C)
        img_hwc = np.moveaxis(image, 0, -1)
        stats = estimate_geometry_weighted(img_hwc.copy())
        half_size = self.scale * stats["major_axis"]
        cropped_hwc = self.crop_(img_hwc, stats["centroid"], half_size)
        return np.moveaxis(cropped_hwc, -1, 0).astype(image.dtype)
