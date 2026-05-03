import numpy as np


class RemoveInvalidImages:
    """Remove images that contain NaN, Inf, or have all pixels the same value.

    Used as a filter transformation in the pipeline. Returns True if the image
    is valid (keep), False otherwise (discard).
    """

    is_filter = True

    def __call__(self, sample: dict) -> bool:
        image = np.array(sample["image"])
        return not (np.isnan(image).any() or np.isinf(image).any() or np.all(image == image.flat[0]))
