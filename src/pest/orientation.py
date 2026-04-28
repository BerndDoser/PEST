import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from skimage import io


def estimate_geometry_weighted(img, q0=0.2):
    # 1. Load image and ensure it's a float for math
    # img = io.imread(image_path, as_gray=True).astype(float)

    # 2. Basic Background Subtraction
    # Moments are very sensitive to background noise.
    # We subtract the median to ensure the "sky" is roughly 0.
    img -= np.median(img)
    img[img < 0] = 0  # Clip negative values

    # 3. Create coordinate grids
    y, x = np.indices(img.shape)

    # 4. Calculate 0th and 1st moments (Total flux and Centroid)
    m00 = np.sum(img)
    m10 = np.sum(x * img)
    m01 = np.sum(y * img)
    x_c = m10 / m00
    y_c = m01 / m00

    # 5. Calculate 2nd order central moments (Variance)
    mu20 = np.sum((x - x_c) ** 2 * img) / m00
    mu02 = np.sum((y - y_c) ** 2 * img) / m00
    mu11 = np.sum((x - x_c) * (y - y_c) * img) / m00

    # 6. Calculate Eigenvalues of the covariance matrix
    # These represent the squared lengths of the axes (a^2 and b^2)
    term1 = (mu20 + mu02) / 2
    term2 = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11**2)

    l1 = term1 + term2  # Major axis variance
    l2 = term1 - term2  # Minor axis variance

    a = 2 * np.sqrt(l1)  # Major axis length (sigma scale)
    b = 2 * np.sqrt(l2)  # Minor axis length (sigma scale)

    # 7. Position Angle and Inclination
    pa_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    axis_ratio = b / a
    cos_i_sq = (axis_ratio**2 - q0**2) / (1 - q0**2)
    # Clip values to [0, 1] to handle edge-on/face-on edge cases
    inclination_deg = np.degrees(np.arccos(np.sqrt(np.clip(cos_i_sq, 0, 1))))

    return {
        "inclination_deg": inclination_deg,
        "pa_rad": pa_rad,
        "major_axis": a,
        "minor_axis": b,
        "centroid": (x_c, y_c),
        "image": img,
    }


# --- Visualization ---
def visualize_results(stats):
    plt.figure(figsize=(8, 6))
    plt.imshow(stats["image"], cmap="inferno", origin="lower")

    x_c, y_c = stats["centroid"]
    el = Ellipse(
        xy=(x_c, y_c),
        width=stats["major_axis"] * 2,
        height=stats["minor_axis"] * 2,
        angle=np.degrees(stats["pa_rad"]),
        edgecolor="white",
        facecolor="none",
        lw=2,
        linestyle="--",
    )

    plt.gca().add_patch(el)
    plt.title(f"Inclination: {stats['inclination_deg']:.1f}°\nPA: {np.degrees(stats['pa_rad']):.1f}°")
    plt.colorbar(label="Intensity")
    plt.show()


# Example usage:
# res = estimate_geometry_weighted('galaxy.tif')
# visualize_results(res)
