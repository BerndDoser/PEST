import numpy as np
import pynbody

from .render_stars import center_and_orient, check_center_and_orientation


class RenderGas:
    """Transform step that adds a gas image of a subhalo to the record.

    The pipeline binds the worker's dataset to this step, which must provide
    `gas(subhalo_id)` returning a snapshot of the gas particles of the subhalo,
    centered on the subhalo (e.g. `PynbodyDataset`). The center is refined on
    the gas with `pynbody.analysis.center`, and the image is rendered with
    `pynbody.plot.image`, by default the projected gas surface density. The
    SPH smoothing needs at least 32 gas particles (pynbody's default number of
    neighbours).

    Args:
        width (float | str): Image width, as a float in the snapshot units
            (kpc for `PynbodyDataset`) or a unit string, e.g. "50 kpc".
        resolution (int | None): Image size in pixels. If `None`, the pynbody
            default is used.
        center_mode (str | None): pynbody centering scheme to refine the center
            on the gas particles: "ssc", "com" or "hyb". If `None`, the pynbody
            default ("ssc") is used. "catalogue" keeps the center of the
            dataset, e.g. the most bound particle of SUBFIND catalogues.
        orientation (str | None): Rotate the galaxy before rendering with
            `pynbody.analysis.faceon` or `pynbody.analysis.sideon`, which orient
            the angular momentum of the gas within `disk_size` along the line
            of sight or perpendicular to it. If `None`, the image is not rotated.
            The velocities of the gas are loaded for this.
        disk_size (float | str): Radius around the center that determines the
            angular momentum for `orientation`, as a float in the snapshot units
            or a unit string.
        qty (str): Gas array to render, e.g. "rho" or "temp".
        units (str | None): Units of the image. Units of `qty` integrated along
            the line of sight give a projected image, e.g. the default
            "Msol kpc**-2" for the surface density of "rho". If `None`, the
            units of `qty` are used, which gives a slice at z = 0 unless
            `render_args` sets a `weight`, e.g. `{"weight": "rho"}` for the
            density-weighted average of `qty` along the line of sight.
        log (bool): Store the log10 of the image. Pixels without a positive
            value are set to the smallest positive one. Only for quantities
            that are positive.
        column (str): Name of the added column.
        render_args (dict | None): Additional arguments for
            `pynbody.plot.image`, e.g. `{"weight": "rho"}` or
            `{"kernel": "WendlandC2"}`.
    """

    def __init__(
        self,
        width: float | str = 50,
        resolution: int | None = None,
        center_mode: str | None = None,
        orientation: str | None = None,
        disk_size: float | str = "5 kpc",
        qty: str = "rho",
        units: str | None = "Msol kpc**-2",
        log: bool = True,
        column: str = "gas_image",
        render_args: dict | None = None,
    ):
        check_center_and_orientation(center_mode, orientation)
        self.width = width
        self.resolution = resolution
        self.center_mode = center_mode
        self.orientation = orientation
        self.disk_size = disk_size
        self.qty = qty
        self.units = units
        self.log = log
        self.column = column
        self.render_args = render_args or {}
        # Arrays besides the default ones of `gas()` that the renderer reads.
        self.arrays = [name for name in (qty, self.render_args.get("weight")) if isinstance(name, str)]
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, "gas"):
            raise TypeError(f"{type(dataset).__name__} does not provide gas particles.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict:
        if self.dataset is None:
            raise RuntimeError("RenderGas must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError("RenderGas requires the 'subhalo_id' column in the extracted records.")
        subhalo_id = record["subhalo_id"]
        gas = self.dataset.gas(subhalo_id, velocities=self.orientation is not None, arrays=self.arrays)
        if len(gas) == 0:
            raise ValueError(f"Subhalo {subhalo_id} has no gas particles to render.")
        center_and_orient(gas, self.center_mode, self.orientation, self.disk_size)
        image = pynbody.plot.image(
            gas,
            qty=self.qty,
            width=self.width,
            resolution=self.resolution,
            units=self.units,
            noplot=True,
            **self.render_args,
        )
        # pynbody puts the lowest y in the first row; flip to the usual top-down image layout.
        image = np.asarray(image[::-1], dtype=np.float64)
        if self.log:
            positive = image > 0
            if not positive.any():
                raise ValueError(f"Subhalo {subhalo_id} has no positive {self.qty!r} in the image to take the log of.")
            image = np.log10(np.where(positive, image, image[positive].min()))
        # Channels first: (1, height, width).
        record[self.column] = image[np.newaxis].astype(np.float32)
        return record
