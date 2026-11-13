import numpy as np
import pynbody

from .units import in_units


def parse_limit(limit: float | str) -> tuple[float, str | None]:
    """Split a limit into a value and its units.

    A number with units, e.g. "1.0e10 Msol", gives `(1e10, "Msol")`, and the units may contain spaces, e.g.
    "1e7 Msol kpc**-2". Numbers and numeric strings, e.g. 1 or "1.0e10" as config files may deliver them, have no
    units.
    """
    if isinstance(limit, str):
        value, _, units = limit.strip().partition(" ")
        try:
            return float(value), (units.strip() or None)
        except ValueError:
            raise ValueError(
                f"A limit must be a number or a number with units, e.g. '1e10 Msol', got {limit!r}."
            ) from None
    return float(limit), None


class FilterRange:
    """Keep records whose `column` value lies within `[min, max]`.

    Works both per record (`__call__`) and vectorized over a whole table of
    per-record columns (`mask`). The pipeline uses `mask` to drop records
    before they are extracted, if the dataset provides a `table()`.

    Args:
        column (str): Name of the column to filter on, e.g. "stars_mass".
        min (float | str): Lower limit (default -inf), as a float in `units`
            or a unit string, e.g. "1.0e10 Msol".
        max (float | str): Upper limit (default inf), as a float in `units`
            or a unit string, e.g. "5.0e10 Msol".
        index (int | None): Index into the last axis of a multi-valued column,
            e.g. `4` for the stellar mass in `SubhaloMassType`.
        units (str | None): Convert unit-carrying values (pynbody `SimArray`)
            to these units before comparing, e.g. "Msol". Limits with units
            are converted to them as well. If `None`, the units of the limits
            are used. Values without units are compared unchanged.
        inclusive (bool): Whether the limits themselves are kept (default True).
    """

    is_filter = True

    def __init__(
        self,
        column: str,
        min: float | str = -np.inf,
        max: float | str = np.inf,
        index: int | None = None,
        units: str | None = None,
        inclusive: bool = True,
    ):
        self.column = column
        (min_value, min_units), (max_value, max_units) = parse_limit(min), parse_limit(max)
        # Plain limits are in `units`, which default to the units of the limits.
        self.units = units or min_units or max_units
        self.min = self._in_units(min_value, min_units)
        self.max = self._in_units(max_value, max_units)
        self.index = index
        self.inclusive = inclusive

    def _in_units(self, value: float, units: str | None) -> float:
        """A limit of `value` in `units` converted to the units of the filter."""
        # The ratio also checks the units of an infinite limit.
        ratio = 1.0 if units is None else pynbody.units.Unit(units).ratio(self.units)
        return value if np.isinf(value) else value * ratio

    def mask(self, table: dict) -> np.ndarray:
        values = in_units(table[self.column], self.units)
        if self.index is not None:
            values = values[..., self.index]
        if self.inclusive:
            return (values >= self.min) & (values <= self.max)
        return (values > self.min) & (values < self.max)

    def __call__(self, record: dict) -> bool:
        value = record[self.column]
        # Keep the units of a SimArray scalar/row by indexing instead of wrapping it.
        return bool(self.mask({self.column: value[None] if hasattr(value, "units") else np.asarray(value)[None]})[0])
