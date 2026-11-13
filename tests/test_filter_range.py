import numpy as np
import pynbody
import pytest

from pest import FilterRange
from pest.filter_range import parse_limit

VALUES = np.array([1.0, 2.0, 3.0, 4.0])


def test_mask_and_call_agree():
    f = FilterRange("x", min=2.0, max=3.0)
    mask = f.mask({"x": VALUES})
    assert list(mask) == [False, True, True, False]
    assert [f({"x": v}) for v in VALUES] == list(mask)


def test_exclusive_and_open_bounds():
    assert list(FilterRange("x", min=2.0, max=3.0, inclusive=False).mask({"x": VALUES})) == [False] * 4
    assert list(FilterRange("x", min=3.0).mask({"x": VALUES})) == [False, False, True, True]
    assert list(FilterRange("x", max=1.0).mask({"x": VALUES})) == [True, False, False, False]


def test_index():
    table = {"x": np.stack([VALUES, 10 * VALUES], axis=-1)}
    f = FilterRange("x", index=1, min=25.0)
    assert list(f.mask(table)) == [False, False, True, True]
    assert f({"x": table["x"][2]})
    assert not f({"x": table["x"][1]})


def test_string_limits():
    f = FilterRange("x", min="2.0", max="3e0")
    assert (f.min, f.max) == (2.0, 3.0)
    assert list(f.mask({"x": VALUES})) == [False, True, True, False]


@pytest.mark.parametrize("units, expected", [("Msol", [False, True, False, False]), (None, [False] * 4)])
def test_units(units, expected):
    table = {"m": pynbody.array.SimArray(VALUES, "1e10 Msol")}
    f = FilterRange("m", min=1.5e10, max=2.5e10, units=units)
    assert list(f.mask(table)) == expected


@pytest.mark.parametrize(
    "limit, expected",
    [
        (1, (1.0, None)),
        ("2.0", (2.0, None)),
        ("1.0e10 Msol", (1e10, "Msol")),
        (" 1e7  Msol kpc**-2 ", (1e7, "Msol kpc**-2")),
    ],
)
def test_parse_limit(limit, expected):
    assert parse_limit(limit) == expected


@pytest.mark.parametrize("limit", ["Msol", "", "ten Msol"])
def test_parse_limit_invalid(limit):
    with pytest.raises(ValueError, match="number with units"):
        parse_limit(limit)


@pytest.mark.parametrize(
    "min, max, units, expected_limits",
    [
        ("1.5e10 Msol", "2.5e10 Msol", None, (1.5e10, 2.5e10)),
        # Limits in different units are converted to the units of the first one.
        ("1.5e10 Msol", "2.5 1e10 Msol", None, (1.5e10, 2.5e10)),
        # A plain limit is in the units of the other one.
        ("1.5e10 Msol", 2.5e10, None, (1.5e10, 2.5e10)),
        ("1.5e10 Msol", "2.5e10 Msol", "1e10 Msol", (1.5, 2.5)),
    ],
)
def test_limits_with_units(min, max, units, expected_limits):
    # Rows like SubhaloMassType, which keep their units when a record is indexed from the table.
    table = {"m": pynbody.array.SimArray(np.stack([10 * VALUES, VALUES], axis=-1), "1e10 Msol")}
    f = FilterRange("m", min=min, max=max, index=1, units=units)
    np.testing.assert_allclose((f.min, f.max), expected_limits)
    assert list(f.mask(table)) == [False, True, False, False]
    assert [f({"m": row}) for row in table["m"]] == [False, True, False, False]


def test_limits_with_incompatible_units():
    with pytest.raises(pynbody.units.UnitsException):
        FilterRange("m", min="1e10 Msol", units="kpc")
    with pytest.raises(pynbody.units.UnitsException):
        FilterRange("m", min="1e10 Msol", max="inf kpc")
