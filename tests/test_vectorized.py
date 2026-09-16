from __future__ import annotations

import numpy as np
import xarray as xr

import pygradsx
from conftest import write_binary
from pygradsx.core import GradsBackendArray


def _multilevel_ctl(write_ctl, *, options="little_endian"):
    return write_ctl(
        f"""
        dset ^data.bin
        undef -999
        options {options}
        xdef 5 linear 130 1
        ydef 4 linear 30 1
        zdef 3 levels 0 10 20
        tdef 2 linear 00Z01jan2020 1dy
        vars 1
        u 3 m/s velocity
        endvars
        """
    )


def _data():
    return np.arange(2 * 3 * 4 * 5, dtype="float32").reshape(2, 3, 4, 5)


def test_vectorized_points_match_numpy(tmp_path, write_ctl):
    data = _data()
    write_binary(tmp_path / "data.bin", data.reshape(-1, 4, 5))
    ds = pygradsx.open_dataset(_multilevel_ctl(write_ctl))

    t = xr.DataArray([0, 0, 1, 1, 1], dims="point")
    z = xr.DataArray([0, 2, 1, 0, 2], dims="point")
    y = xr.DataArray([0, 3, 1, 2, 0], dims="point")
    x = xr.DataArray([4, 0, 3, 1, 2], dims="point")

    actual = ds.u.isel(time=t, lev=z, lat=y, lon=x).values
    expected = data[t.values, z.values, y.values, x.values]
    np.testing.assert_array_equal(actual, expected)


def test_vectorized_points_bypass_full_plane_reader(tmp_path, write_ctl, monkeypatch):
    data = _data()
    write_binary(tmp_path / "data.bin", data.reshape(-1, 4, 5))
    ds = pygradsx.open_dataset(_multilevel_ctl(write_ctl))

    def fail_full_plane(*args, **kwargs):
        raise AssertionError("vectorized point reads must not call _read_plane")

    monkeypatch.setattr(GradsBackendArray, "_read_plane", fail_full_plane)
    actual = ds.u.isel(
        time=xr.DataArray([0, 1, 1], dims="point"),
        lev=xr.DataArray([0, 1, 2], dims="point"),
        lat=xr.DataArray([1, 2, 3], dims="point"),
        lon=xr.DataArray([2, 3, 4], dims="point"),
    ).values
    np.testing.assert_array_equal(actual, [7, 93, 119])


def test_vectorized_points_yrev_zrev_and_undef(tmp_path, write_ctl):
    logical = _data()
    logical[1, 2, 1, 3] = -999
    # zrev and yrev describe storage order; reverse both axes before writing.
    stored = logical[:, ::-1, ::-1, :]
    write_binary(tmp_path / "data.bin", stored.reshape(-1, 4, 5))
    ds = pygradsx.open_dataset(
        _multilevel_ctl(write_ctl, options="little_endian yrev zrev")
    )

    t = xr.DataArray([0, 1, 1], dims="point")
    z = xr.DataArray([2, 0, 2], dims="point")
    y = xr.DataArray([3, 2, 1], dims="point")
    x = xr.DataArray([4, 1, 3], dims="point")
    actual = ds.u.isel(time=t, lev=z, lat=y, lon=x).values
    expected = logical[t.values, z.values, y.values, x.values].astype("float32")
    expected[expected == -999] = np.nan
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_vectorized_points_preserve_multidimensional_shape(tmp_path, write_ctl):
    data = _data()
    write_binary(tmp_path / "data.bin", data.reshape(-1, 4, 5))
    ds = pygradsx.open_dataset(_multilevel_ctl(write_ctl))

    t = xr.DataArray([[0, 0], [1, 1]], dims=("a", "b"))
    z = xr.DataArray([[0, 1], [1, 2]], dims=("a", "b"))
    y = xr.DataArray([[0, 1], [2, 3]], dims=("a", "b"))
    x = xr.DataArray([[1, 2], [3, 4]], dims=("a", "b"))
    actual = ds.u.isel(time=t, lev=z, lat=y, lon=x).values
    expected = data[t.values, z.values, y.values, x.values]
    np.testing.assert_array_equal(actual, expected)


def test_vectorized_points_negative_indices(tmp_path, write_ctl):
    data = _data()
    write_binary(tmp_path / "data.bin", data.reshape(-1, 4, 5))
    ds = pygradsx.open_dataset(_multilevel_ctl(write_ctl))

    actual = ds.u.isel(
        time=xr.DataArray([-1, 0], dims="point"),
        lev=xr.DataArray([-1, -2], dims="point"),
        lat=xr.DataArray([-1, -2], dims="point"),
        lon=xr.DataArray([-1, -2], dims="point"),
    ).values
    expected = data[[-1, 0], [-1, -2], [-1, -2], [-1, -2]]
    np.testing.assert_array_equal(actual, expected)
