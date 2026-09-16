from __future__ import annotations

import numpy as np

import pygradsx
from conftest import write_binary
from pygradsx.core import GradsBackendArray


def _ctl(write_ctl, *, options="little_endian"):
    return write_ctl(
        f"""
        dset ^data.bin
        undef -999
        options {options}
        xdef 5 linear 130 1
        ydef 4 linear 30 1
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )


def test_horizontal_slice_matches_full_plane(tmp_path, write_ctl):
    plane = np.arange(20, dtype="float32").reshape(4, 5)
    write_binary(tmp_path / "data.bin", [plane])
    ds = pygradsx.open_dataset(_ctl(write_ctl))

    expected = ds.t.isel(time=0).values[1:4:2, 1:5:2]
    actual = ds.t.isel(time=0, lat=slice(1, 4, 2), lon=slice(1, 5, 2)).values
    np.testing.assert_array_equal(actual, expected)


def test_horizontal_scalar_and_undef_mask(tmp_path, write_ctl):
    plane = np.arange(20, dtype="float32").reshape(4, 5)
    plane[2, 3] = -999
    write_binary(tmp_path / "data.bin", [plane])
    ds = pygradsx.open_dataset(_ctl(write_ctl))

    assert np.isnan(ds.t.isel(time=0, lat=2, lon=3).values)
    assert float(ds.t.isel(time=0, lat=1, lon=4).values) == 9.0


def test_horizontal_subset_preserves_yrev(tmp_path, write_ctl):
    stored = np.arange(20, dtype="float32").reshape(4, 5)
    write_binary(tmp_path / "data.bin", [stored])
    ds = pygradsx.open_dataset(_ctl(write_ctl, options="little_endian yrev"))

    expected = stored[::-1, :][1:3, 2:5]
    actual = ds.t.isel(time=0, lat=slice(1, 3), lon=slice(2, 5)).values
    np.testing.assert_array_equal(actual, expected)


def test_horizontal_subset_big_endian_sequential(tmp_path, write_ctl):
    plane = np.arange(20, dtype="float32").reshape(4, 5)
    write_binary(tmp_path / "data.bin", [plane], endian=">", sequential=True)
    ds = pygradsx.open_dataset(
        _ctl(write_ctl, options="big_endian sequential")
    )

    actual = ds.t.isel(time=0, lat=slice(1, 4), lon=slice(2, 5)).values
    np.testing.assert_array_equal(actual, plane[1:4, 2:5])


def test_basic_horizontal_selectors_are_pushed_to_plane_reader(
    tmp_path, write_ctl, monkeypatch
):
    plane = np.arange(20, dtype="float32").reshape(4, 5)
    write_binary(tmp_path / "data.bin", [plane])
    calls = []
    original = GradsBackendArray._read_plane

    def wrapped(self, time_index, z_index, y_sel=slice(None), x_sel=slice(None)):
        calls.append((y_sel, x_sel))
        return original(
            self, time_index, z_index, y_sel=y_sel, x_sel=x_sel
        )

    monkeypatch.setattr(GradsBackendArray, "_read_plane", wrapped)
    ds = pygradsx.open_dataset(_ctl(write_ctl))
    ds.t.isel(time=0, lat=slice(1, 3), lon=slice(2, 5)).load()

    assert calls
    y_sel, x_sel = calls[-1]
    # xarray normalizes an omitted slice step to 1 before calling the backend.
    assert (y_sel.start, y_sel.stop, y_sel.step) == (1, 3, 1)
    assert (x_sel.start, x_sel.stop, x_sel.step) == (2, 5, 1)
