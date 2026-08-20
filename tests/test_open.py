from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pygradsx
from pygradsx.backend import PyGradsXBackendEntrypoint
from conftest import write_binary


def test_open_single_file_lazy_values(tmp_path, write_ctl):
    write_binary(
        tmp_path / "data.bin",
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
    )
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 10 1
        zdef 1 levels 1000
        tdef 2 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    assert ds.t.shape == (2, 2, 2)
    np.testing.assert_array_equal(ds.t.isel(time=1).values, [[5, 6], [7, 8]])


def test_undef_masking(tmp_path, write_ctl):
    write_binary(tmp_path / "data.bin", [[[1, -999], [3, 4]]])
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    assert np.isnan(ds.t.values[0, 0, 1])
    raw = pygradsx.open_dataset(ctl, mask_and_scale=False)
    assert raw.t.values[0, 0, 1] == -999


def test_yrev_data_only_coordinate_stays_ydef_order(tmp_path, write_ctl):
    write_binary(tmp_path / "data.bin", [[[11, 11], [10, 10]]])
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian yrev
        xdef 2 linear 0 1
        ydef 2 linear 10 1
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    np.testing.assert_array_equal(ds.lat.values, [10, 11])
    np.testing.assert_array_equal(ds.t.values[0], [[10, 10], [11, 11]])


def test_zrev_sequential_big_endian_multilevel(tmp_path, write_ctl):
    write_binary(
        tmp_path / "data.bin",
        [
            [[850, 850], [850, 850]],
            [[1000, 1000], [1000, 1000]],
        ],
        endian=">",
        sequential=True,
    )
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options big_endian sequential zrev
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        zdef 2 levels 1000 850
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 2 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    np.testing.assert_array_equal(ds.lev.values, [1000, 850])
    assert float(ds.t.sel(lev=1000).isel(time=0, lat=0, lon=0)) == 1000
    assert float(ds.t.sel(lev=850).isel(time=0, lat=0, lon=0)) == 850


def test_open_missing_large_tdef_is_lazy(tmp_path, write_ctl):
    ctl = write_ctl(
        """
        dset ^missing.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 1000 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl, on_missing="nan")
    assert ds.sizes["time"] == 1000
    assert np.isnan(ds.t.isel(time=999).values).all()


def test_chunks_smoke(tmp_path, write_ctl):
    pytest.importorskip("dask")
    write_binary(tmp_path / "data.bin", [[[1, 2], [3, 4]]])
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl, chunks={"time": 1})
    assert hasattr(ds.t.data, "compute")


def test_xarray_engine_entrypoint_open_dataset(tmp_path, write_ctl):
    write_binary(tmp_path / "data.bin", [[[1, 2], [3, 4]]])
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 1 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = xr.open_dataset(ctl, engine="pygradsx")
    np.testing.assert_array_equal(ds.t.values[0], [[1, 2], [3, 4]])


def test_backend_guess_can_open_ctl_only():
    backend = PyGradsXBackendEntrypoint()
    assert backend.guess_can_open("sample.ctl")
    assert not backend.guess_can_open("sample.nc")
