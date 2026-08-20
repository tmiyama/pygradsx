from __future__ import annotations

import numpy as np
import pytest

import pygradsx
from conftest import write_binary


def test_short_single_file_nan_and_skip(tmp_path, write_ctl):
    write_binary(tmp_path / "data.bin", [[[1, 2], [3, 4]]])
    ctl = write_ctl(
        """
        dset ^data.bin
        undef -999
        options little_endian
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 3 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    assert ds.sizes["time"] == 3
    assert np.isnan(ds.t.isel(time=2).values).all()
    skipped = pygradsx.open_dataset(ctl, on_missing="skip")
    assert skipped.sizes["time"] == 1
    with pytest.raises(EOFError):
        pygradsx.open_dataset(ctl, on_missing="error")


def test_template_missing_file_nan_skip_error(tmp_path, write_ctl):
    write_binary(tmp_path / "data_20200101.bin", [[[1, 2], [3, 4]]])
    write_binary(tmp_path / "data_20200103.bin", [[[9, 10], [11, 12]]])
    ctl = write_ctl(
        """
        dset ^data_%y4%m2%d2.bin
        undef -999
        options little_endian template
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 3 linear 00Z01jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    assert ds.sizes["time"] == 3
    assert np.isnan(ds.t.isel(time=1).values).all()
    skipped = pygradsx.open_dataset(ctl, on_missing="skip")
    assert skipped.sizes["time"] == 2
    with pytest.raises(FileNotFoundError):
        pygradsx.open_dataset(ctl, on_missing="error")


def test_template_multiple_steps_per_file_and_partial_tail(tmp_path, write_ctl):
    write_binary(
        tmp_path / "data_202001.bin",
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
        ],
    )
    write_binary(tmp_path / "data_202002.bin", [[[3, 3], [3, 3]]])
    ctl = write_ctl(
        """
        dset ^data_%y4%m2.bin
        undef -999
        options little_endian template
        xdef 2 linear 0 1
        ydef 2 linear 0 1
        tdef 4 linear 00Z30jan2020 1dy
        vars 1
        t 0 K temp
        endvars
        """
    )
    ds = pygradsx.open_dataset(ctl)
    assert ds.sizes["time"] == 4
    assert float(ds.t.isel(time=0, lat=0, lon=0)) == 1
    assert float(ds.t.isel(time=1, lat=0, lon=0)) == 2
    assert float(ds.t.isel(time=2, lat=0, lon=0)) == 3
    assert np.isnan(ds.t.isel(time=3).values).all()
    skipped = pygradsx.open_dataset(ctl, on_missing="skip")
    assert skipped.sizes["time"] == 3
    with pytest.raises(FileNotFoundError):
        pygradsx.open_dataset(ctl, on_missing="error")
