from __future__ import annotations

from pygradsx import open_ctl


def test_ctl_parser(write_ctl):
    ctl = write_ctl(
        """
        dset ^data.bin
        title sample
        undef -999
        options little_endian
        xdef 2 linear 10 1
        ydef 2 levels 20 21
        zdef 2 levels 1000 850
        tdef 2 linear 00Z01jan2020 1dy
        vars 1
        temp 2 K temperature
        endvars
        """
    )
    desc = open_ctl(ctl)
    assert desc.dset.name == "data.bin"
    assert desc.title == "sample"
    assert desc.nx == 2
    assert desc.ny == 2
    assert desc.vars[0].planes == 2

