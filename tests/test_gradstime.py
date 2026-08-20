from __future__ import annotations

import numpy as np

from pygradsx.gradstime import build_times, parse_grads_time


def test_parse_grads_time():
    dt = parse_grads_time("12:30Z05jan2020")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2020, 1, 5, 12, 30)


def test_build_monthly_times():
    times = build_times(3, "00Z31jan2020", "1mo")
    assert list(times.astype("datetime64[D]").astype(str)) == [
        "2020-01-31",
        "2020-02-29",
        "2020-03-31",
    ]

