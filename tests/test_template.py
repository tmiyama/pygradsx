from __future__ import annotations

from datetime import datetime

import pytest

from pygradsx.template import expand_template


def test_expand_template():
    assert (
        expand_template("run_%y4%m2%d2_%h2%n2.bin", datetime(2020, 2, 3, 4, 5))
        == "run_20200203_0405.bin"
    )


def test_unsupported_template_tokens():
    with pytest.raises(NotImplementedError):
        expand_template("member_%e.bin", datetime(2020, 1, 1))
    with pytest.raises(NotImplementedError):
        expand_template("hour_%ch.bin", datetime(2020, 1, 1))
