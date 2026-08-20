from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def write_ctl(tmp_path):
    def _write(text: str) -> Path:
        path = tmp_path / "test.ctl"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    return _write


def write_binary(path: Path, planes, *, endian="<", sequential=False):
    arr = np.asarray(planes, dtype=f"{endian}f4")
    if sequential:
        marker = np.array([arr.shape[-2] * arr.shape[-1] * 4], dtype=f"{endian}i4")
        with path.open("wb") as f:
            for plane in arr:
                marker.tofile(f)
                plane.tofile(f)
                marker.tofile(f)
    else:
        arr.tofile(path)
    return path
