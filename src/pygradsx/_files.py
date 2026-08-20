from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class MemmapCache:
    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._items: "OrderedDict[Tuple[Path, str], np.memmap]" = OrderedDict()

    def get(self, path: Path, dtype: np.dtype) -> np.memmap:
        key = (Path(path), np.dtype(dtype).str)
        try:
            mm = self._items.pop(key)
        except KeyError:
            mm = np.memmap(path, mode="r", dtype=dtype)
        self._items[key] = mm
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)
        return mm


DEFAULT_CACHE = MemmapCache()

