from __future__ import annotations

from pathlib import Path

from xarray.backends import BackendEntrypoint

from .core import open_dataset


class PyGradsXBackendEntrypoint(BackendEntrypoint):
    description = "Open GrADS ctl plus flat binary datasets with PyGradsX"
    url = "https://github.com/"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        on_missing="nan",
        mask_and_scale=True,
        chunks=None,
    ):
        ds = open_dataset(
            filename_or_obj,
            on_missing=on_missing,
            mask_and_scale=mask_and_scale,
            chunks=chunks,
        )
        if drop_variables:
            ds = ds.drop_vars(drop_variables)
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            suffix = Path(filename_or_obj).suffix.lower()
        except TypeError:
            return False
        return suffix == ".ctl"
