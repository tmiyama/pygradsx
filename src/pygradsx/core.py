from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

try:
    from xarray.backends.common import BackendArray
    from xarray.core import indexing
except Exception:  # pragma: no cover
    BackendArray = object
    indexing = None

from ._files import DEFAULT_CACHE
from .ctl import CtlDescriptor, VarDef, open_ctl
from .template import expand_template

Record = Optional[Tuple[Path, int]]


@dataclass(frozen=True)
class Layout:
    times: Sequence[object]
    records: Dict[str, List[List[Record]]]
    plane_bytes: int
    stored_shape: Tuple[int, int]


def _native_times(times: Sequence[object]) -> List[object]:
    out: List[object] = []
    for value in times:
        if isinstance(value, np.datetime64):
            ns = int(value.astype("datetime64[ns]").astype("int64"))
            out.append(datetime(1970, 1, 1) + timedelta(microseconds=ns // 1000))
        else:
            out.append(value)
    return out


def _record_stride(desc: CtlDescriptor) -> int:
    payload = desc.nx * desc.ny * 4 + int(desc.xyheader)
    return payload + 8 if desc.sequential else payload


def _time_stride(desc: CtlDescriptor) -> int:
    return int(desc.theader) + sum(v.planes for v in desc.vars) * _record_stride(desc)


def _plane_offset(desc: CtlDescriptor, time_index: int, plane_index: int) -> int:
    base = int(desc.fileheader) + time_index * _time_stride(desc) + int(desc.theader)
    off = base + plane_index * _record_stride(desc)
    if desc.sequential:
        off += 4
    off += int(desc.xyheader)
    return off


def _plane_order(desc: CtlDescriptor) -> Dict[str, int]:
    order: Dict[str, int] = {}
    plane = 0
    for var in desc.vars:
        order[var.name] = plane
        plane += var.planes
    return order


def _readable_times_for_file(desc: CtlDescriptor, path: Path) -> int:
    if not path.exists():
        return 0
    size = path.stat().st_size
    after_header = size - int(desc.fileheader)
    if after_header <= 0:
        return 0
    return max(0, min(len(desc.times), after_header // _time_stride(desc)))


def build_layout(desc: CtlDescriptor, on_missing: str = "nan") -> Layout:
    if on_missing not in {"nan", "skip", "error"}:
        raise ValueError("on_missing must be 'nan', 'skip', or 'error'")
    plane_order = _plane_order(desc)
    native_times = _native_times(desc.times)
    keep_indices: List[int] = []
    if desc.template:
        expanded_paths = [Path(expand_template(str(desc.dset), when)) for when in native_times]
        first_index_for_path: Dict[Path, int] = {}
        local_indices: Dict[int, int] = {}
        for i, path in enumerate(expanded_paths):
            first = first_index_for_path.setdefault(path, i)
            local_indices[i] = i - first
        available: Dict[int, Tuple[Path, int]] = {}
        readable_cache: Dict[Path, int] = {}
        for i, when in enumerate(native_times):
            path = expanded_paths[i]
            local_index = local_indices[i]
            readable = readable_cache.setdefault(path, _readable_times_for_file(desc, path))
            if readable > local_index:
                available[i] = (path, local_index)
                keep_indices.append(i)
            elif on_missing == "error":
                raise FileNotFoundError(
                    f"Missing or incomplete GrADS template file for time {i}: {path}"
                )
            elif on_missing == "nan":
                keep_indices.append(i)
    else:
        readable = _readable_times_for_file(desc, desc.dset)
        if readable < len(desc.times) and on_missing == "error":
            raise EOFError(
                f"Binary file has {readable} readable times, ctl declares {len(desc.times)}"
            )
        keep_indices = list(range(len(desc.times) if on_missing == "nan" else readable))
        available = {i: (desc.dset, i) for i in range(readable)}

    records: Dict[str, List[List[Record]]] = {}
    for var in desc.vars:
        var_records: List[List[Record]] = []
        start_plane = plane_order[var.name]
        for time_index in keep_indices:
            time_records: List[Record] = []
            for lev in range(var.planes):
                available_record = available.get(time_index)
                if available_record is None:
                    time_records.append(None)
                else:
                    path, file_time_index = available_record
                    off = _plane_offset(desc, file_time_index, start_plane + lev)
                    time_records.append((path, off))
            var_records.append(time_records)
        records[var.name] = var_records
    times = [desc.times[i] for i in keep_indices]
    return Layout(times, records, desc.nx * desc.ny * 4, (desc.ny, desc.nx))


def _normalize_key(key, ndim: int) -> Tuple[object, ...]:
    if hasattr(key, "tuple"):
        key = key.tuple
    if not isinstance(key, tuple):
        key = (key,)
    key = tuple(k for k in key if k is not Ellipsis)
    if len(key) < ndim:
        key = key + (slice(None),) * (ndim - len(key))
    return key


def _indices(selector: object, size: int) -> Tuple[List[int], bool]:
    if isinstance(selector, slice):
        return list(range(size))[selector], False
    if np.isscalar(selector):
        return [int(selector) % size], True
    arr = np.asarray(selector)
    return [int(v) % size for v in arr.ravel()], False


def _is_basic_selector(selector: object) -> bool:
    return isinstance(selector, slice) or np.isscalar(selector)


def _basic_subset_shape(
    y_sel: object, x_sel: object, ny: int, nx: int
) -> Tuple[int, ...]:
    shape: List[int] = []
    for selector, size in ((y_sel, ny), (x_sel, nx)):
        if isinstance(selector, slice):
            start, stop, step = selector.indices(size)
            shape.append(len(range(start, stop, step)))
        elif not np.isscalar(selector):
            raise TypeError("selector is not basic")
    return tuple(shape)


def _normalize_vector_indices(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    values = np.where(values < 0, values + size, values)
    if np.any((values < 0) | (values >= size)):
        raise IndexError("vectorized index is out of bounds")
    return values


class GradsBackendArray(BackendArray):
    def __init__(
        self,
        desc: CtlDescriptor,
        var: VarDef,
        records: List[List[Record]],
        *,
        mask_and_scale: bool = True,
    ):
        self.desc = desc
        self.var = var
        self.records = records
        self.mask_and_scale = mask_and_scale
        if var.levs > 0:
            self.shape = (len(records), var.planes, desc.ny, desc.nx)
        else:
            self.shape = (len(records), desc.ny, desc.nx)
        self.dtype = np.dtype("float32")

    def __getitem__(self, key):
        # LazilyVectorizedIndexedArray sends a VectorizedIndexer directly to
        # the backend. Handle it before the BASIC adapter so sparse (t,z,y,x)
        # point requests can go straight to their byte locations.
        if indexing is not None and isinstance(key, indexing.VectorizedIndexer):
            return self._raw_vectorized_indexing_method(key.tuple)
        if indexing is not None and not isinstance(key, tuple):
            return indexing.explicit_indexing_adapter(
                key,
                self.shape,
                indexing.IndexingSupport.BASIC,
                self._raw_indexing_method,
            )
        return self._raw_indexing_method(key)

    def _raw_vectorized_indexing_method(self, key) -> np.ndarray:
        if self.var.levs > 0:
            if len(key) != 4:
                raise IndexError("expected (time, lev, lat, lon) vectorized index")
            t_raw, z_raw, y_raw, x_raw = key
        else:
            if len(key) != 3:
                raise IndexError("expected (time, lat, lon) vectorized index")
            t_raw, y_raw, x_raw = key
            z_raw = np.asarray(0, dtype=np.int64)

        arrays = np.broadcast_arrays(
            np.asarray(t_raw), np.asarray(z_raw), np.asarray(y_raw), np.asarray(x_raw)
        )
        result_shape = arrays[0].shape
        t_idx = _normalize_vector_indices(arrays[0], self.shape[0]).ravel()
        z_idx = _normalize_vector_indices(arrays[1], self.var.planes).ravel()
        y_idx = _normalize_vector_indices(arrays[2], self.desc.ny).ravel()
        x_idx = _normalize_vector_indices(arrays[3], self.desc.nx).ravel()

        # records are stored in file order. Translate logical z/y coordinates
        # only when looking up the binary record/element.
        stored_z = (
            self.var.planes - 1 - z_idx
            if "zrev" in self.desc.options
            else z_idx
        )
        stored_y = (
            self.desc.ny - 1 - y_idx
            if "yrev" in self.desc.options
            else y_idx
        )

        out = np.empty(t_idx.size, dtype="float32")
        group_id = t_idx * self.var.planes + stored_z
        for gid in np.unique(group_id):
            positions = np.flatnonzero(group_id == gid)
            ti = int(t_idx[positions[0]])
            zi = int(stored_z[positions[0]])
            record = self.records[ti][zi]
            if record is None:
                out[positions] = np.nan
                continue

            path, byte_offset = record
            flat_index = stored_y[positions] * self.desc.nx + x_idx[positions]
            if byte_offset % 4 == 0:
                start = byte_offset // 4
                mm = DEFAULT_CACHE.get(path, self.desc.dtype)
                values = mm[start + flat_index]
            else:
                # Unaligned xyheader layouts are unusual. Preserve correctness
                # with a full-record fallback while keeping the common aligned
                # path sparse and zero-copy until the requested values are read.
                with path.open("rb") as f:
                    plane = np.fromfile(
                        f,
                        dtype=self.desc.dtype,
                        count=self.desc.nx * self.desc.ny,
                        offset=byte_offset,
                    )
                values = plane[flat_index]

            values = np.asarray(values, dtype="float32")
            if self.mask_and_scale:
                values = np.array(values, dtype="float32", copy=True)
                values[values == np.float32(self.desc.undef)] = np.nan
            out[positions] = values

        return out.reshape(result_shape)

    def _raw_indexing_method(self, key):
        key = _normalize_key(key, len(self.shape))
        t_sel = key[0]
        if self.var.levs > 0:
            z_sel, y_sel, x_sel = key[1], key[2], key[3]
        else:
            z_sel, y_sel, x_sel = 0, key[1], key[2]
        t_idx, t_scalar = _indices(t_sel, self.shape[0])
        if self.var.levs > 0:
            z_idx, z_scalar = _indices(z_sel, self.var.planes)
        else:
            z_idx, z_scalar = [0], True

        # xarray's BASIC adapter calls this method with scalar/slice selectors.
        # Push those horizontal selectors into the plane reader so that dtype
        # conversion and undef masking operate only on the requested subset.
        # Keep the previous full-plane path for direct non-basic tuple access.
        if _is_basic_selector(y_sel) and _is_basic_selector(x_sel):
            subset_shape = _basic_subset_shape(
                y_sel, x_sel, self.desc.ny, self.desc.nx
            )
            out = np.empty(
                (len(t_idx), len(z_idx)) + subset_shape, dtype="float32"
            )
            for oi, ti in enumerate(t_idx):
                for oj, zi in enumerate(z_idx):
                    out[oi, oj] = self._read_plane(
                        ti, zi, y_sel=y_sel, x_sel=x_sel
                    )
        else:
            out = np.empty(
                (len(t_idx), len(z_idx), self.desc.ny, self.desc.nx),
                dtype="float32",
            )
            for oi, ti in enumerate(t_idx):
                for oj, zi in enumerate(z_idx):
                    out[oi, oj] = self._read_plane(ti, zi)
            out = out[(slice(None), slice(None), y_sel, x_sel)]

        if self.var.levs == 0:
            out = out[:, 0, ...]
            if t_scalar:
                out = out[0]
        else:
            if t_scalar and z_scalar:
                out = out[0, 0]
            elif t_scalar:
                out = out[0]
            elif z_scalar:
                out = out[:, 0]
        return out

    def _read_plane(
        self,
        time_index: int,
        z_index: int,
        y_sel: object = slice(None),
        x_sel: object = slice(None),
    ) -> np.ndarray:
        if "zrev" in self.desc.options:
            z_index = self.var.planes - 1 - z_index
        record = self.records[time_index][z_index]
        if record is None:
            shape = _basic_subset_shape(
                y_sel, x_sel, self.desc.ny, self.desc.nx
            )
            return np.full(shape, np.nan, dtype="float32")
        path, byte_offset = record
        count = self.desc.nx * self.desc.ny
        if byte_offset % 4 == 0:
            start = byte_offset // 4
            mm = DEFAULT_CACHE.get(path, self.desc.dtype)
            plane = np.asarray(mm[start : start + count]).reshape(
                self.desc.ny, self.desc.nx
            )
        else:
            with path.open("rb") as f:
                plane = np.fromfile(
                    f, dtype=self.desc.dtype, count=count, offset=byte_offset
                ).reshape(self.desc.ny, self.desc.nx)

        if "yrev" in self.desc.options:
            plane = plane[::-1, :]
        plane = plane[y_sel, x_sel]

        # NumPy 2.x treats copy=False as a strict no-copy request. Scalar
        # indexing can no longer satisfy np.array(..., copy=False), so use
        # np.asarray here: it still avoids copies when possible but permits one
        # when dtype/byte-order conversion or a 0-D result requires it.
        plane = np.asarray(plane, dtype="float32")
        if self.mask_and_scale:
            plane = np.array(plane, dtype="float32", copy=True)
            plane[plane == np.float32(self.desc.undef)] = np.nan
        return plane


def open_dataset(
    filename_or_obj: str | Path,
    *,
    on_missing: str = "nan",
    mask_and_scale: bool = True,
    chunks=None,
    **kwargs,
) -> xr.Dataset:
    desc = open_ctl(filename_or_obj)
    layout = build_layout(desc, on_missing=on_missing)
    lat = np.asarray(desc.ydef.values, dtype="float32")
    lev = np.asarray(desc.zdef.values, dtype="float32")
    coords = {
        "time": ("time", np.asarray(layout.times), {"long_name": "time"}),
        "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
        "lon": (
            "lon",
            np.asarray(desc.xdef.values, dtype="float32"),
            {"units": "degrees_east", "long_name": "longitude"},
        ),
    }
    if any(v.levs > 0 for v in desc.vars):
        coords["lev"] = ("lev", lev, {"long_name": "vertical level"})
    data_vars = {}
    for var in desc.vars:
        arr = GradsBackendArray(
            desc, var, layout.records[var.name], mask_and_scale=mask_and_scale
        )
        data = indexing.LazilyIndexedArray(arr) if indexing is not None else arr
        if var.levs > 0:
            dims = ("time", "lev", "lat", "lon")
        else:
            dims = ("time", "lat", "lon")
        data_vars[var.name] = (
            dims,
            data,
            {"long_name": var.description, "units": var.units, "undef": desc.undef},
        )
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"title": desc.title})
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds
