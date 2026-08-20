# PyGradsX

**Lazy, memory-efficient xarray reader for GrADS datasets (`.ctl` + flat binary).**

PyGradsX opens GrADS control files as `xarray.Dataset` objects with fully lazy
loading: opening a dataset parses only the descriptor, and data records are read
from disk on demand via `numpy.memmap`.

## Why PyGradsX?

- **Tolerant of incomplete time axes.** Unlike other GrADS readers (e.g.
  xgrads), PyGradsX does not fail when the files behind `tdef` are incomplete.
  Missing template files, partially written files, or a single binary shorter
  than `tdef` are handled gracefully via the `on_missing` option:
  - `"nan"` (default) — missing time steps are filled with `NaN`, lazily
  - `"skip"` — the time axis is built from the time steps that actually exist
  - `"error"` — strict mode, raise on any inconsistency
- **Memory efficient.** Nothing is loaded at open time. Indexing reads only the
  requested records through memory-mapped files with an LRU handle cache, so
  datasets far larger than RAM can be sliced cheaply.
- **Fast.** Data stays `float32` end to end (no silent float64 promotion), byte
  swapping is applied only to the slices you access, and `undef -> NaN`
  conversion can be disabled entirely with `mask_and_scale=False`.
- **Plays well with dask.** Pass `chunks=` to get a dask-backed dataset;
  dask is an optional dependency.

## Installation

```bash
pip install .            # from a source checkout
pip install .[test]      # with pytest + dask for the test suite
```

Requires Python >= 3.9. Hard dependencies: `numpy` and `xarray` only.
Optional: `dask` (chunked/parallel access), `cftime` (365-day calendars).

## Quick start

```python
import pygradsx

ds = pygradsx.open_dataset("model.ctl")                     # missing steps -> NaN
ds = pygradsx.open_dataset("model.ctl", on_missing="skip")  # keep existing steps only
ds = pygradsx.open_dataset("model.ctl", mask_and_scale=False)  # raw undef values
```

PyGradsX also registers itself as an xarray backend, so this works too
(the engine is auto-detected for `.ctl` files):

```python
import xarray as xr

ds = xr.open_dataset("model.ctl")
ds = xr.open_dataset("model.ctl", engine="pygradsx", chunks={"time": 10})
```

Coordinates are named `time`, `lev`, `lat`, `lon` with CF-style attributes.

## Supported GrADS features

- `dset` (including `^` relative paths and filename templates such as
  `%y4 %m2 %d2 %h2 %n2`), `title`, `undef`, `fileheader`, `theader`, `xyheader`
- `options`: `template`, `little_endian`, `big_endian`, `byteswapped`,
  `yrev`, `zrev`, `sequential`, `365_day_calendar`
  (unknown options warn instead of failing)
- `xdef`, `ydef`, `zdef`: both `linear` and `levels`
- `tdef linear` with `mn`, `hr`, `dy`, `mo`, `yr` increments
- `vars ... endvars`, including multi-level variables
- Templates that map **multiple time steps to one file** (e.g. monthly files
  containing daily records); a truncated last file yields `NaN` only for the
  records that are actually missing

### Out of scope in v0.1

`pdef` projection grids, multi-member `edef` ensembles, and the `%e` / `%ch`
template tokens raise `NotImplementedError` explicitly.

## Handling incomplete data

```text
dset ^data_%y4%m2.bin
tdef 91 linear 00Z01jan2020 1dy
```

If `data_202003.bin` is absent and `data_202002.bin` contains only the first
10 days:

```python
ds = pygradsx.open_dataset("data.ctl")                    # 91 steps, gaps are NaN
ds = pygradsx.open_dataset("data.ctl", on_missing="skip") # 41 steps (31 + 10)
ds = pygradsx.open_dataset("data.ctl", on_missing="error")  # raises
```

File existence is checked once at open time; no data is read until you index.

## Development

```bash
pip install -e .[test]
pytest
```

The test suite generates all GrADS fixtures synthetically — no binary data is
shipped in the repository.

## License

MIT
