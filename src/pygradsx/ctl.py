from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gradstime import build_times


@dataclass(frozen=True)
class AxisDef:
    name: str
    size: int
    values: np.ndarray
    kind: str


@dataclass(frozen=True)
class VarDef:
    name: str
    levs: int
    units: str
    description: str

    @property
    def planes(self) -> int:
        return self.levs if self.levs > 0 else 1


@dataclass
class CtlDescriptor:
    path: Path
    dset: Path
    title: str = ""
    undef: float = -9.99e8
    options: set = field(default_factory=set)
    xdef: Optional[AxisDef] = None
    ydef: Optional[AxisDef] = None
    zdef: Optional[AxisDef] = None
    times: Sequence[object] = field(default_factory=list)
    vars: List[VarDef] = field(default_factory=list)
    fileheader: int = 0
    theader: int = 0
    xyheader: int = 0

    @property
    def nx(self) -> int:
        return _required_axis(self.xdef, "xdef").size

    @property
    def ny(self) -> int:
        return _required_axis(self.ydef, "ydef").size

    @property
    def nz(self) -> int:
        return _required_axis(self.zdef, "zdef").size

    @property
    def template(self) -> bool:
        return "template" in self.options

    @property
    def sequential(self) -> bool:
        return "sequential" in self.options

    @property
    def dtype(self) -> np.dtype:
        endian = "<" if "little_endian" in self.options else ">"
        if "big_endian" not in self.options and "little_endian" not in self.options:
            endian = "<"
        dtype = np.dtype(endian + "f4")
        if "byteswapped" in self.options:
            dtype = dtype.newbyteorder("S")
        return dtype


def _required_axis(axis: Optional[AxisDef], name: str) -> AxisDef:
    if axis is None:
        raise ValueError(f"Missing required {name} in ctl")
    return axis


def _strip_comment(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("*"):
        return ""
    return stripped


def _parse_axis(name: str, parts: List[str], extra_lines: List[str]) -> AxisDef:
    if len(parts) < 4:
        raise ValueError(f"Invalid {name}: {' '.join(parts)}")
    size = int(parts[1])
    kind = parts[2].lower()
    tokens = parts[3:] + extra_lines
    if kind == "linear":
        if len(tokens) < 2:
            raise ValueError(f"Invalid linear {name}")
        start = float(tokens[0])
        inc = float(tokens[1])
        values = start + inc * np.arange(size, dtype=np.float32)
    elif kind == "levels":
        if len(tokens) < size:
            raise ValueError(f"{name} levels expected {size}, got {len(tokens)}")
        values = np.array([float(v) for v in tokens[:size]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported {name} definition kind: {kind}")
    return AxisDef(name, size, values, kind)


def _resolve_dset(ctl_path: Path, dset: str) -> Path:
    raw = dset.strip()
    if raw.startswith("^"):
        return ctl_path.parent / raw[1:]
    return Path(raw) if Path(raw).is_absolute() else ctl_path.parent / raw


def open_ctl(path: str | Path) -> CtlDescriptor:
    ctl_path = Path(path).resolve()
    lines = ctl_path.read_text(encoding="utf-8").splitlines()
    desc = CtlDescriptor(path=ctl_path, dset=ctl_path.parent)
    in_vars = False
    expected_vars = 0
    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i])
        i += 1
        if not line:
            continue
        parts = line.split()
        key = parts[0].lower()
        if in_vars:
            if key == "endvars":
                in_vars = False
                continue
            if len(parts) < 3:
                raise ValueError(f"Invalid vars line: {line}")
            desc.vars.append(
                VarDef(parts[0], int(parts[1]), parts[2], " ".join(parts[3:]))
            )
            continue
        if key == "dset":
            desc.dset = _resolve_dset(ctl_path, " ".join(parts[1:]))
        elif key == "title":
            desc.title = " ".join(parts[1:])
        elif key == "undef":
            desc.undef = float(parts[1])
        elif key == "options":
            known = {
                "template",
                "little_endian",
                "big_endian",
                "byteswapped",
                "yrev",
                "zrev",
                "sequential",
                "365_day_calendar",
            }
            for opt in (p.lower() for p in parts[1:]):
                if opt not in known:
                    warnings.warn(f"Unknown GrADS option ignored: {opt}", RuntimeWarning)
                desc.options.add(opt)
        elif key in {"fileheader", "theader", "xyheader"}:
            setattr(desc, key, int(parts[1]))
        elif key in {"xdef", "ydef", "zdef"}:
            size = int(parts[1])
            kind = parts[2].lower()
            needed = max(0, size - len(parts[3:])) if kind == "levels" else 0
            extra: List[str] = []
            while needed > 0 and i < len(lines):
                more = _strip_comment(lines[i]).split()
                i += 1
                extra.extend(more)
                needed = size - (len(parts[3:]) + len(extra))
            setattr(desc, key, _parse_axis(key, parts, extra))
        elif key == "tdef":
            if len(parts) < 5 or parts[2].lower() != "linear":
                raise ValueError("Only tdef linear is supported")
            desc.times = build_times(
                int(parts[1]),
                parts[3],
                parts[4],
                noleap="365_day_calendar" in desc.options,
            )
        elif key == "vars":
            expected_vars = int(parts[1])
            in_vars = True
        elif key == "edef":
            members = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if members > 1:
                raise NotImplementedError("Only a single ensemble member is supported")
        elif key == "pdef":
            raise NotImplementedError("pdef projection grids are not supported in v0.1")
    if in_vars:
        raise ValueError("vars section was not closed with endvars")
    if expected_vars and len(desc.vars) != expected_vars:
        raise ValueError(f"Expected {expected_vars} variables, got {len(desc.vars)}")
    _required_axis(desc.xdef, "xdef")
    _required_axis(desc.ydef, "ydef")
    if desc.zdef is None:
        desc.zdef = AxisDef("zdef", 1, np.array([0.0], dtype=np.float32), "levels")
    if not len(desc.times):
        raise ValueError("Missing required tdef in ctl")
    if not desc.vars:
        raise ValueError("Missing vars section in ctl")
    return desc
