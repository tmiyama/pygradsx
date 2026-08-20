from __future__ import annotations

import calendar
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence, Union

import numpy as np

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}


@dataclass(frozen=True)
class GradsIncrement:
    value: int
    unit: str


def parse_increment(text: str) -> GradsIncrement:
    match = re.fullmatch(r"\s*(\d+)\s*(mn|hr|dy|mo|yr)\s*", text.lower())
    if not match:
        raise ValueError(f"Invalid GrADS time increment: {text!r}")
    return GradsIncrement(int(match.group(1)), match.group(2))


def parse_grads_time(text: str) -> datetime:
    value = text.strip().lower()
    match = re.fullmatch(
        r"(?:(\d{1,2})(?::(\d{2}))?z)?(?:(\d{1,2}))?([a-z]{3})(\d{2,4})",
        value,
    )
    if not match:
        raise ValueError(f"Invalid GrADS time: {text!r}")
    hour = int(match.group(1) or 0)
    minute = int(match.group(2) or 0)
    day = int(match.group(3) or 1)
    month_text = match.group(4)
    year_text = match.group(5)
    month = MONTHS.get(month_text)
    if month is None:
        raise ValueError(f"Invalid GrADS month: {month_text!r}")
    year = int(year_text)
    if year < 100:
        year += 2000 if year < 50 else 1900
    return datetime(year, month, day, hour, minute)


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def add_increment(dt: datetime, inc: GradsIncrement, steps: int = 1) -> datetime:
    amount = inc.value * steps
    if inc.unit == "mn":
        return dt + timedelta(minutes=amount)
    if inc.unit == "hr":
        return dt + timedelta(hours=amount)
    if inc.unit == "dy":
        return dt + timedelta(days=amount)
    if inc.unit == "mo":
        return _add_months(dt, amount)
    if inc.unit == "yr":
        return _add_months(dt, amount * 12)
    raise ValueError(f"Unsupported GrADS increment unit: {inc.unit!r}")


def build_times(
    count: int,
    start: str,
    increment: str,
    *,
    noleap: bool = False,
) -> Sequence[Union[np.datetime64, object]]:
    start_dt = parse_grads_time(start)
    inc = parse_increment(increment)
    dts = [add_increment(start_dt, inc, i) for i in range(count)]
    if noleap:
        try:
            import cftime
        except Exception:
            warnings.warn(
                "cftime is not installed; using numpy datetime64 for 365_day_calendar",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            return [
                cftime.DatetimeNoLeap(dt.year, dt.month, dt.day, dt.hour, dt.minute)
                for dt in dts
            ]
    return np.array(dts, dtype="datetime64[ns]")

