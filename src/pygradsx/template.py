from __future__ import annotations

from datetime import datetime


def expand_template(pattern: str, when: object) -> str:
    unsupported = [token for token in ("%e", "%ch") if token in pattern]
    if unsupported:
        joined = ", ".join(unsupported)
        raise NotImplementedError(f"Unsupported GrADS template token(s): {joined}")
    if not isinstance(when, datetime):
        year = int(getattr(when, "year"))
        month = int(getattr(when, "month"))
        day = int(getattr(when, "day"))
        hour = int(getattr(when, "hour", 0))
        minute = int(getattr(when, "minute", 0))
    else:
        year, month, day, hour, minute = (
            when.year,
            when.month,
            when.day,
            when.hour,
            when.minute,
        )
    repl = {
        "%y4": f"{year:04d}",
        "%y2": f"{year % 100:02d}",
        "%m2": f"{month:02d}",
        "%m1": f"{month}",
        "%d2": f"{day:02d}",
        "%d1": f"{day}",
        "%h2": f"{hour:02d}",
        "%h1": f"{hour}",
        "%n2": f"{minute:02d}",
        "%n1": f"{minute}",
    }
    result = pattern
    for key in sorted(repl, key=len, reverse=True):
        result = result.replace(key, repl[key])
    return result
