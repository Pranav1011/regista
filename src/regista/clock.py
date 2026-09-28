"""Match clock: period + seconds since period start <-> a readable match time.

The second half's clock starts at 45:00, as on a broadcast. Stoppage time is
written as ``45+m:ss`` (first half) or ``90+m:ss`` (second half), so a time is
never ambiguous between the halves.
"""

from __future__ import annotations

import re

PERIOD_OFFSET_S = {1: 0.0, 2: 45 * 60.0}
PERIOD_LENGTH_S = 45 * 60.0
_CLOCK = re.compile(r"^(\d+)(?::(\d{2}))?$|^(45|90)\+(\d+)(?::(\d{2}))?$")


def match_seconds(period: int, t: float) -> float:
    if period not in PERIOD_OFFSET_S:
        raise ValueError(f"unknown period {period}; expected 1 or 2")
    return PERIOD_OFFSET_S[period] + t


def match_clock(period: int, t: float) -> str:
    """'mm:ss' match clock for ``t`` seconds into ``period``; stoppage as '45+m:ss'."""
    match_seconds(period, t)
    if t >= PERIOD_LENGTH_S:
        extra = int(t - PERIOD_LENGTH_S)
        base = 45 if period == 1 else 90
        return f"{base}+{extra // 60}:{extra % 60:02d}"
    total = int(PERIOD_OFFSET_S[period] + t)
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_clock(clock: str) -> tuple[int | None, float]:
    """Parse a match clock into (period if implied, seconds since that period's start).

    '62:03' -> (None, 3723 match seconds): the caller resolves the period;
    '45+2:10' -> (1, 2830.0); '90+1:00' -> (2, 2760.0). Raises ValueError on
    malformed input.
    """
    m = _CLOCK.match(clock.strip())
    if not m:
        raise ValueError(f"malformed match clock {clock!r}; expected 'mm:ss', 'mm' or '45+m:ss'")
    seconds = m.group(2) if m.group(1) is not None else m.group(5)
    if seconds is not None and int(seconds) >= 60:
        raise ValueError(f"malformed match clock {clock!r}; seconds must be < 60")
    if m.group(1) is not None:
        return None, int(m.group(1)) * 60.0 + int(m.group(2) or 0)
    base, extra_min, extra_s = int(m.group(3)), int(m.group(4)), int(m.group(5) or 0)
    period = 1 if base == 45 else 2
    return period, PERIOD_LENGTH_S + extra_min * 60.0 + extra_s
