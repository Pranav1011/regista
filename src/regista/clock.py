"""Match clock: period + seconds since period start <-> a readable match time.

The second half's clock starts at 45:00, as on a broadcast; stoppage time in
the first half therefore shows past 45:00 while still in period 1.
"""

from __future__ import annotations

PERIOD_OFFSET_S = {1: 0.0, 2: 45 * 60.0}


def match_seconds(period: int, t: float) -> float:
    if period not in PERIOD_OFFSET_S:
        raise ValueError(f"unknown period {period}; expected 1 or 2")
    return PERIOD_OFFSET_S[period] + t


def match_clock(period: int, t: float) -> str:
    """'mm:ss' match clock for a time ``t`` seconds into ``period``."""
    total = int(match_seconds(period, t))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_clock(clock: str) -> float:
    """'mm:ss' (or 'mm') -> match seconds. Raises ValueError on malformed input."""
    parts = clock.strip().split(":")
    if len(parts) not in (1, 2) or not all(p.isdigit() for p in parts):
        raise ValueError(f"malformed match clock {clock!r}; expected 'mm:ss' or 'mm'")
    minutes = int(parts[0])
    seconds = int(parts[1]) if len(parts) == 2 else 0
    if seconds >= 60:
        raise ValueError(f"malformed match clock {clock!r}; seconds must be < 60")
    return minutes * 60.0 + seconds
