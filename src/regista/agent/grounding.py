"""Deterministic number-grounding check for agent answers.

Every number in an answer must appear in the tool outputs (or the question),
allowing for rounding: a number written with d decimals is grounded if some
source value rounds to it at d decimals, and a percentage is grounded if some
source fraction does so after multiplying by 100. Match clocks ("62:03",
"45+2:00") and formation labels ("4-4-2") are compared as whole tokens.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

_CLOCK = re.compile(r"\b\d{1,3}\+\d{1,2}(?::\d{2})?\b|\b\d{1,3}:\d{2}\b")
_FORMATION = re.compile(r"\b\d-\d(?:-\d){1,3}\b")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?(\s?%| percent)?")
_ID_LIKE = re.compile(r"\b[A-Za-z]+[_-]?\d+\b")  # player ids like home_11, P3573


_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def normalise(text: str) -> str:
    """Unicode dashes to '-', and '90 + 3:48' to '90+3:48', before any matching."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\b(45|90)\s*\+\s*(\d)", r"\1+\2", text)


@dataclass
class GroundingResult:
    grounded: bool
    checked: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def _source_values(sources: list) -> tuple[set[str], list[float]]:
    """Tokens (clocks, formations, ids) and numeric values from tool outputs and question."""
    text = " ".join(s if isinstance(s, str) else json.dumps(s, default=str) for s in sources)
    tokens = set(_CLOCK.findall(text)) | set(_FORMATION.findall(text)) | set(_ID_LIKE.findall(text))
    stripped = _ID_LIKE.sub(" ", _FORMATION.sub(" ", _CLOCK.sub(" ", text)))
    values = []
    for m in _NUMBER.finditer(stripped):
        try:
            values.append(float(re.sub(r"\s?(%|percent)$", "", m.group(0)).replace(",", "")))
        except ValueError:
            continue
    return tokens, values


def _decimals(num: str) -> int:
    return len(num.split(".")[1]) if "." in num else 0


def _matches(value: float, decimals: int, percent: bool, sources: list[float]) -> bool:
    for s in sources:
        for c in [s, s * 100] if percent else [s]:
            if round(c, decimals) == round(value, decimals):
                return True
            if abs(c - value) <= 0.5 * 10 ** (-decimals) + 1e-9:
                return True
    return False


def check_numbers(answer: str, sources: list) -> GroundingResult:
    """Check that every number in ``answer`` appears in ``sources`` (tool outputs, question)."""
    answer = normalise(answer)
    tokens, values = _source_values(sources)
    checked, ungrounded = [], []
    for pattern in (_CLOCK, _FORMATION, _ID_LIKE):
        for tok in pattern.findall(answer):
            if pattern is _ID_LIKE:
                continue  # ids are not numbers; unknown ids are caught by the tools
            checked.append(tok)
            if tok not in tokens:
                ungrounded.append(tok)
    rest = _ID_LIKE.sub(" ", _FORMATION.sub(" ", _CLOCK.sub(" ", answer)))
    for m in _NUMBER.finditer(rest):
        raw = m.group(0)
        percent = m.group(1) is not None
        num = raw.replace(",", "").replace("percent", "").replace("%", "").strip()
        try:
            value = float(num)
        except ValueError:
            continue
        checked.append(raw.strip())
        if not _matches(value, _decimals(num), percent, values):
            ungrounded.append(raw.strip())
    return GroundingResult(grounded=not ungrounded, checked=checked, ungrounded=ungrounded)


CLOCK_TOLERANCE_S = 60.0


def _clock_instant(clock: str) -> tuple[int, float] | None:
    """A written match clock as (period, seconds into the period); None if malformed."""
    from regista.clock import PERIOD_LENGTH_S, parse_clock

    try:
        period, seconds = parse_clock(clock)
    except ValueError:
        return None
    if period is not None:
        return period, seconds
    if seconds < PERIOD_LENGTH_S:
        return 1, seconds
    return 2, seconds - PERIOD_LENGTH_S


def check_clocks(answer: str, evidence: list[dict], exempt_texts: list[str]) -> list[str]:
    """Match clocks in ``answer`` that fall outside every evidence range (with 1 min slack).

    Clocks that appear verbatim in ``exempt_texts`` (the question, tool error
    messages) are allowed: they are the user's own reference or a stated limit.
    """
    ranges = []
    for ev in evidence:
        a, b = _clock_instant(ev["clock_start"]), _clock_instant(ev["clock_end"])
        if a and b:
            ranges.append((ev["period"], a[1], b[1]))
    exempt = " ".join(normalise(t) for t in exempt_texts)
    uncited = []
    for clock in _CLOCK.findall(normalise(answer)):
        if clock in exempt:
            continue
        at = _clock_instant(clock)
        if at is None or not any(
            p == at[0] and lo - CLOCK_TOLERANCE_S <= at[1] <= hi + CLOCK_TOLERANCE_S
            for p, lo, hi in ranges
        ):
            uncited.append(clock)
    return uncited
