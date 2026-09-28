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
from dataclasses import dataclass, field

_CLOCK = re.compile(r"\b\d{1,3}\+\d{1,2}(?::\d{2})?\b|\b\d{1,3}:\d{2}\b")
_FORMATION = re.compile(r"\b\d-\d(?:-\d){1,3}\b")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?(\s?%| percent)?")
_ID_LIKE = re.compile(r"\b[A-Za-z]+[_-]?\d+\b")  # player ids like home_11, P3573


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
