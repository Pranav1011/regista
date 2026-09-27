"""Canonical match-state schema.

Every data source (Metrica, SkillCorner, StatsBomb 360, and later Regista's own
computer-vision pipeline) is converted into this single long-format table.
Analytics code reads ONLY this format, so replacing open tracking data with CV
output requires zero changes in the analytics layer.

Coordinate convention
---------------------
- Units are metres on a 105 x 68 pitch.
- Origin is the centre spot.
- Direction is normalised so the HOME team attacks towards +x in every period
  (ingest adapters flip the second half as needed).
- +y points to the home team's left when attacking +x (right-handed axes).

One row = one object (player or ball) in one frame. Objects not visible in a
frame (common in broadcast or CV data) simply have no row for that frame.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
BOUNDS_MARGIN_M = 5.0  # objects may legitimately sit a little outside the lines


class Team(StrEnum):
    HOME = "home"
    AWAY = "away"
    BALL = "ball"


class Source(StrEnum):
    METRICA = "metrica"
    SKILLCORNER = "skillcorner"
    STATSBOMB360 = "statsbomb360"
    CV = "cv"


BALL_ID = "ball"

# column -> pandas dtype
FRAME_COLUMNS: dict[str, str] = {
    "match_id": "string",
    "period": "int8",
    "frame": "int64",
    "t": "float64",  # seconds since the start of the period
    "team": "string",  # a Team value
    "player_id": "string",  # stable within a match; BALL_ID for the ball
    "x": "float64",
    "y": "float64",
    "vx": "float64",  # m/s, NaN until kinematics has run
    "vy": "float64",
    "confidence": "float64",  # 1.0 for provider data, detector score for CV
    "source": "string",  # a Source value
}

KEY_COLUMNS = ["match_id", "period", "frame", "player_id"]


class SchemaError(ValueError):
    """Raised when a frame table violates the canonical schema."""


def empty_frames() -> pd.DataFrame:
    """Return an empty, correctly typed frame table."""
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in FRAME_COLUMNS.items()})


def coerce_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to canonical dtypes and order them. Raises on missing columns."""
    missing = [c for c in FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    return df[list(FRAME_COLUMNS)].astype(FRAME_COLUMNS)


def validate_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a frame table and return it with canonical dtypes.

    Fails loudly: a pipeline stage that produces invalid frames must raise,
    never silently continue with bad or placeholder data.
    """
    df = coerce_frames(df)

    bad_team = ~df["team"].isin([t.value for t in Team])
    if bad_team.any():
        raise SchemaError(f"invalid team values: {sorted(df.loc[bad_team, 'team'].unique())}")

    bad_source = ~df["source"].isin([s.value for s in Source])
    if bad_source.any():
        raise SchemaError(f"invalid source values: {sorted(df.loc[bad_source, 'source'].unique())}")

    ball_rows = df["team"] == Team.BALL.value
    if (ball_rows != (df["player_id"] == BALL_ID)).any():
        raise SchemaError("ball rows must have team='ball' and player_id='ball', and vice versa")

    dupes = df.duplicated(subset=KEY_COLUMNS)
    if dupes.any():
        raise SchemaError(f"{int(dupes.sum())} duplicate (match, period, frame, object) rows")

    x_lim = PITCH_LENGTH_M / 2 + BOUNDS_MARGIN_M
    y_lim = PITCH_WIDTH_M / 2 + BOUNDS_MARGIN_M
    located = df[["x", "y"]].notna().all(axis=1)
    out = located & ((df["x"].abs() > x_lim) | (df["y"].abs() > y_lim))
    if out.any():
        raise SchemaError(f"{int(out.sum())} rows outside pitch bounds (+/-{x_lim}, +/-{y_lim}) m")

    conf = df["confidence"]
    if ((conf < 0) | (conf > 1)).any() or conf.isna().any():
        raise SchemaError("confidence must be present and within [0, 1]")

    if not np.isfinite(df["t"]).all() or (df["t"] < 0).any():
        raise SchemaError("t must be finite and non-negative")

    return df
