"""Tests for the canonical frame schema. All data here is synthetic, by design."""

import pandas as pd
import pytest

from regista.schema import BALL_ID, SchemaError, empty_frames, validate_frames


def _frames(**overrides) -> pd.DataFrame:
    rows = [
        {"player_id": "h1", "team": "home", "x": -10.0, "y": 5.0},
        {"player_id": "a1", "team": "away", "x": 12.0, "y": -3.0},
        {"player_id": BALL_ID, "team": "ball", "x": 0.0, "y": 0.0},
    ]
    base = {
        "match_id": "synthetic",
        "period": 1,
        "frame": 0,
        "t": 0.0,
        "vx": float("nan"),
        "vy": float("nan"),
        "confidence": 1.0,
        "source": "metrica",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    for col, val in overrides.items():
        df.loc[0, col] = val
    return df


def test_valid_frames_pass():
    out = validate_frames(_frames())
    assert len(out) == 3
    assert str(out["frame"].dtype) == "int64"


def test_empty_frames_is_valid():
    assert validate_frames(empty_frames()).empty


def test_out_of_bounds_rejected():
    with pytest.raises(SchemaError, match="outside pitch bounds"):
        validate_frames(_frames(x=80.0))


def test_duplicate_rows_rejected():
    df = pd.concat([_frames(), _frames().iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_frames(df)


def test_ball_identity_enforced():
    with pytest.raises(SchemaError, match="ball rows"):
        validate_frames(_frames(team="ball"))


def test_unknown_source_rejected():
    with pytest.raises(SchemaError, match="invalid source"):
        validate_frames(_frames(source="made_up"))


def test_missing_column_rejected():
    with pytest.raises(SchemaError, match="missing columns"):
        validate_frames(_frames().drop(columns=["confidence"]))
