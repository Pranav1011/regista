"""Shared helpers for eval scripts: load (ingesting if needed) and prepare real data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regista import io
from regista.analytics.kinematics import add_velocities
from regista.schema import Source

EVAL_DIR = Path(__file__).resolve().parent
PARAMS_PATH = EVAL_DIR / "params_phase1.json"
REPORTS_DIR = EVAL_DIR / "reports"

TRAIN_GAMES = (1, 2)
TEST_GAME = 3


def ensure_metrica(game: int) -> None:
    """Ingest a Metrica game if its processed files are missing."""
    match_id = str(game)
    if not (
        io.frames_path(Source.METRICA, match_id).exists()
        and io.events_path(Source.METRICA, match_id).exists()
    ):
        from regista.ingest import metrica

        metrica.ingest(game)


def metrica_game(game: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frames with velocities, events) for one Metrica sample game."""
    ensure_metrica(game)
    frames = add_velocities(io.read_frames(Source.METRICA, str(game)))
    return frames, io.read_events(Source.METRICA, str(game))


def true_passes(events: pd.DataFrame) -> pd.DataFrame:
    """Metrica's labelled completed passes (type PASS) with a known passer and receiver."""
    passes = events[events["type"] == "PASS"]
    return passes.dropna(subset=["start_frame", "from_player", "to_player"]).reset_index(drop=True)


def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{PARAMS_PATH} missing; run `uv run python eval/tune_possession.py`"
        )
    return json.loads(PARAMS_PATH.read_text())
