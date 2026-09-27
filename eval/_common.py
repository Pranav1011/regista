"""Shared helpers for eval scripts: load (ingesting if needed) and prepare real data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regista import io
from regista.analytics.kinematics import add_velocities, frame_interval
from regista.analytics.possession import ball_owner, match_radius, team_in_possession
from regista.schema import Source

EVAL_DIR = Path(__file__).resolve().parent
PARAMS_PATH = EVAL_DIR / "params_phase1.json"
REPORTS_DIR = EVAL_DIR / "reports"

TRAIN_GAMES = (1, 2)
TEST_GAME = 3
TUNED_FPS = 25.0  # Metrica frame rate; frame-count parameters were tuned at this rate

# Every match in SkillCorner's open data (github.com/SkillCorner/opendata, MIT).
SKILLCORNER_MATCHES = (
    "1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846",
    "1986691", "1996435", "1996436", "2006229", "2006363", "2007448", "2007721",
    "2010085", "2011166", "2013725", "2015213", "2016236", "2017461",
)  # fmt: skip


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


def ensure_skillcorner(match_id: str) -> None:
    """Ingest a SkillCorner match if its processed files are missing."""
    if not io.frames_path(Source.SKILLCORNER, match_id).exists():
        from regista.ingest import skillcorner

        skillcorner.ingest(match_id)


def skillcorner_match(match_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frames with velocities, players table) for one SkillCorner match."""
    ensure_skillcorner(match_id)
    frames = add_velocities(io.read_frames(Source.SKILLCORNER, match_id))
    return frames, io.read_players(Source.SKILLCORNER, match_id)


def scaled_frames(n_frames_at_tuned_fps: int | None, frames: pd.DataFrame) -> int | None:
    """Convert a frame count tuned at 25 fps to the same duration at this data's rate."""
    if n_frames_at_tuned_fps is None:
        return None
    return max(int(round(n_frames_at_tuned_fps / TUNED_FPS / frame_interval(frames))), 1)


def v2_owner(frames: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, float]:
    """Frame-level owner with the v2 per-match radius; returns (owner, radius)."""
    radius = match_radius(frames, params["v2"]["radius_quantile"])
    return ball_owner(frames, radius, params["v2"]["max_ball_speed"]), radius


def possession_phase(frames: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per-frame team in possession, using the v2 owner and its max gap."""
    owner, _ = v2_owner(frames, params)
    return team_in_possession(owner, scaled_frames(params["v2"]["max_gap_frames"], frames))
