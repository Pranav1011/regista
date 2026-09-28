"""Shared helpers for eval scripts: load (ingesting if needed) and prepare real data."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regista import io
from regista.analytics.kinematics import add_velocities
from regista.pipeline import TUNED_FPS, possession_phase, scaled_frames, v2_owner  # noqa: F401
from regista.schema import Source

EVAL_DIR = Path(__file__).resolve().parent
PARAMS_PATH = EVAL_DIR / "params_phase1.json"
REPORTS_DIR = EVAL_DIR / "reports"

TRAIN_GAMES = (1, 2)
TEST_GAME = 3

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


@functools.lru_cache(maxsize=3)
def metrica_game(game: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frames with velocities, events) for one Metrica sample game.

    Cached: callers must not modify the returned tables in place.
    """
    ensure_metrica(game)
    frames = add_velocities(io.read_frames(Source.METRICA, str(game)))
    return frames, io.read_events(Source.METRICA, str(game))


def true_passes(events: pd.DataFrame, labels: str = "corrected") -> pd.DataFrame:
    """Metrica's labelled completed passes (type PASS) with a known passer and receiver.

    ``labels="corrected"`` uses player ids after ``metrica_events.ID_CORRECTIONS``;
    ``"raw"`` uses the ids as published.
    """
    if labels not in ("corrected", "raw"):
        raise ValueError(f"labels must be 'corrected' or 'raw', got {labels!r}")
    passes = events[events["type"] == "PASS"]
    if labels == "raw":
        passes = passes.assign(
            from_player=passes["from_player_raw"], to_player=passes["to_player_raw"]
        )
    return passes.dropna(subset=["start_frame", "from_player", "to_player"]).reset_index(drop=True)


def event_id_mismatches(
    frames: pd.DataFrame, events: pd.DataFrame, player_col: str = "from_player"
) -> pd.DataFrame:
    """Player-halves whose events start nearest to a different tracked player.

    For each event with a start position, finds the tracked player closest to
    that position at the start frame. A player-half is flagged when the most
    common nearest player is not the credited player. An empty result means
    event ids and tracking ids agree.
    """
    e = events.dropna(subset=["start_frame", "start_x", player_col])
    e = e[["period", "start_frame", player_col, "start_x", "start_y"]].rename(
        columns={"start_frame": "frame", player_col: "credited"}
    )
    e = e.astype({"frame": "int64"}).reset_index(drop=True).rename_axis("event").reset_index()
    players = frames.loc[frames["team"] != "ball", ["period", "frame", "player_id", "x", "y"]]
    pairs = e.merge(players, on=["period", "frame"])
    pairs["dist"] = np.hypot(pairs["x"] - pairs["start_x"], pairs["y"] - pairs["start_y"])
    nearest = pairs.loc[pairs.groupby("event")["dist"].idxmin()]
    summary = nearest.groupby(["period", "credited"]).agg(
        events=("event", "size"),
        nearest_player=("player_id", lambda s: s.value_counts().index[0]),
        share=("player_id", lambda s: s.value_counts().iloc[0] / len(s)),
        median_dist_to_nearest=("dist", "median"),
    )
    summary = summary.reset_index()
    return summary[summary["nearest_player"] != summary["credited"]].reset_index(drop=True)


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
