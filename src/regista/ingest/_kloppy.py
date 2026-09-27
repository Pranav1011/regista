"""Helpers for turning a kloppy TrackingDataset into canonical tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from kloppy.domain import Ground, TrackingDataset

from regista import io
from regista.ingest._common import (
    drop_out_of_bounds,
    home_attack_flips,
    normalise_direction,
    wide_to_long,
)
from regista.schema import Source, Team, validate_frames

_GROUND_TO_TEAM = {Ground.HOME: Team.HOME, Ground.AWAY: Team.AWAY}


@dataclass
class IngestResult:
    frames: pd.DataFrame
    players: pd.DataFrame
    info: dict = field(default_factory=dict)  # provenance written next to the parquet

    def save(self) -> Path:
        """Write frames, players, and provenance; return the frames path."""
        source, match_id = Source(self.info["source"]), self.info["match_id"]
        path = io.write_frames(self.frames, source, match_id)
        io.write_players(self.players, source, match_id)
        io.write_info(self.info, source, match_id)
        return path


def players_table(dataset: TrackingDataset) -> pd.DataFrame:
    """One row per listed player: id, team, shirt number, provider position, starter flag."""
    rows = []
    for team in dataset.metadata.teams:
        for p in team.players:
            position = p.starting_position
            rows.append(
                {
                    "player_id": str(p.player_id),
                    "team": _GROUND_TO_TEAM[team.ground].value,
                    "jersey_no": p.jersey_no,
                    "position": None if position is None else position.name,
                    "starting": bool(p.starting),
                }
            )
    return pd.DataFrame(rows)


def dataset_to_canonical(dataset: TrackingDataset, match_id: str, source: Source) -> IngestResult:
    """kloppy tracking dataset -> validated canonical frames, home attacking +x."""
    wide = dataset.to_df()
    players = players_table(dataset)
    tracked = players[players["player_id"].map(lambda pid: f"{pid}_x" in wide.columns)]
    player_teams = {
        pid: Team(team) for pid, team in zip(tracked["player_id"], tracked["team"], strict=True)
    }
    frames = wide_to_long(wide, player_teams, match_id, source)
    flips = home_attack_flips(frames)
    frames = normalise_direction(frames, flips)
    frames, dropped = drop_out_of_bounds(frames)
    dims = dataset.metadata.pitch_dimensions
    info = {
        "source": source.value,
        "match_id": match_id,
        "frame_rate": float(dataset.metadata.frame_rate),
        "provider_pitch_m": [dims.pitch_length, dims.pitch_width],
        "flipped_periods": sorted(p for p, f in flips.items() if f),
        "rows": len(frames),
        "rows_dropped_out_of_bounds": dropped,
    }
    return IngestResult(validate_frames(frames), players, info)
