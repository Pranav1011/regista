"""Metrica Sports sample-game event data: the labels used to score pass detection.

kloppy parses only Metrica's JSON event format (game 3), so both formats are
read here directly: the raw CSV for games 1-2 and the JSON for game 3. Files
are downloaded once into ``data/raw/metrica/``.

Events are returned in canonical coordinates with the same per-period
direction flips as the tracking data. Metrica records completed passes as
PASS; failed passes appear as BALL LOST.

Known label error: in game 3's second half, the event file swaps players P3573
and P3580 relative to the tracking file. Every event credited to one of them
starts exactly on the other's tracked position (median 0.0 m, vs 21 m from
their own track). ``ID_CORRECTIONS`` fixes this in ``from_player`` /
``to_player``; the published ids are kept in ``from_player_raw`` /
``to_player_raw`` so uncorrected scores stay reproducible. The evidence is
re-checked by ``eval/_common.py::event_id_mismatches``.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from regista import io
from regista.ingest._common import to_metres
from regista.schema import Source

_BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
_FILES = {
    1: "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    2: "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    3: "Sample_Game_3/Sample_Game_3_events.json",
}

# game -> period -> {published id: id matching the tracking data}
ID_CORRECTIONS: dict[int, dict[int, dict[str, str]]] = {
    3: {2: {"P3573": "P3580", "P3580": "P3573"}},
}

EVENT_COLUMNS = [
    "period",
    "type",
    "subtype",
    "team",
    "from_player",
    "to_player",
    "from_player_raw",
    "to_player_raw",
    "start_frame",
    "end_frame",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def download(game: int) -> Path:
    """Fetch the raw event file for a sample game (cached). Raises on network failure."""
    rel = _FILES[game]
    path = io.raw_dir(Source.METRICA) / Path(rel).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{_BASE_URL}/{rel}", tmp)
        tmp.rename(path)
    return path


def parse_csv(path: Path) -> pd.DataFrame:
    """Games 1-2. Player 'Player11' of team 'Home' becomes 'home_11', matching tracking ids.

    The CSV's y axis points up, unlike kloppy's tracking frame (y down), so it is
    flipped here to match before the shared conversion.
    """
    raw = pd.read_csv(path)
    team = raw["Team"].str.lower()

    def player(col: str) -> pd.Series:
        num = raw[col].str.extract(r"Player(\d+)", expand=False)
        return (team + "_" + num).where(num.notna())

    return pd.DataFrame(
        {
            "period": raw["Period"],
            "type": raw["Type"],
            "subtype": raw["Subtype"],
            "team": team,
            "from_player": player("From"),
            "to_player": player("To"),
            "start_frame": raw["Start Frame"],
            "end_frame": raw["End Frame"],
            "start_x": raw["Start X"],
            "start_y": 1.0 - raw["Start Y"],
            "end_x": raw["End X"],
            "end_y": 1.0 - raw["End Y"],
        }
    )


def _subtype(value: object) -> str | None:
    if not value:
        return None
    items = value if isinstance(value, list) else [value]
    return "-".join(item["name"] for item in items)


def parse_json(path: Path, players: pd.DataFrame) -> pd.DataFrame:
    """Game 3. Team is resolved from the acting player via the tracking players table."""
    team_of = dict(zip(players["player_id"], players["team"], strict=True))
    rows = []
    for e in json.loads(path.read_text())["data"]:
        start, end = e.get("start") or {}, e.get("end") or {}
        from_id = (e.get("from") or {}).get("id")
        to_id = (e.get("to") or {}).get("id")
        rows.append(
            {
                "period": e["period"],
                "type": e["type"]["name"],
                "subtype": _subtype(e.get("subtypes")),
                "team": team_of.get(from_id),
                "from_player": from_id,
                "to_player": to_id,
                "start_frame": start.get("frame"),
                "end_frame": end.get("frame"),
                "start_x": start.get("x"),
                "start_y": start.get("y"),
                "end_x": end.get("x"),
                "end_y": end.get("y"),
            }
        )
    return pd.DataFrame(rows)


def to_canonical(events: pd.DataFrame, flipped_periods: list[int]) -> pd.DataFrame:
    """Convert normalised event coordinates to canonical metres with direction flips."""
    out = events.copy()
    flip = np.where(out["period"].isin(flipped_periods), -1.0, 1.0)
    for end in ("start", "end"):
        x, y = to_metres(out[f"{end}_x"].astype(float), out[f"{end}_y"].astype(float))
        out[f"{end}_x"], out[f"{end}_y"] = x * flip, y * flip
    out = out.astype(
        {
            "period": "int8",
            "type": "string",
            "subtype": "string",
            "team": "string",
            "from_player": "string",
            "to_player": "string",
            "from_player_raw": "string",
            "to_player_raw": "string",
            "start_frame": "Int64",
            "end_frame": "Int64",
        }
    )
    return out[EVENT_COLUMNS].reset_index(drop=True)


def load(game: int, players: pd.DataFrame, flipped_periods: list[int]) -> pd.DataFrame:
    path = download(game)
    raw = parse_json(path, players) if path.suffix == ".json" else parse_csv(path)
    unknown_team = raw["from_player"].notna() & raw["team"].isna()
    if unknown_team.any():
        raise ValueError(f"game {game}: {int(unknown_team.sum())} events by unknown players")
    return to_canonical(correct_ids(raw, ID_CORRECTIONS.get(game, {})), flipped_periods)


def correct_ids(events: pd.DataFrame, corrections: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Apply per-period player-id corrections, keeping the published ids as ``*_raw``."""
    out = events.copy()
    for col in ("from_player", "to_player"):
        out[f"{col}_raw"] = out[col]
        for period, mapping in corrections.items():
            in_period = out["period"] == period
            out.loc[in_period, col] = out.loc[in_period, col].replace(mapping)
    return out
