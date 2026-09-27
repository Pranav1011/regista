"""Reading and writing canonical tables on disk. All file I/O for processed data lives here."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from regista.schema import Source, validate_frames


def data_dir() -> Path:
    """Root of local data. Override with REGISTA_DATA_DIR."""
    return Path(os.environ.get("REGISTA_DATA_DIR", "data"))


def raw_dir(source: Source) -> Path:
    return data_dir() / "raw" / source.value


def _processed(source: Source, match_id: str, suffix: str) -> Path:
    return data_dir() / "processed" / source.value / f"{match_id}{suffix}"


def frames_path(source: Source, match_id: str) -> Path:
    return _processed(source, match_id, ".parquet")


def players_path(source: Source, match_id: str) -> Path:
    return _processed(source, match_id, "_players.parquet")


def events_path(source: Source, match_id: str) -> Path:
    return _processed(source, match_id, "_events.parquet")


def info_path(source: Source, match_id: str) -> Path:
    return _processed(source, match_id, "_ingest.json")


def _require(path: Path, source: Source, match_id: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `regista ingest {source.value}` for match {match_id} first"
        )
    return path


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_frames(frames: pd.DataFrame, source: Source, match_id: str) -> Path:
    """Validate and write a frame table. Raises SchemaError on invalid frames."""
    return _write_table(validate_frames(frames), frames_path(source, match_id))


def read_frames(source: Source, match_id: str) -> pd.DataFrame:
    """Read and re-validate a frame table. Raises FileNotFoundError if not ingested yet."""
    path = _require(frames_path(source, match_id), source, match_id)
    return validate_frames(pd.read_parquet(path))


def write_players(players: pd.DataFrame, source: Source, match_id: str) -> Path:
    return _write_table(players, players_path(source, match_id))


def read_players(source: Source, match_id: str) -> pd.DataFrame:
    return pd.read_parquet(_require(players_path(source, match_id), source, match_id))


def write_events(events: pd.DataFrame, source: Source, match_id: str) -> Path:
    return _write_table(events, events_path(source, match_id))


def read_events(source: Source, match_id: str) -> pd.DataFrame:
    return pd.read_parquet(_require(events_path(source, match_id), source, match_id))


def write_info(info: dict, source: Source, match_id: str) -> Path:
    path = info_path(source, match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    return path


def read_info(source: Source, match_id: str) -> dict:
    return json.loads(_require(info_path(source, match_id), source, match_id).read_text())
