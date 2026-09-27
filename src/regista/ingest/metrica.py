"""Metrica Sports sample data (games 1-3), via kloppy's open-data loader.

Data: https://github.com/metrica-sports/sample-data. Acknowledge Metrica Sports
in anything public. Coordinates are normalised by the provider, so they are
mapped onto the canonical 105 x 68 m pitch.
"""

from __future__ import annotations

from kloppy import metrica

from regista.ingest._kloppy import IngestResult, dataset_to_canonical
from regista.schema import Source

GAMES = (1, 2, 3)


def match_id(game: int) -> str:
    return str(game)


def load(game: int) -> IngestResult:
    """Download one sample game and convert it to canonical tables."""
    if game not in GAMES:
        raise ValueError(f"Metrica sample game must be one of {GAMES}, got {game}")
    dataset = metrica.load_open_data(match_id=game)
    return dataset_to_canonical(dataset, match_id(game), Source.METRICA)
