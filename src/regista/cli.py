"""Regista command-line interface."""

from __future__ import annotations

import typer

from regista import io
from regista.ingest._kloppy import IngestResult
from regista.schema import Source

app = typer.Typer(help="Regista: football match intelligence.", no_args_is_help=True)
ingest_app = typer.Typer(help="Convert open data into canonical frames.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@app.callback()
def main() -> None:
    """Regista: football match intelligence."""


def _write(source: Source, match_id: str, result: IngestResult) -> None:
    path = io.write_frames(result.frames, source, match_id)
    io.write_players(result.players, source, match_id)
    io.write_info(result.info, source, match_id)
    dropped = result.info["rows_dropped_out_of_bounds"]
    typer.echo(
        f"wrote {len(result.frames):,} rows to {path} ({dropped} out-of-bounds rows dropped)"
    )


@ingest_app.command("metrica")
def ingest_metrica(game: int = typer.Option(..., help="Sample game number (1-3).")) -> None:
    """Ingest a Metrica Sports sample game."""
    from regista.ingest import metrica, metrica_events

    match_id = metrica.match_id(game)
    result = metrica.load(game)
    _write(Source.METRICA, match_id, result)
    events = metrica_events.load(game, result.players, result.info["flipped_periods"])
    path = io.write_events(events, Source.METRICA, match_id)
    typer.echo(f"wrote {len(events):,} events to {path}")


@ingest_app.command("skillcorner")
def ingest_skillcorner(
    match: str = typer.Option(..., help="SkillCorner open-data match id."),
) -> None:
    """Ingest a SkillCorner open-data match."""
    from regista.ingest import skillcorner

    _write(Source.SKILLCORNER, match, skillcorner.load(match))


if __name__ == "__main__":
    app()
