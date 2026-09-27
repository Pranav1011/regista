"""Regista command-line interface."""

from __future__ import annotations

import typer

from regista.ingest._kloppy import IngestResult

app = typer.Typer(help="Regista: football match intelligence.", no_args_is_help=True)
ingest_app = typer.Typer(help="Convert open data into canonical frames.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@app.callback()
def main() -> None:
    """Regista: football match intelligence."""


def _report(result: IngestResult) -> None:
    info = result.info
    d = info["rows_dropped_out_of_bounds"]
    typer.echo(
        f"{info['source']} {info['match_id']}: {info['rows']:,} rows "
        f"(out of bounds, dropped: {d['ball']} ball, {d['player']} player rows)"
    )
    if "events" in info:
        typer.echo(f"{info['events']:,} events")


@ingest_app.command("metrica")
def ingest_metrica(game: int = typer.Option(..., help="Sample game number (1-3).")) -> None:
    """Ingest a Metrica Sports sample game (tracking and events)."""
    from regista.ingest import metrica

    _report(metrica.ingest(game))


@ingest_app.command("skillcorner")
def ingest_skillcorner(
    match: str = typer.Option(..., help="SkillCorner open-data match id."),
) -> None:
    """Ingest a SkillCorner open-data match."""
    from regista.ingest import skillcorner

    _report(skillcorner.ingest(match))


if __name__ == "__main__":
    app()
