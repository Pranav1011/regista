"""Regista command-line interface."""

from __future__ import annotations

import typer

app = typer.Typer(help="Regista: football match intelligence.", no_args_is_help=True)
ingest_app = typer.Typer(help="Convert open data into canonical frames.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@app.callback()
def main() -> None:
    """Regista: football match intelligence."""


if __name__ == "__main__":
    app()
