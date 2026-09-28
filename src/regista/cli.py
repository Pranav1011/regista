"""Regista command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from regista import io
from regista.ingest._kloppy import IngestResult
from regista.schema import Source

DEFAULT_PARAMS = Path("eval/params_phase1.json")
DEFAULT_CALIBRATION = Path("eval/calibration_phase1.json")
DEFAULT_MOMENTS = Path("eval/params_phase2.json")

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


@app.command("build-store")
def build_store_cmd(
    source: str = typer.Option(..., help="metrica or skillcorner."),
    game: int = typer.Option(None, help="Metrica sample game number."),
    match: str = typer.Option(None, help="SkillCorner match id."),
    params: Path = typer.Option(DEFAULT_PARAMS, help="Frozen pipeline parameters."),
    calibration: Path = typer.Option(DEFAULT_CALIBRATION, help="Calibrator."),
    moments: Path = typer.Option(DEFAULT_MOMENTS, help="Frozen detector configuration."),
) -> None:
    """Precompute a match store under data/store/<source>/<match_id>/."""
    import json

    from regista.analytics.kinematics import add_velocities
    from regista.store import build_store

    src = Source(source)
    if src == Source.METRICA:
        if game is None:
            raise typer.BadParameter("--game is required for metrica")
        match_id = str(game)
    elif src == Source.SKILLCORNER:
        if match is None:
            raise typer.BadParameter("--match is required for skillcorner")
        match_id = match
    else:
        raise typer.BadParameter(f"unsupported source {source!r}")
    frames = add_velocities(io.read_frames(src, match_id))
    path = build_store(
        frames,
        json.loads(params.read_text()),
        json.loads(calibration.read_text()),
        json.loads(moments.read_text()),
        src.value,
        match_id,
        io.data_dir() / "store",
    )
    typer.echo(f"wrote store to {path}")


@app.command("mcp")
def mcp_cmd(store_root: Path = typer.Option(None, help="Store root (default data/store).")) -> None:
    """Run the Regista MCP server over stdio."""
    from regista.agent.mcp_server import build_server

    build_server(store_root).run("stdio")


@app.command("export-viewer")
def export_viewer_cmd(
    game: list[int] = typer.Option([1, 2, 3], help="Metrica sample games to export."),
    out: Path = typer.Option(Path("viewer/public/data"), help="Viewer data directory."),
) -> None:
    """Export Metrica match stores for the static viewer (games 1-2 are tuning matches)."""
    from regista.viewer_export import export_match

    tuning = {1, 2}
    for g in game:
        label = (
            "tuning match (Metrica game used to design thresholds)" if g in tuning else "held out"
        )
        path = export_match(io.data_dir() / "store" / "metrica" / str(g), out, label)
        size = sum(f.stat().st_size for f in path.iterdir())
        typer.echo(f"exported {path} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
