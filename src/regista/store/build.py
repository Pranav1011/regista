"""Build a match store: precompute the frozen pipeline's outputs for one match into parquet.

Layout: ``<root>/<source>/<match_id>/<table>.parquet`` plus ``manifest.json``.
Tables:
- ``frames``: canonical frames with velocities
- ``possession``: per frame, ball owner (v2 radius) and team in possession
- ``passes``: v2 passes and turnovers, with ``pass_id``
- ``pass_moments``: pass attempts (v2 passes + released turnovers) with raw pitch
  control at the target and the calibrated display probability
- ``pass_options``: pitch control at every onside teammate at each pass moment
- ``formations`` / ``roles``: formation windows with runner-up and margins
- ``shape_windows``: team shape medians per window and phase
- ``network_nodes`` / ``network_edges``: full-match pass network per team
- ``pressure``: per carrier frame, nearest defender and defenders within 5 yd
- ``press_windows``: press intensity per pressing team, window, and pitch third
- ``stream_windows``: causal per-minute trailing-window features (formation state,
  press intensity, line height) - what a live viewer may show at each time
- ``moments``: alerts from the frozen causal detectors (evidence as JSON)
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regista import __version__
from regista.analytics.calibration import from_dict
from regista.analytics.formations import detect_formations, window_shapes
from regista.analytics.kinematics import frame_interval
from regista.analytics.pass_network import node_positions, pass_edges
from regista.analytics.passing_options import build_scene, default_params, passing_options
from regista.analytics.possession import team_in_possession
from regista.analytics.pressing import press_windows, pressure_frames
from regista.analytics.shape import goalkeepers, shape_windows, team_shape
from regista.moments import DetectorConfig, StreamConfig, detect_moments, stream_windows
from regista.pipeline import (
    detect,
    pass_attempts,
    scaled_frames,
    score_attempts,
    v2_owner,
)
from regista.schema import validate_frames

WINDOW_S = 300.0
STORE_FORMAT = 1


def config_hash(params: dict, calibration: dict, moments_config: dict) -> str:
    """Stable hash of the parameters, calibrator, and detector config a store was built with."""
    blob = json.dumps(
        {"params": params, "calibration": calibration, "moments": moments_config}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _pass_options(frames: pd.DataFrame, moments: pd.DataFrame, display) -> pd.DataFrame:
    p = default_params()
    gk = goalkeepers(frames).set_index(["period", "team"])["gk_id"]
    indexed = frames.set_index(["period", "frame"]).sort_index()
    rows = []
    for m in moments.itertuples():
        scene = build_scene(indexed.loc[(m.period, m.frame)].reset_index(), m.period, m.frame,
                            m.team, {t: gk.get((m.period, t)) for t in ("home", "away")},
                            ball_xy=np.array([m.start_x, m.start_y]), p=p)  # fmt: skip
        options = passing_options(scene, p)
        if options.empty:
            continue
        sign = -1.0 if m.team == "away" else 1.0  # back to canonical coordinates
        rows.append(
            options.assign(
                moment_id=m.moment_id,
                x=options["x"] * sign,
                y=options["y"] * sign,
                display_probability=display.predict(options["pitch_control"].to_numpy()),
                is_target=options["player_id"] == m.to_player,
            )
        )
    if not rows:
        return pd.DataFrame(columns=["moment_id", "player_id", "x", "y", "pitch_control",
                                     "display_probability", "is_target"])  # fmt: skip
    out = pd.concat(rows, ignore_index=True)
    return out[["moment_id", "player_id", "x", "y", "pitch_control", "display_probability",
                "is_target"]]  # fmt: skip


def compute_tables(
    frames: pd.DataFrame,
    params: dict,
    calibration: dict,
    moments_config: dict,
    window_s: float = WINDOW_S,
) -> dict[str, pd.DataFrame]:
    """All store tables for one match, computed directly (no I/O)."""
    frames = validate_frames(frames)
    owner, radius = v2_owner(frames, params)
    phase = team_in_possession(owner, scaled_frames(params["v2"]["max_gap_frames"], frames))
    possession = owner.merge(
        phase.rename(columns={"team": "possession_team"}), on=["match_id", "period", "frame"],
        how="left",
    )  # fmt: skip

    _, detected, _ = detect(frames, "v2", params)
    times = frames.drop_duplicates(["period", "frame"]).set_index(["period", "frame"])["t"]
    passes = detected.reset_index(drop=True).rename_axis("pass_id").reset_index()
    passes["t_start"] = times.reindex(
        pd.MultiIndex.from_arrays([passes["period"], passes["start_frame"]])
    ).to_numpy()

    display = from_dict(calibration[calibration["display"]])
    moments = score_attempts(frames, pass_attempts(frames, params))
    moments = moments.rename_axis("moment_id").reset_index()
    moments["display_probability"] = display.predict(moments["pitch_control"].to_numpy())

    shapes = window_shapes(frames, phase, window_s, window_s)
    formations, roles = detect_formations(shapes)
    nodes = node_positions(frames, phase)
    pressure = pressure_frames(frames, owner)
    stream = stream_windows(frames, params, StreamConfig(**moments_config["stream"]))
    alerts = detect_moments(stream, DetectorConfig(**moments_config["detectors"]))
    alerts = alerts.assign(evidence=alerts["evidence"].map(json.dumps))
    edges = pass_edges(detected[detected["kind"] == "pass"])
    return {
        "frames": frames,
        "possession": possession,
        "passes": passes,
        "pass_moments": moments,
        "pass_options": _pass_options(frames, moments, display),
        "formations": formations,
        "roles": roles,
        "shape_windows": shape_windows(team_shape(frames), phase, window_s),
        "network_nodes": nodes,
        "network_edges": edges,
        "pressure": pressure,
        "press_windows": press_windows(pressure, window_s),
        "stream_windows": stream,
        "moments": alerts,
        "_radius": pd.DataFrame({"radius_m": [radius]}),
    }


def build_store(
    frames: pd.DataFrame,
    params: dict,
    calibration: dict,
    moments_config: dict,
    source: str,
    match_id: str,
    root: Path,
    window_s: float = WINDOW_S,
) -> Path:
    """Compute and write one match's store; returns its directory."""
    tables = compute_tables(frames, params, calibration, moments_config, window_s)
    radius = float(tables.pop("_radius")["radius_m"].iat[0])
    out = Path(root) / source / match_id
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    manifest = {
        "store_format": STORE_FORMAT,
        "regista_version": __version__,
        "config_hash": config_hash(params, calibration, moments_config),
        "source": source,
        "match_id": match_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_s": window_s,
        "ownership_radius_m": radius,
        "frame_rate": round(1.0 / frame_interval(frames), 3),
        "periods": {
            str(int(p)): {
                "frames": [int(g["frame"].min()), int(g["frame"].max())],
                "t_max": float(g["t"].max()),
            }
            for p, g in frames.groupby("period")
        },  # fmt: skip
        "tables": {name: len(df) for name, df in tables.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out
