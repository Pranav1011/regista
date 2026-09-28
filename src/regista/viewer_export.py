"""Export one match store to the static viewer's data format (``viewer/public/data/<id>/``).

Files:
- ``manifest.json``: match, split label (tuning match / held out), periods, object
  order, frame rate, credits
- ``frames.i16z``: gzip-compressed positions at 10 fps, int16 decimetres, x then y per object,
  ``-32768`` when an object is not tracked; one block of objects per frame
- ``cards.json``: causal formation cards per minute (the trailing-window state a
  live viewer may show at that time, never a window with future frames)
- ``alerts.json``: detector moments, shown only once their emit time is reached
- ``pass_moments.json``: pass attempts with passing options and the Platt display
  probability ("model estimate")
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regista import io
from regista.clock import match_clock
from regista.schema import Source
from regista.store import Store

VIEWER_FPS = 10.0
ANSWERS_DIR = Path(__file__).resolve().parents[2] / "eval" / "agent" / "answers"
ABSENT = -32768
CREDITS = {
    "metrica": "Tracking data: Metrica Sports sample data",
    "skillcorner": "Tracking data: SkillCorner open data (MIT)",
    "method": "Pitch control adapted from LaurieOnTracking (MIT)",
}


def _objects(frames: pd.DataFrame, jerseys: dict[str, int]) -> list[dict]:
    """Ball first, then players; each player labelled with their shirt number."""
    players = frames[frames["team"] != "ball"].drop_duplicates("player_id")[["player_id", "team"]]
    players = players.sort_values(["team", "player_id"], ascending=[False, True])
    objs = [{"id": "ball", "team": "ball", "label": ""}]
    for r in players.itertuples():
        if r.player_id not in jerseys:
            raise ValueError(f"no shirt number for {r.player_id}")
        objs.append({"id": r.player_id, "team": r.team, "label": str(jerseys[r.player_id])})
    return objs


def _positions(frames: pd.DataFrame, objs: list[dict]) -> tuple[bytes, list[dict]]:
    """Downsample to 10 fps per period; pack int16 decimetres."""
    order = {o["id"]: i for i, o in enumerate(objs)}
    blocks, periods = [], []
    offset = 0
    for period, pf in frames.groupby("period", sort=True):
        times = np.sort(pf["t"].unique())
        grid = np.arange(times[0], times[-1] + 1e-9, 1.0 / VIEWER_FPS)
        idx = np.searchsorted(times, grid)
        idx = np.clip(idx, 0, len(times) - 1)
        chosen = times[idx]
        sel = pf[pf["t"].isin(set(chosen))]
        arr = np.full((len(chosen), len(objs), 2), ABSENT, dtype=np.int16)
        row_of = {t: i for i, t in enumerate(chosen)}
        rows = sel["t"].map(row_of).to_numpy()
        cols = sel["player_id"].map(order).to_numpy()
        arr[rows, cols, 0] = np.round(sel["x"].to_numpy() * 10).astype(np.int16)
        arr[rows, cols, 1] = np.round(sel["y"].to_numpy() * 10).astype(np.int16)
        blocks.append(arr)
        periods.append(
            {
                "period": int(period),
                "first_index": offset,
                "n_frames": len(chosen),
                "t_start": float(chosen[0]),
                "t_end": float(chosen[-1]),
                "clock_start": match_clock(int(period), float(chosen[0])),
                "clock_end": match_clock(int(period), float(chosen[-1])),
            }
        )
        offset += len(chosen)
    return np.concatenate(blocks).tobytes(), periods


def export_match(
    store_path: Path, out_root: Path, split_label: str, min_coverage_min: float
) -> Path:
    store = Store(store_path)
    frames = store.table("frames")
    players = io.read_players(Source(store.source), store.match_id)
    objs = _objects(
        frames, dict(zip(players["player_id"], players["jersey_no"].astype(int), strict=True))
    )
    blob, periods = _positions(frames, objs)
    out = Path(out_root) / f"{store.source}-{store.match_id}"
    out.mkdir(parents=True, exist_ok=True)
    # gzip body with a neutral extension, so no server adds Content-Encoding and the
    # viewer always decompresses it itself
    (out / "frames.i16z").write_bytes(gzip.compress(blob, compresslevel=9))

    stream = store.table("stream_windows")
    median_margin = float(stream["margin_out"].median())
    cards = []
    for r in stream.itertuples():
        card = {
            "period": int(r.period),
            "t": float(r.t_end),
            "team": r.team,
            "usable": bool(r.coverage_s >= min_coverage_min * 60 - 1e-6),
        }
        for ph in ("out", "in"):
            label = getattr(r, f"label_{ph}")
            margin = getattr(r, f"margin_{ph}")
            card[ph] = (
                None
                if not isinstance(label, str)
                else {
                    "label": label,
                    "runner_up": getattr(r, f"runner_up_{ph}"),
                    "margin": round(float(margin), 4),
                    "close_call": bool(margin < median_margin),
                }
            )
        cards.append(card)
    alerts = []
    for r in store.table("moments").itertuples():
        ev = json.loads(r.evidence)
        alerts.append(
            {
                "type": r.type,
                "team": r.team,
                "period": int(r.period),
                "emit_t": float(r.emit_t),
                "start_t": float(r.start_t),
                "emit_clock": match_clock(int(r.period), float(r.emit_t)),
                "start_clock": match_clock(int(r.period), float(r.start_t)),
                "severity": float(r.severity),
                "before": ev["before"],
                "after": ev["after"],
            }
        )
    moments = store.table("pass_moments")
    options = store.table("pass_options")
    pm = []
    for m in moments.itertuples():
        opts = options[options["moment_id"] == m.moment_id]
        pm.append(
            {
                "id": int(m.moment_id),
                "period": int(m.period),
                "t": float(m.t),
                "team": m.team,
                "from": m.from_player,
                "to": m.to_player,
                "completed": bool(m.completed),
                "options": [
                    {
                        "player": o.player_id,
                        "p": round(float(o.display_probability), 3),
                        "target": bool(o.is_target),
                    }
                    for o in opts.itertuples()
                ],
            }
        )
    manifest = {
        "id": out.name,
        "source": store.source,
        "match_id": store.match_id,
        "split": split_label,
        "fps": VIEWER_FPS,
        "objects": objs,
        "periods": periods,
        "absent": ABSENT,
        "units": "decimetres, centre origin, home attacks +x",
        "median_margin": median_margin,
        "credits": CREDITS,
        "notes": [
            "Pass probabilities are model estimates (pitch control, Platt-calibrated).",
            "Formation cards and alerts use only data up to the playback time.",
        ],
    }
    for name, data in (
        ("manifest", manifest),
        ("cards", cards),
        ("alerts", alerts),
        ("pass_moments", pm),
    ):
        (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    answers = ANSWERS_DIR / f"{out.name}.json"
    if answers.exists():  # pre-generated by eval/agent/pregenerate.py with the frozen model
        (out / "answers.json").write_text(answers.read_text())
    return out
