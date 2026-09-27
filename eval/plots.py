"""Phase 1 figures (matplotlib + mplsoccer). Tracking data: Metrica Sports sample data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mplsoccer import Pitch  # noqa: E402

from regista.analytics.formations import TEMPLATES, unit_shape  # noqa: E402

CREDIT = "Data: Metrica Sports sample data"
DPI = 110


def _pitch() -> Pitch:
    return Pitch(pitch_type="skillcorner", pitch_length=105, pitch_width=68, line_color="#888",
                 pitch_color="white", linewidth=1)  # fmt: skip


def reliability_diagram(tables: dict[str, pd.DataFrame], title: str, path: Path) -> Path:
    """One panel per model: mean predicted vs observed per bin, with per-bin counts below."""
    fig, axes = plt.subplots(2, len(tables), figsize=(4.2 * len(tables), 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)  # fmt: skip
    for col, (name, t) in enumerate(tables.items()):
        ax, axc = axes[0, col], axes[1, col]
        lo = np.array([float(i.split("-")[0]) for i in t.index])
        ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--")
        ax.plot(t["mean_predicted"], t["observed"], marker="o", color="#1f5f99")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("observed completion rate" if col == 0 else "")
        axc.bar(lo + 0.05, t["count"], width=0.09, color="#9ab")
        for x, c in zip(lo, t["count"], strict=True):
            axc.text(x + 0.05, c, str(int(c)), ha="center", va="bottom", fontsize=7)
        axc.set_yscale("log")
        axc.set_xlabel("predicted probability")
        axc.set_ylabel("passes per bin" if col == 0 else "")
        axc.set_xlim(0, 1)
    fig.suptitle(f"{title}\n{CREDIT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def pass_networks(
    panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]], team: str, title: str, path: Path
) -> Path:
    """Side-by-side pass networks: panel name -> (nodes, edges) for one team."""
    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=len(panels), figsize=(5.2 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, (nodes, edges)) in zip(axes, panels.items(), strict=True):
        n = nodes[nodes["team"] == team].set_index("player_id")
        e = edges[(edges["team"] == team) & (edges["passes"] >= 3)]
        e = e[e["from_player"].isin(n.index) & e["to_player"].isin(n.index)]
        for r in e.itertuples():
            a, b = n.loc[r.from_player], n.loc[r.to_player]
            pitch.lines(a.x, a.y, b.x, b.y, lw=0.4 + r.passes * 0.35, color="#1f5f99",
                        alpha=0.55, ax=ax, zorder=1)  # fmt: skip
        made = e.groupby("from_player")["passes"].sum().reindex(n.index).fillna(0)
        pitch.scatter(n.x, n.y, s=40 + made * 6, color="#c8553d", edgecolors="black",
                      linewidth=0.5, ax=ax, zorder=2)  # fmt: skip
        for pid, r in n.iterrows():
            ax.text(r.x, r.y + 2.6, str(pid), fontsize=6, ha="center")
        ax.set_title(f"{name} ({int(e['passes'].sum())} passes on edges >= 3)", fontsize=9)
    fig.suptitle(f"{title} - attacking left to right\n{CREDIT}", fontsize=10)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def formation_snapshots(windows: list[dict], path: Path) -> Path:
    """Per window: players' unit-shape positions, matched template slots, label and margin."""
    fig, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 4.4), squeeze=False)
    for ax, w in zip(axes[0], windows, strict=True):
        shape = unit_shape(w["xy"])
        template = unit_shape(np.array([[s.x, s.y] for s in TEMPLATES[w["label"]]]))
        ax.scatter(template[:, 0], template[:, 1], marker="s", s=90, facecolors="none",
                   edgecolors="#999", label=f"{w['label']} slots")  # fmt: skip
        ax.scatter(shape[:, 0], shape[:, 1], s=45, color="#c8553d", edgecolors="black",
                   linewidth=0.5, label="players (mean, unit shape)")  # fmt: skip
        for (x, y), role in zip(shape, w["roles"], strict=True):
            ax.text(x, y + 0.18, role, fontsize=7, ha="center")
        ax.set_title(
            f"{w['name']}\n{w['label']}  (runner-up {w['runner_up']}, margin {w['margin']:.3f})",
            fontsize=9,
        )
        ax.set_xlabel("depth (towards opponent goal)")
        ax.set_ylabel("width (+ = left)")
        ax.set_aspect("equal")
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.3, 2.3)
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Formation snapshots, 5-minute windows\n{CREDIT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def passing_options_snapshot(
    surface: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    scene,
    options: pd.DataFrame,
    title: str,
    path: Path,
) -> Path:
    """Pitch-control surface for one moment with each teammate's option value."""
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=(7, 4.6))
    mesh = ax.pcolormesh(xgrid, ygrid, surface, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75,
                         shading="auto", zorder=0)  # fmt: skip
    pitch.scatter(scene.att_pos[:, 0], scene.att_pos[:, 1], s=45, color="#b2182b",
                  edgecolors="black", ax=ax, zorder=2)  # fmt: skip
    pitch.scatter(scene.def_pos[:, 0], scene.def_pos[:, 1], s=45, color="#2166ac",
                  edgecolors="black", ax=ax, zorder=2)  # fmt: skip
    pitch.scatter(scene.ball[0], scene.ball[1], s=20, color="black", ax=ax, zorder=3)
    for r in options.itertuples():
        ax.text(r.x, r.y + 2.2, f"{r.pitch_control:.2f}", fontsize=7, ha="center", zorder=4)
    fig.colorbar(mesh, ax=ax, fraction=0.03,
                 label="attacking pitch control (raw model output); red = attackers")  # fmt: skip
    ax.set_title(f"{title}\n{CREDIT}", fontsize=9)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
