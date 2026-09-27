"""Pass networks: nodes at players' mean positions in possession, edges weighted by passes.

Positions use each team's attacking frame (+x towards the opponent goal), so
networks of both teams read the same way. Networks built from different pass
sources (inferred from tracking, or labelled events) share the same nodes and
can be compared edge by edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from regista.analytics.formations import to_attacking_frame
from regista.schema import Team

EDGE_KEYS = ["match_id", "team", "from_player", "to_player"]


def node_positions(frames: pd.DataFrame, possession: pd.DataFrame) -> pd.DataFrame:
    """Mean position of each player over frames in which their team has the ball."""
    keys = ["match_id", "period", "frame"]
    players = to_attacking_frame(frames[frames["team"] != Team.BALL.value])
    players = players.merge(possession.rename(columns={"team": "possession_team"}), on=keys)
    players = players[players["possession_team"] == players["team"]]
    return (
        players.groupby(["match_id", "team", "player_id"])
        .agg(x=("x", "mean"), y=("y", "mean"), n_frames=("frame", "size"))
        .reset_index()
    )


def pass_edges(passes: pd.DataFrame) -> pd.DataFrame:
    """Directed edge weights: number of passes from each player to each teammate."""
    completed = passes.dropna(subset=["from_player", "to_player"])
    return completed.groupby(EDGE_KEYS).size().rename("passes").reset_index()


def compare_networks(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per match and team, how similar two edge tables are.

    Edges are compared over the union of directed player pairs, with absent
    edges counted as 0 passes. ``weighted_jaccard`` = sum(min) / sum(max).
    """
    union = a.merge(b, on=EDGE_KEYS, how="outer", suffixes=("_a", "_b")).fillna(
        {"passes_a": 0, "passes_b": 0}
    )
    rows = []
    for (match_id, team), g in union.groupby(["match_id", "team"]):
        wa, wb = g["passes_a"].to_numpy(float), g["passes_b"].to_numpy(float)
        varied = len(g) > 2 and wa.std() > 0 and wb.std() > 0
        rows.append(
            {
                "match_id": match_id,
                "team": team,
                "edges_a": int((wa > 0).sum()),
                "edges_b": int((wb > 0).sum()),
                "passes_a": int(wa.sum()),
                "passes_b": int(wb.sum()),
                "pearson": pearsonr(wa, wb).statistic if varied else np.nan,
                "spearman": spearmanr(wa, wb).statistic if varied else np.nan,
                "weighted_jaccard": np.minimum(wa, wb).sum() / np.maximum(wa, wb).sum(),
            }
        )
    return pd.DataFrame(rows)
