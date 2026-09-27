"""Known-answer tests for pass networks. All data here is synthetic, by design."""

import numpy as np
import pandas as pd
import pytest

from regista.analytics.pass_network import compare_networks, node_positions, pass_edges


def _passes(pairs: list[tuple[str, str]], team: str = "home") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": "synthetic",
            "team": team,
            "from_player": [a for a, _ in pairs],
            "to_player": [b for _, b in pairs],
        }
    )


def test_edges_count_directed_passes():
    edges = pass_edges(_passes([("a", "b"), ("a", "b"), ("b", "a"), ("b", "c")]))
    got = {(r.from_player, r.to_player): r.passes for r in edges.itertuples()}
    assert got == {("a", "b"): 2, ("b", "a"): 1, ("b", "c"): 1}


def test_edges_skip_passes_without_receiver():
    passes = _passes([("a", "b")])
    passes.loc[1] = ["synthetic", "home", "a", None]
    assert pass_edges(passes)["passes"].sum() == 1


def test_nodes_use_only_own_possession_and_attacking_frame():
    rows = []
    for f in range(4):
        rows.append({"frame": f, "team": "home", "player_id": "h1", "x": -10.0 + f, "y": 5.0})
        rows.append({"frame": f, "team": "away", "player_id": "a1", "x": 10.0, "y": 5.0})
    frames = pd.DataFrame(rows)
    frames["match_id"], frames["period"] = "synthetic", 1
    possession = frames[["match_id", "period", "frame"]].drop_duplicates()
    possession["team"] = np.where(possession["frame"] < 2, "home", "away")
    nodes = node_positions(frames, possession).set_index("player_id")
    assert nodes.loc["h1", "x"] == pytest.approx(-9.5)  # frames 0-1 only
    assert nodes.loc["h1", "n_frames"] == 2
    assert (nodes.loc["a1", "x"], nodes.loc["a1", "y"]) == (-10.0, -5.0)  # mirrored


def test_identical_networks_score_one_and_disjoint_score_zero():
    a = pass_edges(_passes([("a", "b")] * 3 + [("b", "c")] * 2 + [("c", "a")]))
    same = compare_networks(a, a).iloc[0]
    assert same["pearson"] == pytest.approx(1.0)
    assert same["weighted_jaccard"] == pytest.approx(1.0)
    other = pass_edges(_passes([("d", "e")] * 2 + [("e", "f")] + [("f", "d")] * 4))
    disjoint = compare_networks(a, other).iloc[0]
    assert disjoint["weighted_jaccard"] == 0.0
    assert disjoint["pearson"] < 0
    assert (disjoint["passes_a"], disjoint["passes_b"]) == (6, 7)
