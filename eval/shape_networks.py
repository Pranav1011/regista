"""Team shape summaries and pass-network similarity on the Metrica games.

Pass networks are built twice per team, from passes inferred from tracking
(pass detector v1 and v2) and from Metrica's labelled PASS events, on the same
nodes (mean positions in possession). Game 3 is the held-out game; games 1-2
were used to tune pass detection, so their network scores are optimistic.

Run: uv run python eval/shape_networks.py
"""

from __future__ import annotations

import pandas as pd
from _common import TEST_GAME, TRAIN_GAMES, load_params, metrica_game, possession_phase, true_passes
from pass_detection import predict

from regista.analytics.pass_network import (
    EDGE_KEYS,
    compare_networks,
    node_positions,
    pass_edges,
)
from regista.analytics.shape import SHAPE_METRICS, shape_windows, team_shape

GAMES = (*TRAIN_GAMES, TEST_GAME)
VERSIONS = ("v1", "v2")


def evaluate() -> dict:
    params = load_params()
    shapes, comparisons, networks = [], [], {}
    for game in GAMES:
        frames, events = metrica_game(game)
        possession = possession_phase(frames, params)
        shapes.append(shape_windows(team_shape(frames), possession))

        nodes = node_positions(frames, possession)
        truth = true_passes(events).assign(match_id=str(game))
        event_edges = pass_edges(truth)
        networks[(game, "events")] = (nodes, event_edges)
        for version in VERSIONS:
            _, detected, _ = predict(frames, version, params)
            edges = pass_edges(detected[detected["kind"] == "pass"])
            networks[(game, version)] = (nodes, edges)
            comparisons.append(
                compare_networks(edges, event_edges).assign(
                    game=game, version=version, held_out=game == TEST_GAME
                )
            )
    shape = pd.concat(shapes, ignore_index=True)
    return {
        "shape_by_phase": shape.groupby(["match_id", "team", "phase"])[SHAPE_METRICS].median(),
        "shape_windows": shape,
        "network_similarity": pd.concat(comparisons, ignore_index=True),
        "networks": networks,
    }


def edge_residuals(networks: dict, game: int, version: str, team: str) -> pd.DataFrame:
    """Per directed edge: inferred minus labelled pass count, largest absolute first."""
    inferred = networks[(game, version)][1]
    events = networks[(game, "events")][1]
    e = inferred.merge(events, on=EDGE_KEYS, how="outer", suffixes=("_inferred", "_events"))
    e = e[e["team"] == team].fillna({"passes_inferred": 0, "passes_events": 0})
    e["residual"] = e["passes_inferred"] - e["passes_events"]
    return e.reindex(e["residual"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def player_residuals(edges: pd.DataFrame) -> pd.DataFrame:
    """Per player: inferred minus labelled passes made (out) and received (in)."""
    out = edges.groupby("from_player")[["passes_inferred", "passes_events"]].sum()
    inn = edges.groupby("to_player")[["passes_inferred", "passes_events"]].sum()
    table = pd.DataFrame(
        {
            "made_events": out["passes_events"],
            "made_residual": out["passes_inferred"] - out["passes_events"],
            "received_events": inn["passes_events"],
            "received_residual": inn["passes_inferred"] - inn["passes_events"],
        }
    ).fillna(0)
    return table.sort_values("made_events", ascending=False)


def main() -> None:
    r = evaluate()
    print("## Team shape (median of 5-minute window medians)")
    print(r["shape_by_phase"].round(1).to_markdown())
    print("\n## Pass networks: inferred (a) vs labelled events (b)")
    cols = ["game", "held_out", "version", "team", "edges_a", "edges_b", "passes_a", "passes_b",
            "pearson", "spearman", "weighted_jaccard"]  # fmt: skip
    print(r["network_similarity"][cols].round(3).to_markdown(index=False))
    for version in VERSIONS:
        for team in ("home", "away"):
            edges = edge_residuals(r["networks"], TEST_GAME, version, team)
            print(f"\n### Game {TEST_GAME} {team}, {version}: largest edge residuals")
            print(f"sum |residual| = {edges['residual'].abs().sum():.0f} over "
                  f"{int(edges['passes_events'].sum())} labelled passes")  # fmt: skip
            print(edges.head(8)[["from_player", "to_player", "passes_inferred", "passes_events",
                                 "residual"]].to_markdown(index=False))  # fmt: skip
            print(player_residuals(edges).head(11).to_markdown())


if __name__ == "__main__":
    main()
