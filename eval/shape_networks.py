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

from regista.analytics.pass_network import compare_networks, node_positions, pass_edges
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


def main() -> None:
    r = evaluate()
    print("## Team shape (median of 5-minute window medians)")
    print(r["shape_by_phase"].round(1).to_markdown())
    print("\n## Pass networks: inferred (a) vs labelled events (b)")
    cols = ["game", "held_out", "version", "team", "edges_a", "edges_b", "passes_a", "passes_b",
            "pearson", "spearman", "weighted_jaccard"]  # fmt: skip
    print(r["network_similarity"][cols].round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
