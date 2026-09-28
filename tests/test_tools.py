"""Tool-layer and MCP server tests on a synthetic match store. All data is synthetic, by design."""

import asyncio
import json
from pathlib import Path

import pytest
from synthetic_match import MatchSpec, make_match

from regista.agent.mcp_server import build_server
from regista.agent.tools import TOOL_NAMES, Toolbox, ToolError
from regista.store import build_store

REPO = Path(__file__).resolve().parent.parent
CONFIGS = [
    json.loads((REPO / "eval" / name).read_text())
    for name in ("params_phase1.json", "calibration_phase1.json", "params_phase2.json")
]
MATCH = "synthetic/t1"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("stores")
    frames = make_match(MatchSpec(period_s=600.0, fps=5.0))
    build_store(frames, *CONFIGS, "synthetic", "t1", root)
    return root


@pytest.fixture(scope="module")
def tools(root):
    return Toolbox(root)


def test_formation_known_answer_with_evidence_and_reliability(tools):
    r = tools.get_formation(MATCH, "home", "out")
    assert r.most_common_label == "4-4-2"
    assert r.back_line_counts == {"4": len(r.windows)}
    assert all(w.margin is not None and w.runner_up for w in r.windows)
    assert r.evidence and all(e.match == MATCH and e.frame_end >= e.frame_start for e in r.evidence)
    assert "not probabilities" in r.notes[0]


def test_time_ranges_are_validated(tools):
    # period 1 lasts 10 minutes, period 2 runs from 45:00 to 55:00
    first = tools.get_formation(MATCH, "home", "out", "00:00", "05:00")
    whole_first_half = tools.get_team_dimensions(MATCH, "home", "out", "00:00", "45:00")
    assert {e.period for e in whole_first_half.evidence} == {1}
    assert {w.period for w in first.windows} == {1}
    both = tools.get_team_dimensions(MATCH, "home", "out", "02:00", "50:00")
    assert {e.period for e in both.evidence} == {1, 2}
    with pytest.raises(ToolError, match="outside the recorded match"):
        tools.get_formation(MATCH, "home", "out", "30:00", "40:00")
    with pytest.raises(ToolError, match="malformed"):
        tools.get_team_dimensions(MATCH, "home", "out", "ten past", None)
    with pytest.raises(ToolError, match="empty time range"):
        tools.get_press_stats(MATCH, "home", "05:00", "02:00")


def test_unsupported_topics_are_explicitly_unavailable(tools):
    assert not tools.check_capability("What was the xG?").available
    assert not tools.check_capability("player names").available
    assert tools.check_capability("pressing after 60 minutes").available


def test_passes_options_and_players(tools):
    player = tools.get_pass_network(MATCH, "home", top_k=3).top_edges[0].from_player
    pp = tools.get_player_passes(MATCH, player)
    assert pp.attempted >= pp.completed > 0
    options = tools.get_passing_options(MATCH, pp.passes[0].pass_id)
    assert options.options and all(0 <= o.display_probability <= 1 for o in options.options)
    assert "model estimate" in options.notes[0]
    with pytest.raises(ToolError, match="unknown pass_id"):
        tools.get_passing_options(MATCH, 10**6)
    with pytest.raises(ToolError, match="no passes"):
        tools.get_player_passes(MATCH, "nobody")


def test_press_stats_cover_every_third_and_clean_match_has_no_moments(tools):
    r = tools.get_press_stats(MATCH, "away")
    assert [t.third for t in r.by_third] == ["all", "defensive", "middle", "attacking"]
    assert tools.find_moments(MATCH).moments == []
    with pytest.raises(ToolError, match="unknown match"):
        tools.get_match_overview("metrica/99")


def test_mcp_server_lists_tools_and_calls_one(root):
    from mcp import Client

    async def run():
        async with Client(build_server(root)) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            ok = await client.call_tool("get_match_overview", {"match": MATCH})
            bad = await client.call_tool(
                "get_formation",
                {"match": MATCH, "team": "home", "phase": "out", "from_clock": "30:00"},
            )
            return names, ok, bad

    names, ok, bad = asyncio.run(run())
    assert set(TOOL_NAMES) | {"list_matches"} <= names
    assert not ok.is_error and ok.structured_content["match"] == MATCH
    assert bad.is_error and "outside the recorded match" in bad.content[0].text


def test_team_passing_ranks_players_and_moment_types_accept_plain_words(tools):
    r = tools.get_team_passing(MATCH, "home")
    assert r.players and r.players[0].attempted >= r.players[-1].attempted
    assert all(0 <= p.completion_share <= 1 for p in r.players)
    from regista.agent.tools import moment_type

    assert moment_type("pressing change") == "press_change"
    assert moment_type("back line") == "back_line_change"
    assert moment_type("line_height_shift") == "line_height_shift"
    with pytest.raises(ToolError, match="unknown moment type"):
        tools.find_moments(MATCH, type="goals")


def test_period_argument_selects_a_whole_half(tools):
    first = tools.get_team_dimensions(MATCH, "home", "out", period=1)
    assert {e.period for e in first.evidence} == {1}
    assert {e.period for e in tools.get_press_stats(MATCH, "away", period=2).evidence} == {2}
    both = tools.get_team_dimensions(MATCH, "home", "out")
    assert {e.period for e in both.evidence} == {1, 2}
    with pytest.raises(ToolError, match="period 3"):
        tools.get_formation(MATCH, "home", "out", period=3)


def test_comparisons_state_both_values_which_is_higher_and_the_gap(tools):
    d = tools.get_team_dimensions(MATCH, "home", "out", period=1)
    c = d.line_height_comparison
    assert c.home == d.line_height_m
    other = tools.get_team_dimensions(MATCH, "away", "out", period=1).line_height_m
    assert c.away == other
    assert c.higher == ("home" if c.home > c.away else "away" if c.away > c.home else None)
    assert c.difference == pytest.approx(abs(c.home - c.away), abs=1e-4)
    p = tools.get_press_stats(MATCH, "home").press_intensity_comparison
    assert p.home == tools.get_press_stats(MATCH, "home").by_third[0].press_intensity
    assert p.away == tools.get_press_stats(MATCH, "away").by_third[0].press_intensity
