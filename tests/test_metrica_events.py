"""Tests for Metrica event parsing. All data here is synthetic, by design."""

import json

import pandas as pd
import pytest

from regista.ingest.metrica_events import parse_csv, parse_json, to_canonical

CSV = (
    (
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
        "From,To,Start X,Start Y,End X,End Y\n"
    )
    + """Home,PASS,,1,10,0.4,20,0.8,Player7,Player9,0.25,0.75,0.5,0.5
Away,BALL LOST,INTERCEPTION,2,30,1.2,40,1.6,Player21,,0.5,0.5,NaN,NaN
"""
)


def test_parse_csv_maps_players_and_flips_y(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(CSV)
    ev = parse_csv(path)
    assert ev["from_player"].tolist() == ["home_7", "away_21"]
    assert ev.loc[0, "to_player"] == "home_9"
    assert pd.isna(ev.loc[1, "to_player"])
    assert ev.loc[0, "start_y"] == pytest.approx(0.25)  # CSV y is up; tracking y is down


def test_parse_json_resolves_team_and_subtypes(tmp_path):
    data = {
        "data": [
            {
                "period": 1,
                "type": {"name": "PASS"},
                "subtypes": None,
                "start": {"frame": 5, "x": 0.5, "y": 0.5},
                "end": {"frame": 9, "x": 0.6, "y": 0.4},
                "from": {"id": "P1"},
                "to": {"id": "P2"},
            },
            {
                "period": 1,
                "type": {"name": "CHALLENGE"},
                "subtypes": [{"name": "GROUND"}, {"name": "WON"}],
                "start": {"frame": 12, "x": 0.5, "y": 0.5},
                "end": {"frame": 12, "x": 0.5, "y": 0.5},
                "from": {"id": "P9"},
                "to": None,
            },
        ]
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps(data))
    players = pd.DataFrame({"player_id": ["P1", "P2", "P9"], "team": ["home", "home", "away"]})
    ev = parse_json(path, players)
    assert ev["team"].tolist() == ["home", "away"]
    assert ev.loc[1, "subtype"] == "GROUND-WON"
    assert pd.isna(ev.loc[0, "subtype"])


def test_to_canonical_converts_and_flips_periods():
    raw = pd.DataFrame(
        {
            "period": [1, 2],
            "type": ["PASS", "PASS"],
            "subtype": [None, None],
            "team": ["home", "home"],
            "from_player": ["h1", "h1"],
            "to_player": ["h2", "h2"],
            "start_frame": [1, 2],
            "end_frame": [3, 4],
            "start_x": [1.0, 1.0],
            "start_y": [0.0, 0.0],
            "end_x": [0.5, 0.5],
            "end_y": [0.5, 0.5],
        }
    )
    ev = to_canonical(raw, flipped_periods=[2])
    assert (ev.loc[0, "start_x"], ev.loc[0, "start_y"]) == pytest.approx((52.5, 34.0))
    assert (ev.loc[1, "start_x"], ev.loc[1, "start_y"]) == pytest.approx((-52.5, -34.0))
