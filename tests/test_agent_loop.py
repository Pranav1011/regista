"""Agent-loop tests with a scripted fake provider (no model, no network). Synthetic data only."""

import json
from pathlib import Path

import pytest
from synthetic_match import MatchSpec, make_match

from regista.agent.loop import Agent, ProviderReply, tool_schemas
from regista.agent.tools import TOOL_NAMES, Toolbox
from regista.store import build_store

REPO = Path(__file__).resolve().parent.parent
CONFIGS = [
    json.loads((REPO / "eval" / n).read_text())
    for n in ("params_phase1.json", "calibration_phase1.json", "params_phase2.json")
]
MATCH = "synthetic/a1"


class Scripted:
    """Returns pre-written replies in order; records what it was sent."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages, tools=None):
        self.sent.append((list(messages), tools is not None))
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def toolbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("stores")
    build_store(make_match(MatchSpec(period_s=600.0, fps=5.0)), *CONFIGS, "synthetic", "a1", root)
    return Toolbox(root)


def _call(name, **args):
    return ProviderReply("", [(name, args)], 0.0)


def _say(text):
    return ProviderReply(text, [], 0.0)


def test_schemas_cover_every_tool_with_typed_parameters(toolbox):
    schemas, _ = tool_schemas(toolbox)
    assert [s["function"]["name"] for s in schemas] == list(TOOL_NAMES)
    formation = next(s for s in schemas if s["function"]["name"] == "get_formation")
    props = formation["function"]["parameters"]["properties"]
    assert props["team"]["enum"] == ["home", "away"] and "from_clock" in props


def test_grounded_answer_is_verified_with_citations(toolbox):
    label = toolbox.get_formation(MATCH, "home", "out").most_common_label
    provider = Scripted(
        [
            _call("get_formation", match=MATCH, team="home", phase="out"),
            _say(f"Home defended in a {label}."),
        ]
    )
    a = Agent(toolbox, provider).answer("What was home's shape without the ball?", MATCH)
    assert a.status == "verified" and not a.retried
    assert a.citations and a.citations[0]["match"] == MATCH
    assert a.tools_used[0]["tool"] == "get_formation" and not a.tools_used[0]["error"]
    assert any("not probabilities" in c for c in a.caveats)


def test_invented_number_triggers_one_retry_then_unverified(toolbox):
    provider = Scripted(
        [
            _call("get_press_stats", match=MATCH, pressing_team="away"),
            _say("Away pressed in 87% of frames."),
            _say("Away pressed in 87% of frames."),
        ]
    )
    a = Agent(toolbox, provider).answer("How hard did away press?", MATCH)
    assert a.retried and a.status == "unverified" and a.ungrounded_numbers == ["87%"]
    retry_messages, used_tools = provider.sent[-1]
    assert "87%" in retry_messages[-1]["content"] and not used_tools


def test_retry_can_fix_the_answer(toolbox):
    provider = Scripted(
        [
            _call("get_press_stats", match=MATCH, pressing_team="away"),
            _say("Away pressed in 87% of frames."),
            _say("Away's press intensity is reported by the tool above."),
        ]
    )
    a = Agent(toolbox, provider).answer("How hard did away press?", MATCH)
    assert a.retried and a.status == "verified"


def test_tool_errors_reach_the_model_and_unknown_topics_are_declined(toolbox):
    provider = Scripted(
        [
            _call("get_formation", match=MATCH, team="home", phase="out", from_clock="80:00"),
            _call("check_capability", topic="xG"),
            _say("That time is outside the match, and xG is not available."),
        ]
    )
    a = Agent(toolbox, provider).answer("What was the xG after 80:00?", MATCH)
    tool_messages = [m for m in provider.sent[-1][0] if m["role"] == "tool"]
    assert "outside the recorded match" in tool_messages[0]["content"]
    assert a.tools_used[0]["error"] and a.status == "verified"
    assert any("xG" in c for c in a.caveats)


def test_truncated_answers_are_never_verified(toolbox):
    label = toolbox.get_formation(MATCH, "home", "out").most_common_label
    cut = ProviderReply(f"Home defended in a {label} and", [], 0.0, truncated=True)
    provider = Scripted([_call("get_formation", match=MATCH, team="home", phase="out"), cut])
    a = Agent(toolbox, provider).answer("What was home's shape without the ball?", MATCH)
    assert a.status == "unverified" and any("cut off" in c for c in a.caveats)


def test_model_sees_results_without_evidence_but_citations_keep_it(toolbox):
    provider = Scripted(
        [_call("get_formation", match=MATCH, team="home", phase="out"), _say("Done.")]
    )
    a = Agent(toolbox, provider).answer("Shape?", MATCH)
    tool_message = next(m for m in provider.sent[-1][0] if m["role"] == "tool")
    assert '"evidence"' not in tool_message["content"] and a.citations
