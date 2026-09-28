"""Template-generated agent questions with gold answers computed from the match store.

No gold answer is hand-written: each template calls the same tools the agent
uses (directly, deterministically) to compute its answer, then states how an
answer is scored. Categories: lookup, comparison, temporal, multi-step,
unanswerable, reliability.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field

from regista.agent.tools import Toolbox, ToolError
from regista.clock import match_clock, parse_clock

TEAMS = ("home", "away")
PHASE_TEXT = {"out": "out of possession", "in": "in possession"}
DECLINE = re.compile(
    r"not (?:be )?(?:available|modell?ed|possible|recorded|in the data)|"
    r"(?:cannot|can't|unable to|no way to) (?:be )?(?:answer|determine|provide|"
    r"know|say|tell|calculate|compute)|outside (?:of )?the (?:recorded )?match|"
    r"(?:does not|doesn't|do not|don't) (?:have|include|contain|model|record|"
    r"track|provide)|not (?:tracked|provided|included|supported)|anonymi[sz]ed",
    re.IGNORECASE,
)
AMBIGUITY = re.compile(
    r"close call|low margin|small margin|narrow|uncertain|ambigu|"
    r"runner-?up|not (?:very )?confident|noisy|could also be|or a \d-\d",
    re.IGNORECASE,
)


@dataclass
class Question:
    qid: str
    match: str
    category: str
    template: str
    question: str
    gold: dict
    expected_tools: list[str]
    gold_range: dict | None = None  # {"period", "t_from", "t_to"} for citation checks
    scoring: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clock_range(period: int, t0: float, t1: float) -> tuple[str, str]:
    return match_clock(period, t0), match_clock(period, t1)


def _clock_seconds(match_clock_text: str) -> float:
    period, seconds = parse_clock(match_clock_text)
    if period == 1:
        return seconds
    if period == 2:
        return 45 * 60 + seconds
    return seconds


def generate(toolbox: Toolbox, match: str, seed: int = 0) -> list[Question]:
    rng = random.Random(f"{match}-{seed}")
    qs: list[Question] = []
    store = toolbox._match(match).store
    formations = store.table("formations").dropna(subset=["label"])
    median_margin = float(store.table("formations")["margin"].median())

    def add(category, template, question, gold, tools, gold_range=None, scoring="", **extra):
        qs.append(
            Question(
                f"{match}:{template}:{len(qs)}",
                match,
                category,
                template,
                question,
                gold,
                tools,
                gold_range,
                scoring,
                extra,
            )
        )

    # ---- lookup: formation in a clear (high-margin) window
    clear = formations[formations["margin"] >= median_margin]
    for r in clear.sample(n=min(2, len(clear)), random_state=rng.randrange(10**6)).itertuples():
        a, b = _clock_range(r.period, r.t_start, r.t_end)
        add(
            "lookup",
            "formation_window",
            f"What formation did the {r.team} team use {PHASE_TEXT[r.phase]} between {a} and {b}?",
            {"label": r.label},
            ["get_formation"],
            {"period": int(r.period), "t_from": float(r.t_start), "t_to": float(r.t_end)},
            "answer names the gold label",
        )
    # ---- lookup: line height in a half
    for team in rng.sample(TEAMS, 2):
        half, (a, b) = rng.choice([(1, ("00:00", "45:00")), (2, ("45:00", None))])
        r = toolbox.get_shape(match, team, "out", a, b)
        add(
            "lookup",
            "line_height_half",
            f"What was the {team} team's defensive line height out of possession in the "
            f"{'first' if half == 1 else 'second'} half?",
            {"value": r.line_height_m, "tolerance": 1.0},
            ["get_shape"],
            {"period": half, "t_from": 0.0, "t_to": 1e9},
            "within 1.0 m of the tool value",
        )
    # ---- lookup: top passing pair
    team = rng.choice(TEAMS)
    net = toolbox.get_pass_network(match, team, top_k=1)
    top = net.top_edges[0]
    add(
        "lookup",
        "top_pass_pair",
        f"Which two {team} players combined for the most passes in the match?",
        {"players": [top.from_player, top.to_player]},
        ["get_pass_network"],
        None,
        "answer names both players of the top directed pair",
    )

    # ---- comparison: who pressed more after a time
    for clock in rng.sample(["30:00", "60:00", "70:00"], 2):
        vals = {
            t: toolbox.get_press_stats(match, t, clock, None).by_third[0].press_intensity
            for t in TEAMS
        }
        if None in vals.values() or abs(vals["home"] - vals["away"]) < 0.02:
            continue
        add(
            "comparison",
            "press_after",
            f"Which team pressed more intensely after {clock}?",
            {"team": max(vals, key=vals.get), "values": vals},
            ["get_press_stats"],
            None,
            "answer names the team with the higher press intensity",
        )
    # ---- comparison: higher line in a half
    half, (a, b) = rng.choice([(1, ("00:00", "45:00")), (2, ("45:00", None))])
    vals = {t: toolbox.get_shape(match, t, "out", a, b).line_height_m for t in TEAMS}
    if None not in vals.values() and abs(vals["home"] - vals["away"]) >= 1.0:
        add(
            "comparison",
            "higher_line",
            f"Which team held a higher defensive line out of possession in the "
            f"{'first' if half == 1 else 'second'} half?",
            {"team": max(vals, key=vals.get), "values": vals},
            ["get_shape"],
            None,
            "answer names the team with the higher line",
        )

    # ---- temporal: first detected moment of a type
    moments = toolbox.find_moments(match).moments
    for kind, text in (
        ("back_line_change", "change the number of players in its back line out of possession"),
        ("press_change", "change its pressing intensity"),
    ):
        for team in TEAMS:
            first = next((m for m in moments if m.type == kind and m.team == team), None)
            if first is None:
                continue
            add(
                "temporal",
                f"first_{kind}",
                f"When did the {team} team first {text}, according to Regista's detectors?",
                {"clock": first.emit_clock, "tolerance_min": 1.0},
                ["find_moments"],
                {"period": first.period, "t_from": 0.0, "t_to": 1e9},
                "answer gives the emit clock within one minute",
            )
            break

    # ---- multi-step: most frequent passer and completion
    team = rng.choice(TEAMS)
    net = toolbox.get_pass_network(match, team, top_k=50)
    passer = max(net.nodes, key=lambda n: n.passes_made).player_id
    pp = toolbox.get_player_passes(match, passer)
    add(
        "multi_step",
        "top_passer_completion",
        f"Which {team} player made the most passes, and what share of their pass attempts "
        f"were completed?",
        {"player": passer, "completion": pp.completed / pp.attempted, "tolerance_pct": 2.0},
        ["get_pass_network", "get_player_passes"],
        None,
        "names the player and a completion share within 2 percentage points",
    )
    # ---- multi-step: formation at the first press change
    first = next((m for m in moments if m.type == "press_change"), None)
    if first is not None:
        t = _clock_seconds(first.emit_clock) if "+" not in first.emit_clock else None
        f = formations[
            (formations["team"] == first.team)
            & (formations["phase"] == "out")
            & (formations["period"] == first.period)
        ]
        emit_t = next(
            m
            for m in store.table("moments").itertuples()
            if m.type == "press_change" and m.team == first.team
        )
        f = f[(f["t_start"] <= emit_t.emit_t) & (f["t_end"] > emit_t.emit_t - 1e-6)]
        if len(f) and t is not None:
            add(
                "multi_step",
                "formation_at_press_change",
                f"When the {first.team} team's pressing first changed, what formation were they "
                f"using out of possession at that moment?",
                {"label": f["label"].iat[0]},
                ["find_moments", "get_formation"],
                {
                    "period": first.period,
                    "t_from": float(f["t_start"].iat[0]),
                    "t_to": float(f["t_end"].iat[0]),
                },
                "answer names the label of that window",
            )

    # ---- unanswerable
    periods = store.manifest["periods"]
    last = max(int(p) for p in periods)
    beyond = (
        f"{int(match_clock(last, periods[str(last)]['t_max']).split('+')[0].split(':')[0]) + 25}:00"
    )
    for template, question in (
        ("xg", "What was the home team's expected goals (xG) in this match?"),
        ("player_name", f"What is the real name of player {passer}?"),
        ("score", "What was the final score?"),
        ("out_of_range", f"What formation did the away team use out of possession at {beyond}?"),
        ("shot_quality", "How good were the away team's shooting chances?"),
    ):
        add(
            "unanswerable",
            template,
            question,
            {"decline": True},
            [],
            None,
            "declines (says it is not available or outside the match) and states no numbers",
        )

    # ---- reliability: formation in a low-margin window
    close = formations[formations["margin"] < median_margin]
    for r in close.sample(n=min(3, len(close)), random_state=rng.randrange(10**6)).itertuples():
        a, b = _clock_range(r.period, r.t_start, r.t_end)
        add(
            "reliability",
            "close_call_formation",
            f"What formation did the {r.team} team use {PHASE_TEXT[r.phase]} between {a} and {b}?",
            {"label": r.label, "runner_up": r.runner_up, "margin": float(r.margin)},
            ["get_formation"],
            {"period": int(r.period), "t_from": float(r.t_start), "t_to": float(r.t_end)},
            "flags the ambiguity (close call, low margin, runner-up)",
        )
    return qs


# ------------------------------------------------------------------------- scoring


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def score(q: dict, answer: dict, toolbox: Toolbox) -> dict:
    """Correctness, tool selection, citation validity, abstention for one answer."""
    text = answer["answer_text"]
    used = [t["tool"] for t in answer["tools_used"]]
    g = q["gold"]
    tmpl = q["template"]
    if q["category"] == "unanswerable":
        declined = bool(DECLINE.search(text))
        correct = declined and not _numbers(
            re.sub(r"\d{1,3}:\d{2}|\d-\d(-\d)+|[A-Za-z]+_?\d+", "", text)
        )
    elif tmpl in ("formation_window", "formation_at_press_change"):
        correct = g["label"] in text
    elif tmpl == "close_call_formation":
        correct = bool(AMBIGUITY.search(text))
    elif tmpl == "line_height_half":
        correct = g["value"] is not None and any(
            abs(v - g["value"]) <= g["tolerance"] for v in _numbers(text)
        )
    elif tmpl == "top_pass_pair":
        correct = all(p in text for p in g["players"])
    elif tmpl in ("press_after", "higher_line"):
        other = "away" if g["team"] == "home" else "home"
        low = text.lower()
        correct = g["team"] in low and (other not in low or low.index(g["team"]) < low.index(other))
    elif tmpl.startswith("first_"):
        want = _clock_seconds(g["clock"]) if "+" not in g["clock"] else None
        clocks = re.findall(r"\d{1,3}\+\d+(?::\d{2})?|\d{1,3}:\d{2}", text)
        correct = any(
            (c == g["clock"])
            or (
                want is not None
                and "+" not in c
                and abs(_clock_seconds(c) - want) <= 60 * g["tolerance_min"]
            )
            for c in clocks
        )
    elif tmpl == "top_passer_completion":
        pct = [v for v in _numbers(text) if 0 <= v <= 100]
        share = 100 * g["completion"]
        correct = g["player"] in text and any(
            abs(v - share) <= g["tolerance_pct"] or abs(100 * v - share) <= g["tolerance_pct"]
            for v in pct
        )
    else:
        raise ValueError(f"no scorer for template {tmpl}")

    tools_ok = set(q["expected_tools"]) <= set(used) if q["expected_tools"] else True
    citations = answer["citations"]
    valid = _citations_valid(citations, q, toolbox)
    return {
        "correct": bool(correct),
        "tool_selection": tools_ok,
        "citation_valid": valid,
        "grounded": answer["status"] == "verified",
        "abstained": bool(DECLINE.search(text)),
    }


def _citations_valid(citations: list[dict], q: dict, toolbox: Toolbox) -> bool | None:
    """Cited frames exist in the match; one citation overlaps the question's range, if any."""
    if q["category"] == "unanswerable":
        return None
    if not citations:
        return False
    m = toolbox._match(q["match"])
    ft = m.frame_times
    for c in citations:
        if c.get("match") != q["match"]:
            return False
        frames = ft[ft["period"] == c["period"]]["frame"]
        if frames.empty or c["frame_start"] < frames.min() or c["frame_end"] > frames.max():
            return False
    rng = q.get("gold_range")
    if rng is None:
        return True
    for c in citations:
        if c["period"] != rng["period"]:
            continue
        sel = ft[
            (ft["period"] == c["period"]) & ft["frame"].between(c["frame_start"], c["frame_end"])
        ]
        if len(sel) and sel["t"].max() >= rng["t_from"] and sel["t"].min() <= rng["t_to"]:
            return True
    return False


__all__ = ["Question", "ToolError", "generate", "score"]
