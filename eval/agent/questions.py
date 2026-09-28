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

from regista.agent.grounding import normalise
from regista.agent.tools import Toolbox, ToolError
from regista.clock import match_clock, parse_clock

TEAMS = ("home", "away")
PHASE_TEXT = {"out": "out of possession", "in": "in possession"}
DECLINE = re.compile(
    r"not (?:be )?(?:available|modell?ed|possible|recorded|in the data)|"
    r"(?:cannot|can't|unable to|no way to) (?:be )?(?:answer|determine|provide|"
    r"know|say|tell|calculate|compute)|outside (?:of )?the (?:recorded )?(?:match|period|data)|"
    r"(?:does not|doesn't|do not|don't) (?:have|include|contain|model|record|"
    r"track|provide)|not (?:tracked|provided|included|supported)|anonymi[sz]ed|"
    r"no (?:data|information|record)",
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
    expected_tools: list[list[str]]  # any one of these tool sets is a valid route
    gold_range: dict | None = None  # {"period", "t_from", "t_to"} for citation checks
    scoring: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clock_range(period: int, t0: float, t1: float, t_max: float) -> tuple[str, str]:
    """Clock strings for a window, with the end clamped to the last recorded frame."""
    return match_clock(period, t0), match_clock(period, min(t1, t_max))


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
    t_max = {int(p): v["t_max"] for p, v in store.manifest["periods"].items()}

    def add(
        category, template, question, gold, tools, gold_range=None, scoring="", v=None, **extra
    ):
        if v is not None:
            extra["vars"] = v
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
        a, b = _clock_range(r.period, r.t_start, r.t_end, t_max[int(r.period)])
        add(
            "lookup",
            "formation_window",
            f"What formation did the {r.team} team use {PHASE_TEXT[r.phase]} between {a} and {b}?",
            {"label": r.label},
            [["get_formation"]],
            {"period": int(r.period), "t_from": float(r.t_start), "t_to": float(r.t_end)},
            "answer names the gold label",
            v={"team": r.team, "phase_text": PHASE_TEXT[r.phase], "a": a, "b": b},
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
            [["get_shape"]],
            {"period": half, "t_from": 0.0, "t_to": 1e9},
            "within 1.0 m of the tool value",
            v={"team": team, "half": "first" if half == 1 else "second"},
        )
    # ---- lookup: top passing pair
    team = rng.choice(TEAMS)
    net = toolbox.get_pass_network(match, team, top_k=1)
    top = net.top_edges[0]
    add(
        "lookup",
        "top_pass_pair",
        f"Which pair of {team} players passed to each other most often?",
        {"players": [top.from_player, top.to_player]},
        [["get_pass_network"]],
        None,
        "answer names both players of the top directed pair",
        v={"team": team},
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
            [["get_press_stats"]],
            None,
            "answer names the team with the higher press intensity",
            v={"clock": clock},
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
            [["get_shape"]],
            None,
            "answer names the team with the higher line",
            v={"half": "first" if half == 1 else "second"},
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
                [["find_moments"]],
                {"period": first.period, "t_from": 0.0, "t_to": 1e9},
                "answer gives the emit clock within one minute",
                v={"team": team},
            )
            break

    # ---- multi-step: most frequent passer and completion
    team = rng.choice(TEAMS)
    net = toolbox.get_pass_network(match, team, top_k=50)
    passer = max(net.nodes, key=lambda n: n.passes_made).player_id  # most completed passes
    pp = toolbox.get_player_passes(match, passer)
    top_attempts = toolbox.get_team_passing(match, team).players[0]  # most attempts
    # "made the most passes" can mean completed passes or attempts; both readings are accepted
    accepted = [
        {"player": passer, "completion": pp.completed / pp.attempted},
        {"player": top_attempts.player_id, "completion": top_attempts.completion_share},
    ]
    add(
        "multi_step",
        "top_passer_completion",
        f"Which {team} player made the most passes, and what share of their pass attempts "
        f"were completed?",
        {"accepted": accepted, "tolerance_pct": 2.0},
        [["get_team_passing"], ["get_pass_network", "get_player_passes"]],
        None,
        "names the player and a completion share within 2 percentage points",
        v={"team": team},
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
        f = f.sort_values("t_start").reset_index(drop=True)
        containing = f.index[(f["t_start"] <= emit_t.emit_t) & (f["t_end"] > emit_t.emit_t - 1e-6)]
        # "at that moment" is read as the window containing it or the one just before it
        chosen = [i for k in containing for i in (k - 1, k) if 0 <= i < len(f)]
        f = f.loc[chosen]
        if len(f) and t is not None:
            add(
                "multi_step",
                "formation_at_press_change",
                f"When the {first.team} team's pressing first changed, what formation were they "
                f"using out of possession at that moment?",
                {"labels": sorted(set(f["label"].dropna()))},
                [["find_moments", "get_formation"]],
                {
                    "period": first.period,
                    "t_from": float(f["t_start"].min()),
                    "t_to": float(f["t_end"].max()),
                },
                "answer names the label of the window containing that moment or the one before",
                v={"team": first.team},
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
            "declines (not available / outside the match); any number must be grounded",
            v={},
        )

    # ---- reliability: formation in a low-margin window
    close = formations[formations["margin"] < median_margin]
    for r in close.sample(n=min(3, len(close)), random_state=rng.randrange(10**6)).itertuples():
        a, b = _clock_range(r.period, r.t_start, r.t_end, t_max[int(r.period)])
        add(
            "reliability",
            "close_call_formation",
            f"What formation did the {r.team} team use {PHASE_TEXT[r.phase]} between {a} and {b}?",
            {"label": r.label, "runner_up": r.runner_up, "margin": float(r.margin)},
            [["get_formation"]],
            {"period": int(r.period), "t_from": float(r.t_start), "t_to": float(r.t_end)},
            "flags the ambiguity (close call, low margin, runner-up)",
            v={"team": r.team, "phase_text": PHASE_TEXT[r.phase], "a": a, "b": b},
        )

    false_premises(add, toolbox, match, formations, moments, t_max, rng)
    return [*qs, *paraphrases(qs)]


# ---------------------------------------------------------------- paraphrase category

# Three rewordings per template. Each is asked about the same instance as the
# template's first question in a match, with the same gold answer and scoring.
PARAPHRASES: dict[str, tuple[str, str, str]] = {
    "formation_window": (
        "Between {a} and {b}, how was the {team} team lined up {phase_text}?",
        "In the {a} to {b} window, what was the {team} side's formation {phase_text}?",
        "{team} team, {phase_text}, from {a} to {b}: which formation?",
    ),
    "line_height_half": (
        "How far from their own goal did the {team} team's back line sit out of possession "
        "in the {half} half?",
        "Out of possession in the {half} half, how high was the {team} team's defensive line?",
        "Give me the {team} team's out-of-possession line height for the {half} half.",
    ),
    "top_pass_pair": (
        "Which two {team} players combined for the most passes in the match?",
        "Which {team} passing connection, from one player to another, was used most often?",
        "In the {team} team, who passed to whom most often?",
    ),
    "press_after": (
        "After {clock}, which side's press was more intense?",
        "From {clock} onwards, did the home or the away team press harder?",
        "Compare the two teams' pressing intensity after {clock}: which was higher?",
    ),
    "higher_line": (
        "In the {half} half, which team defended with the higher line out of possession?",
        "Out of possession in the {half} half, whose defensive line sat higher up the pitch?",
        "Which side pushed its defensive line further up without the ball in the {half} half?",
    ),
    "first_back_line_change": (
        "According to Regista's detectors, at what time did the {team} team first alter the "
        "number of defenders in its back line?",
        "When was the {team} team's first detected back-line change?",
        "What match clock does Regista give for the {team} team's first back-line change?",
    ),
    "first_press_change": (
        "At what time did Regista first detect a change in the {team} team's pressing?",
        "When did the {team} team's press intensity first shift, according to the detectors?",
        "What match clock does Regista give for the {team} team's first press change?",
    ),
    "top_passer_completion": (
        "Who was the {team} team's busiest passer, and what was their pass completion rate?",
        "Which {team} player attempted the most passes, and what percentage did they complete?",
        "Name the {team} player with the most passes and their completion share.",
    ),
    "formation_at_press_change": (
        "What formation was the {team} team in out of possession when its pressing first changed?",
        "At the {team} team's first detected press change, which out-of-possession shape were "
        "they using?",
        "When Regista first flagged a {team} press change, what was their formation without "
        "the ball?",
    ),
    "close_call_formation": (
        "How was the {team} team set up {phase_text} from {a} to {b}, and how certain is that?",
        "What shape did the {team} team use {phase_text} between {a} and {b}? Is the label "
        "reliable?",
        "Between {a} and {b}, what was the {team} team's formation {phase_text}?",
    ),
    "xg": (
        "How many expected goals did the home team create?",
        "What was the home side's xG?",
        "Give me the home team's expected-goals total.",
    ),
}


def paraphrases(qs: list[Question]) -> list[Question]:
    out, done = [], set()
    for q in qs:
        if q.template not in PARAPHRASES or q.template in done or "vars" not in q.extra:
            continue
        done.add(q.template)
        for k, text in enumerate(PARAPHRASES[q.template]):
            out.append(Question(
                f"{q.qid}~p{k}", q.match, "paraphrase", q.template, text.format(**q.extra["vars"]),
                q.gold, q.expected_tools, q.gold_range, q.scoring,
                {"base_qid": q.qid, "base_category": q.category, "variant": k},
            ))  # fmt: skip
    return out


# ------------------------------------------------------------- false-premise category

CORRECTION = re.compile(
    r"not supported|premise|(?:did not|didn't|does not|doesn't|never) (?:switch|change|happen|"
    r"occur|play|press|hold|make|use)|no (?:such|detected|record|evidence|back-?line change|"
    r"press change|change|extra time|third)|actually|in fact|rather than|contrary|incorrect|"
    r"not the case|not correct|was not|wasn't|were not|weren't|there (?:is|was|were) no|"
    r"outside (?:of )?the (?:recorded )?(?:match|period|data)|only (?:two|2) (?:halves|periods)",
    re.IGNORECASE,
)


def false_premises(add, toolbox: Toolbox, match: str, formations, moments, t_max, rng) -> None:
    """Questions that assert something the data contradicts."""
    other = {"home": "away", "away": "home"}

    def fp(template, question, gold, tools, gold_range=None):
        add(
            "false_premise",
            template,
            question,
            gold,
            tools,
            gold_range,
            "corrects the premise and states what the data shows",
        )

    # 1. a switch to a back five that never happened
    out = formations[formations["phase"] == "out"]
    for team in TEAMS:
        if not (out[out["team"] == team]["label"].str.startswith("5")).any():
            clock = rng.choice(["55:00", "62:00", "70:00"])
            fp(
                "fp_back_five",
                f"Why did the {team} team switch to a back five at {clock}?",
                {"must_mention_any": ["no", "not"], "truth": "no back-five window"},
                [["get_formation"], ["find_moments"]],
            )
            break
    # 2. a real moment attributed to the wrong team
    press = [m for m in moments if m.type == "press_change"]
    if press and not any(
        m.team != press[0].team for m in press if m.emit_clock == press[0].emit_clock
    ):
        m = press[0]
        if not any(x.type == "press_change" and x.team == other[m.team] for x in moments):
            fp(
                "fp_wrong_team",
                f"Why did the {other[m.team]} team change its pressing at {m.emit_clock}?",
                {"must_mention_any": [m.team], "truth": f"it was {m.team}"},
                [["find_moments"]],
            )
    # 3. extra time that does not exist
    fp(
        "fp_extra_time",
        "Why did the home team drop deep during extra time?",
        {"must_mention_any": ["extra time", "90", "no"], "truth": "no extra time"},
        [["get_match_overview"], ["get_shape"], ["get_formation"]],
    )
    # 4. the wrong formation for a clear window
    clear = formations[formations["margin"] >= formations["margin"].median()]
    if len(clear):
        r = clear.sample(n=1, random_state=rng.randrange(10**6)).iloc[0]
        wrong = next(
            w
            for w in ("5-3-2", "3-4-3", "4-3-3", "4-4-2", "3-5-2")
            if w not in (r["label"], r["runner_up"])
        )
        a, b = _clock_range(int(r["period"]), r["t_start"], r["t_end"], t_max[int(r["period"])])
        fp(
            "fp_wrong_formation",
            f"Why did the {r['team']} team play a {wrong} "
            f"{PHASE_TEXT[r['phase']]} between {a} and {b}?",
            {"must_mention_any": [r["label"]], "truth": r["label"]},
            [["get_formation"]],
            {"period": int(r["period"]), "t_from": float(r["t_start"]), "t_to": float(r["t_end"])},
        )
    # 5. the weaker pressing team said to press harder
    vals = {
        t: toolbox.get_press_stats(match, t, "60:00", None).by_third[0].press_intensity
        for t in TEAMS
    }
    if None not in vals.values() and abs(vals["home"] - vals["away"]) >= 0.02:
        weak, strong = sorted(TEAMS, key=vals.get)
        fp(
            "fp_press_harder",
            f"Why did the {weak} team press so much harder than the {strong} team after 60:00?",
            {"must_mention_any": [strong], "truth": f"{strong} pressed more"},
            [["get_press_stats"]],
        )
    # 6. the lower line said to be higher
    vals = {t: toolbox.get_shape(match, t, "out", "00:00", "45:00").line_height_m for t in TEAMS}
    if None not in vals.values() and abs(vals["home"] - vals["away"]) >= 1.0:
        low, high = sorted(TEAMS, key=vals.get)
        fp(
            "fp_higher_line",
            f"Why did the {low} team hold a higher defensive line than the "
            f"{high} team in the first half?",
            {"must_mention_any": [high], "truth": f"{high} held the higher line"},
            [["get_shape"]],
        )
    # 7. a moment type that never happened for a team
    for team in TEAMS:
        if not any(m.type == "back_line_change" and m.team == team for m in moments):
            fp(
                "fp_no_back_line_change",
                f"At what time did the {team} team change its back line, and why?",
                {
                    "must_mention_any": ["no", "not", "none"],
                    "truth": "no back-line change detected",
                },
                [["find_moments"]],
            )
            break
    # 8. the wrong player named as the top passer
    for team in TEAMS:
        players = toolbox.get_team_passing(match, team).players
        if len(players) >= 5:
            wrong = players[4].player_id
            fp(
                "fp_top_passer",
                f"Why did {wrong} make the most passes for the {team} team?",
                {"must_mention_any": [players[0].player_id], "truth": players[0].player_id},
                [["get_team_passing"], ["get_pass_network"]],
            )
            break
    # 9. a third half
    fp(
        "fp_third_half",
        "What did the away team change in the third half?",
        {"must_mention_any": ["two", "2", "no", "not"], "truth": "only two halves"},
        [["get_match_overview"]],
    )
    # 10. a real moment moved to the wrong time
    if press:
        m = press[0]
        mins = int(m.emit_clock.split("+")[0].split(":")[0])
        wrong_clock = f"{(mins + 20) % 90:02d}:00" if mins + 20 < 90 else f"{mins - 20:02d}:00"
        fp(
            "fp_wrong_time",
            f"Why did the {m.team} team's pressing change at {wrong_clock}?",
            {"must_mention_any": [m.emit_clock], "truth": f"detected at {m.emit_clock}"},
            [["find_moments"]],
        )


# ------------------------------------------------------------------------- scoring


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def score(q: dict, answer: dict, toolbox: Toolbox) -> dict:
    """Correctness, tool selection, citation validity, abstention for one answer."""
    text = normalise(answer["answer_text"])
    used = [t["tool"] for t in answer["tools_used"]]
    g = q["gold"]
    tmpl = q["template"]
    category = q.get("extra", {}).get("base_category", q["category"])
    if category == "false_premise":
        g_any = g["must_mention_any"]
        correct = bool(CORRECTION.search(text)) and any(
            tok.lower() in text.lower() for tok in g_any
        )
    elif category == "unanswerable":
        # a correct refusal declines and states only numbers the tools returned
        correct = bool(DECLINE.search(text)) and answer["status"] == "verified"
    elif tmpl == "formation_window":
        correct = g["label"] in text
    elif tmpl == "formation_at_press_change":
        correct = any(label in text for label in g["labels"])
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
        correct = any(
            acc["player"] in text
            and any(
                abs(v - 100 * acc["completion"]) <= g["tolerance_pct"]
                or abs(100 * v - 100 * acc["completion"]) <= g["tolerance_pct"]
                for v in pct
            )
            for acc in g["accepted"]
        )
    else:
        raise ValueError(f"no scorer for template {tmpl}")

    alternatives = q["expected_tools"]
    tools_ok = any(set(alt) <= set(used) for alt in alternatives) if alternatives else True
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
    if q.get("extra", {}).get("base_category", q["category"]) == "unanswerable":
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
