"""Scoring rules for agent answers (kept separate from the question bank so each can be
hashed and frozen on its own).

Rule-based scoring for every category. For the false-premise category the primary
scorer is an LLM judge (eval/agent/premise_judge.py); the pattern rule here is the
secondary scorer, and the report shows how often the two agree.
"""

from __future__ import annotations

import re

from questions import _clock_seconds

from regista.agent.grounding import check_clocks, normalise
from regista.agent.tools import Toolbox

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
CORRECTION = re.compile(
    r"not supported|premise|(?:did not|didn't|does not|doesn't|never) (?:switch|change|happen|"
    r"occur|play|press|hold|make|use)|no (?:such|detected|record|evidence|back-?line change|"
    r"press change|change|extra time|third)|actually|in fact|rather than|contrary|incorrect|"
    r"not the case|not correct|was not|wasn't|were not|weren't|there (?:is|was|were) no|"
    r"outside (?:of )?the (?:recorded )?(?:match|period|data)|only (?:two|2) (?:halves|periods)",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def score(q: dict, answer: dict, toolbox: Toolbox) -> dict:
    """Correctness, tool selection, citation validity, abstention for one answer."""
    text = normalise(answer["answer_text"])
    used = [t["tool"] for t in answer["tools_used"]]
    g = q["gold"]
    tmpl = q["template"]
    category = q.get("extra", {}).get("base_category", q["category"])
    full_correction = None
    if category == "false_premise":
        # correct = the premise is explicitly rejected, with grounded numbers; "full
        # correction" additionally names what the data shows (team, time, player, label)
        # (this pattern rule is the secondary scorer; premise_judge.py is primary)
        rejected = bool(CORRECTION.search(text))
        full_correction = rejected and any(t.lower() in text.lower() for t in g["must_mention_any"])
        correct = rejected and answer["status"] == "verified"
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
    if "uncited_clocks" in answer:  # the runtime check, which also exempts tool-error text
        clocks_ok = not answer["uncited_clocks"]
    else:  # answers from before the runtime check: post-hoc, question text exempt only
        clocks_ok = not check_clocks(answer["answer_text"], answer["citations"], [q["question"]])
    return {
        "premise_rejected_pattern": rejected if category == "false_premise" else None,
        "clocks_in_evidence": clocks_ok,
        "full_correction": full_correction,
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
