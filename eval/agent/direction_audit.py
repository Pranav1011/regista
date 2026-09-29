"""Deterministic audit of line-height direction words in test answers and summaries.

Post-test analysis only; the agent is not changed. For each sentence that uses a
direction word about a line (higher, deeper, lower, further up, dropped deep, and
raised / increased / decreased for changes) each claim is checked against the values
stated in the same sentence:
  - two teams with a value each ("home ... 34.07 m while away sat deeper at 35.13 m"):
    the team the word attaches to must have the higher value for higher / further up,
    and the lower value for deeper / lower / dropped deep;
  - a change "from A to B": B must be above A for higher / further up / pushed, and
    below A for deeper / lower / dropped.
Sentences with a direction word but no such values are counted as unchecked.

  uv run python eval/agent/direction_audit.py --split test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import FROZEN, RESULTS, results_path  # noqa: E402

from regista.agent.grounding import normalise  # noqa: E402

UP = r"higher|further up|pushed (?:up|higher)|raised|rose|increased"
DOWN = r"deeper|lower|dropped deep(?:er)?|dropped|dropping deeper|decreased|fell"
DIRECTION = re.compile(rf"\b(?P<up>{UP})\b|\b(?P<down>{DOWN})\b", re.I)
LINE_CONTEXT = re.compile(r"\blines?\b|line height|\bsat\b|pushed|dropped deep|defended", re.I)
BACK_LINE = re.compile(r"back[ -]line|defenders", re.I)  # counts, not heights
# a team's value: the first "<value> m" after the team name, not a difference ("by 4.13 m")
TEAM_VALUE = re.compile(
    r"\b(?P<team>home|away)\b(?:(?!\b(?:home|away)\b)[^.;])*?"
    r"(?<!by )(?<!gap of )(?<!difference of )(?<![\d.])"
    r"(?P<v>\d+(?:\.\d+)?)\s?(?:m\b|metres|meters)",
    re.I,
)
FROM_TO = re.compile(
    r"from ~?(?P<a>\d+(?:\.\d+)?)\s?(?:m|metres|meters)? (?:back up |back )?to "
    r"~?(?P<b>\d+(?:\.\d+)?)",
    re.I,
)
SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z])")


NEAR_CHARS = 60  # a direction word describes a "from A to B" only if it is this close


def _gap(c: re.Match, d: re.Match) -> int:
    return max(0, c.start() - d.end(), d.start() - c.end())


def check_sentence(s: str) -> list[tuple[str, str]]:
    """(verdict, detail) for each direction claim in a sentence about line height.

    Each direction word is checked against the nearest "from A to B" in the sentence;
    without one, against the two teams' values in the sentence."""
    if not LINE_CONTEXT.search(s) or BACK_LINE.search(s):
        return []
    changes = list(FROM_TO.finditer(s))
    pairs: dict[str, float] = {}
    for m in TEAM_VALUE.finditer(s):
        pairs.setdefault(m["team"].lower(), float(m["v"]))
    out = []
    words = list(DIRECTION.finditer(s))
    for d in words:
        up = d.group("up") is not None
        word = d.group(0)
        near = [c for c in changes if _gap(c, d) <= NEAR_CHARS]
        if not near and len(changes) == 1 and len(words) == 1:
            near = changes  # one claim and one change in the sentence: they belong together
        if near:
            ft = min(near, key=lambda c: _gap(c, d))
            a, b = float(ft["a"]), float(ft["b"])
            ok = (b > a) if up else (b < a)
            out.append(("ok" if ok else "error", f"{word!r}: from {a} to {b}"))
        elif not changes and len(pairs) == 2:
            before = list(re.finditer(r"\b(home|away)\b", s[: d.start()], re.I))
            if not before:
                out.append(("unchecked", f"{word!r}: no team before the word"))
                continue
            subject = before[-1].group(1).lower()
            other = "away" if subject == "home" else "home"
            ok = pairs[subject] > pairs[other] if up else pairs[subject] < pairs[other]
            out.append((
                "ok" if ok else "error",
                f"{subject} {word!r}: {subject} {pairs[subject]} m vs {other} {pairs[other]} m",
            ))  # fmt: skip
        else:
            out.append(("unchecked", f"{word!r}: no nearby from-to or team values"))
    return out


def texts(split: str) -> list[tuple[str, str, str]]:
    """(source, id, text) for every answer and summary of the frozen model on a split."""
    model = json.loads(FROZEN.read_text())["model"]
    out = [
        ("answer", r["question"]["qid"], r["answer"]["answer_text"])
        for r in map(json.loads, results_path(split, model).read_text().splitlines())
    ]
    summaries = RESULTS / f"summaries_{split}_{model.replace(':', '_')}.jsonl"
    out += [
        ("summary", r["match"], r["summary"]["answer_text"])
        for r in map(json.loads, summaries.read_text().splitlines())
    ]
    return out


def audit(split: str) -> dict:
    rows = []
    for source, ident, text in texts(split):
        for sent in SENTENCE.split(normalise(text)):
            for verdict, detail in check_sentence(sent):
                rows.append({"source": source, "id": ident, "verdict": verdict,
                             "detail": detail, "sentence": sent.strip()})  # fmt: skip
    summary = {}
    for source in ("answer", "summary"):
        r = [x for x in rows if x["source"] == source]
        checked = [x for x in r if x["verdict"] != "unchecked"]
        errors = [x for x in checked if x["verdict"] == "error"]
        summary[source] = {
            "claims": len(r),
            "checked": len(checked),
            "errors": len(errors),
            "error_rate": len(errors) / len(checked) if checked else None,
            "texts_with_error": len({x["id"] for x in errors}),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    res = audit(args.split)
    out = RESULTS / f"direction_audit_{args.split}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    for x in res["rows"]:
        if x["verdict"] == "error":
            print(f"ERROR {x['source']} {x['id']}: {x['detail']}\n    {x['sentence'][:220]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
