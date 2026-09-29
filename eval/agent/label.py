"""Hand-label a stratified sample so each automatic scorer's agreement with a human is measured.

30 items drawn with a fixed seed from one split's results, plus an extra stratum:
  - 10 false-premise answers (from matches other than the extra stratum's): the
    same two questions the premise judge answers (rejects the premise? with
    evidence?), for judge-human agreement;
  - 10 match summaries: the same 1-5 rubric the summary judge gives;
  - 10 answers from the remaining categories, spread across them round-robin:
    is the answer correct?, for rule-scorer-human agreement;
  - extra stratum: every false-premise answer on Metrica game 3 (on the test split
    the judge scored them 1.00 and the pattern rule 0.375), same two questions.
Each label is appended to eval/agent/human_labels.jsonl as it is given, with its
round, item number, and optional notes; the model and the automatic verdict are
hidden; already-labelled items are skipped, so you can stop and resume.

Rounds: "human_unassisted" (the first pass) and "human_reviewed" (revisions made
after a rubric-consistency review, by item number, recording which items were
discussed with Claude). Agreement is reported against both.

  uv run python eval/agent/label.py --split test
  uv run python eval/agent/label.py --split test --revise 1 18 20 --discussed 1 18 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import FROZEN, RESULTS, results_path  # noqa: E402
from summaries import CRITERIA, RUBRIC, fact_sheet  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.tools import Toolbox  # noqa: E402

LABELS = HERE / "human_labels.jsonl"
PER_STRATUM = 10
SEED = 0
EXTRA_PREMISE_MATCH = "metrica/3"  # every false-premise item here is labelled


def _answers(split: str) -> list[dict]:
    model = json.loads(FROZEN.read_text())["model"]
    path = results_path(split, model)
    if not path.exists():
        sys.exit(f"no {path.name}; run the {split} split first")
    return [json.loads(line) for line in path.read_text().splitlines()]


def sample(split: str) -> list[dict]:
    rng = random.Random(SEED)
    rows = _answers(split)
    premise = [r for r in rows if r["question"]["category"] == "false_premise"]
    extra = [r for r in premise if r["question"]["match"] == EXTRA_PREMISE_MATCH]
    premise = [r for r in premise if r["question"]["match"] != EXTRA_PREMISE_MATCH]
    rng.shuffle(premise)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        if r["question"]["category"] != "false_premise":
            by_cat.setdefault(r["question"]["category"], []).append(r)
    for group in by_cat.values():
        rng.shuffle(group)
    other: list[dict] = []
    cats = sorted(by_cat)
    while len(other) < PER_STRATUM and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(other) < PER_STRATUM:
                other.append(by_cat[c].pop())
    summaries = [
        json.loads(line)
        for f in sorted(RESULTS.glob(f"summaries_{split}_*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    rng.shuffle(summaries)
    items = [
        {"kind": "false_premise", "id": r["question"]["qid"], "row": r}
        for r in premise[:PER_STRATUM]
    ]
    items += [
        {"kind": "summary", "id": f"{s['match']}|{s['model']}", "row": s}
        for s in summaries[:PER_STRATUM]
    ]
    items += [{"kind": "answer", "id": r["question"]["qid"], "row": r} for r in other]
    items += [{"kind": "false_premise", "id": r["question"]["qid"], "row": r} for r in extra]
    rng.shuffle(items)
    return items


def ask(prompt: str, allowed: dict[str, object]) -> object:
    keys = "/".join(allowed)
    while True:
        raw = input(f"  {prompt} ({keys}, q to quit): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in allowed:
            return allowed[raw]
        print(f"  please type one of {keys}")


YES_NO = {"y": True, "n": False}
SCALE = {str(i): i for i in range(1, 6)}


def show_answer(row: dict) -> None:
    q, a = row["question"], row["answer"]
    print(f"QUESTION: {q['question']}")
    print(f"WHAT THE DATA SHOWS: {json.dumps(q['gold'], default=str)}")
    print("\nANSWER:")
    print(textwrap.fill(a["answer_text"], 100))
    cites = [f"P{c['period']} {c['clock_start']}-{c['clock_end']}" for c in a["citations"]]
    print(f"EVIDENCE CITED: {', '.join(cites) or 'none'}")


def label(item: dict, toolbox: Toolbox) -> dict:
    row = item["row"]
    if item["kind"] == "summary":
        print(f"MATCH SUMMARY: {row['match']} (model hidden)")
        print("FACT SHEET:")
        print(json.dumps(fact_sheet(toolbox, row["match"]), indent=1, default=str))
        print("\nSUMMARY:")
        print(textwrap.fill(row["summary"]["answer_text"], 100))
        return {c: ask(c, SCALE) for c in CRITERIA}
    show_answer(row)
    if item["kind"] == "false_premise":
        return {
            "rejects_premise": ask("rejects the premise, explicitly or implicitly?", YES_NO),
            "with_evidence": ask("supports the rejection with evidence?", YES_NO),
        }
    return {"correct": ask("is the answer correct?", YES_NO)}


ROUNDS = ("human_unassisted", "human_reviewed")


def load_labels(round_: str = "human_unassisted") -> dict[tuple[str, str], dict]:
    """Labels per (kind, id). ``human_reviewed`` is the unassisted labels with every later
    revision applied; rows without a round are unassisted."""
    out: dict[tuple[str, str], dict] = {}
    if not LABELS.exists():
        return out
    for r in map(json.loads, LABELS.read_text().splitlines()):
        rnd = r.get("round", "human_unassisted")
        if rnd == "human_unassisted" or round_ == "human_reviewed":
            out[(r["kind"], r["id"])] = r
    return out


def _write(row: dict) -> None:
    with LABELS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _note() -> str:
    return input("  notes (optional, Enter to skip): ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--revise",
        type=int,
        nargs="+",
        metavar="N",
        help="re-label items by their [N/total] number; stored as round human_reviewed",
    )
    ap.add_argument(
        "--discussed",
        type=int,
        nargs="*",
        default=[],
        metavar="N",
        help="item numbers discussed with Claude before revising (recorded on the revision)",
    )
    args = ap.parse_args()
    items = sample(args.split)
    toolbox = Toolbox(io.data_dir() / "store")
    print(RUBRIC.split("Answer only")[0])
    try:
        if args.revise:
            current = load_labels("human_reviewed")
            for n in args.revise:
                if not 1 <= n <= len(items):
                    sys.exit(f"no item {n}; items are numbered 1-{len(items)}")
                it = items[n - 1]
                prev = current.get((it["kind"], it["id"]))
                print("=" * 80)
                print(f"[{n}/{len(items)}] {it['kind']} (revising)")
                if prev:
                    shown = {k: v for k, v in prev.items() if k not in ("split", "kind", "id")}
                    print(f"CURRENT LABEL: {json.dumps(shown)}")
                scores = label(it, toolbox)
                _write(
                    {
                        "split": args.split,
                        "kind": it["kind"],
                        "id": it["id"],
                        **scores,
                        "round": "human_reviewed",
                        "index": n,
                        "notes": _note(),
                        "discussed_with_claude": n in args.discussed,
                    }
                )
            return
        done = set(load_labels())
        todo = [(n, it) for n, it in enumerate(items, 1) if (it["kind"], it["id"]) not in done]
        for n, it in todo:
            print("=" * 80)
            print(f"[{n}/{len(items)}] {it['kind']}")
            scores = label(it, toolbox)
            _write(
                {
                    "split": args.split,
                    "kind": it["kind"],
                    "id": it["id"],
                    **scores,
                    "round": "human_unassisted",
                    "index": n,
                    "notes": _note(),
                }
            )
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; labels so far are saved")


if __name__ == "__main__":
    main()
