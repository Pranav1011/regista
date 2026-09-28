"""Hand-label a stratified sample so each automatic scorer's agreement with a human is measured.

30 items, drawn with a fixed seed from one split's results:
  - 10 false-premise answers: the same two questions the premise judge answers
    (rejects the premise? with evidence?), for judge-human agreement;
  - 10 match summaries: the same 1-5 rubric the summary judge gives;
  - 10 answers from the remaining categories, spread across them round-robin:
    is the answer correct?, for rule-scorer-human agreement.
Each label is appended to eval/agent/human_labels.jsonl as it is given; the model
and the automatic verdict are hidden; already-labelled items are skipped, so you
can stop and resume.

  uv run python eval/agent/label.py --split test
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    done = set()
    if LABELS.exists():
        done = {(r["kind"], r["id"]) for r in map(json.loads, LABELS.read_text().splitlines())}
    items = sample(args.split)
    todo = [it for it in items if (it["kind"], it["id"]) not in done]
    toolbox = Toolbox(io.data_dir() / "store")
    print(RUBRIC.split("Answer only")[0])
    try:
        for i, it in enumerate(todo, 1):
            print("=" * 80)
            print(f"[{len(items) - len(todo) + i}/{len(items)}] {it['kind']}")
            scores = label(it, toolbox)
            with LABELS.open("a") as fh:
                fh.write(
                    json.dumps({"split": args.split, "kind": it["kind"], "id": it["id"], **scores})
                    + "\n"
                )
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; labels so far are saved")


if __name__ == "__main__":
    main()
