"""Hand-label match summaries so judge-human agreement can be measured.

Shows a summary and its fact sheet, asks for the same 1-5 rubric scores the judge
gives, and appends each label to eval/agent/human_labels.jsonl. Items are drawn
from the judged summaries in a fixed random order; already-labelled items are
skipped, so you can stop and resume.

  uv run python eval/agent/label.py --split test --n 30
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

from run import RESULTS  # noqa: E402
from summaries import CRITERIA, RUBRIC, fact_sheet  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.tools import Toolbox  # noqa: E402

LABELS = HERE / "human_labels.jsonl"


def items(split: str) -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob(f"summaries_{split}_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out.append(
                {"match": r["match"], "model": r["model"], "summary": r["summary"]["answer_text"]}
            )
    random.Random(0).shuffle(out)
    return out


def ask(criterion: str) -> int:
    while True:
        raw = input(f"  {criterion} (1-5, q to quit): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("  please type a number from 1 to 5")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    done = set()
    if LABELS.exists():
        done = {(r["match"], r["model"]) for r in map(json.loads, LABELS.read_text().splitlines())}
    toolbox = Toolbox(io.data_dir() / "store")
    todo = [it for it in items(args.split) if (it["match"], it["model"]) not in done]
    todo = todo[: max(args.n - len(done), 0)]
    print(RUBRIC.split("Answer only")[0])
    try:
        for i, it in enumerate(todo, 1):
            print("=" * 80)
            print(f"[{len(done) + i}/{args.n}] {it['match']} (model hidden)")
            print("FACT SHEET:")
            print(json.dumps(fact_sheet(toolbox, it["match"]), indent=1, default=str))
            print("\nSUMMARY:")
            print(textwrap.fill(it["summary"], 100))
            scores = {c: ask(c) for c in CRITERIA}
            with LABELS.open("a") as fh:
                fh.write(json.dumps({"match": it["match"], "model": it["model"], **scores}) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; labels so far are saved")


if __name__ == "__main__":
    main()
