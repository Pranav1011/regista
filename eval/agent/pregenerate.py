"""Pre-generate agent answers for the hosted viewer's suggested questions.

Uses the frozen model and prompt (eval/agent/frozen.json). Answers are written to
eval/agent/answers/<viewer-id>.json (committed, since the hosted demo is static
and cannot run a model) and copied into the viewer data by `regista export-viewer`.
Each answer keeps its verification status and citations.

  uv run python eval/agent/pregenerate.py --game 1 --game 2 --game 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import FROZEN, check_frozen  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.loop import Agent, OllamaProvider  # noqa: E402
from regista.agent.tools import Toolbox  # noqa: E402

ANSWERS = HERE / "answers"
SUGGESTED = (
    "What formation did each team use without the ball, and how clear-cut was it?",
    "Which team pressed more intensely in the second half?",
    "When did either team change its back line, according to the detectors?",
    "How high was each team's defensive line out of possession in the first half?",
    "Which home player attempted the most passes, and how many did they complete?",
    "Which away player attempted the most passes, and how many did they complete?",
    "What tactical moments did Regista detect after 60:00?",
    "In which third did the away team press most?",
    "What was the home team's formation in possession between 60:00 and 65:00?",
    "What was the expected goals (xG) for each team?",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", type=int, action="append", required=True)
    ap.add_argument(
        "--release",
        type=Path,
        default=FROZEN,
        help="frozen.json (v1.0) or a post-test release record such as release_v1.1.json",
    )
    ap.add_argument(
        "--only", type=int, nargs="+", metavar="N",
        help="regenerate only these suggested questions (1-based); keep the other answers",
    )  # fmt: skip
    args = ap.parse_args()
    frozen = check_frozen(args.release)
    version = frozen.get("version", "v1.0")
    provider = OllamaProvider(frozen["model"], think=frozen.get("think"))
    agent = Agent(Toolbox(io.data_dir() / "store"), provider)
    ANSWERS.mkdir(exist_ok=True)
    today = time.strftime("%Y-%m-%d")
    try:
        for game in args.game:
            match = f"metrica/{game}"
            out = ANSWERS / f"metrica-{game}.json"
            old = json.loads(out.read_text()) if out.exists() else {"items": []}
            if args.only and len(old["items"]) != len(SUGGESTED):
                sys.exit(f"{out.name} does not hold every suggested question; regenerate all")
            items = []
            for n, q in enumerate(SUGGESTED, 1):
                if args.only and n not in args.only:
                    kept = old["items"][n - 1]
                    base = {
                        "agent_version": old.get("agent_version"),
                        "generated": old.get("generated"),
                    }
                    items.append({**base, **kept})
                    continue
                a = agent.answer(q, match)
                items.append(
                    {
                        "question": q,
                        "answer_text": a.answer_text,
                        "status": a.status,
                        "citations": a.citations,
                        "caveats": a.caveats,
                        "agent_version": version,
                        "generated": today,
                    }
                )
                print(f"{match} [{a.status}] {q}")
            versions = {it["agent_version"] for it in items}
            out.write_text(
                json.dumps(
                    {
                        "model": frozen["model"],
                        "agent_version": versions.pop() if len(versions) == 1 else "mixed",
                        "generated": max(it["generated"] for it in items),
                        "items": items,
                    },
                    indent=1,
                )
                + "\n"
            )
            print(f"wrote {out}")
    finally:
        provider.unload()


if __name__ == "__main__":
    main()
