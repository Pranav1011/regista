"""Match summaries written by the agent, and LLM-as-judge scoring with a rubric.

- The agent writes one summary per match (same tools, same grounding check).
- A judge model from a different family scores each summary 1-5 on three
  criteria against a fact sheet computed from the store: faithful (every claim
  matches the facts), coverage (mentions the flagged moments), caveats (states
  the relevant limitations). It also compares pairs of summaries for the same
  match in BOTH orders; order agreement measures position bias.
- The agent and judge models are never loaded together: generation finishes and
  unloads before judging starts.

  uv run python eval/agent/summaries.py generate --split dev --model qwen3.5:9b
  uv run python eval/agent/summaries.py judge --split dev --judge gemma4:12b
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import RESULTS, SPLITS  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.loop import Agent, OllamaProvider  # noqa: E402
from regista.agent.tools import Toolbox  # noqa: E402

SUMMARY_QUESTION = (
    "Write a tactical summary of this match in five to eight sentences. Cover: each team's "
    "usual formation with and without the ball, and whether it was clear-cut or a close call; "
    "each team's pressing intensity; each team's defensive line height out of possession; and "
    "every tactical moment from find_moments, each with its team and match clock. End with one "
    "sentence of caveats: formation labels are noisy at 5-minute windows, and passes are "
    "inferred from tracking."
)
CRITERIA = ("faithful", "coverage", "caveats")
RUBRIC = """You are grading a football match summary against a fact sheet computed from data.
Score each criterion from 1 (poor) to 5 (excellent):
- faithful: every claim in the summary agrees with the fact sheet; no invented numbers, \
formations, times, or events. Any contradiction caps this at 2.
- coverage: the summary mentions the detected tactical moments listed in the fact sheet.
- caveats: the summary states relevant limitations (formation labels are noisy or close \
calls; passing values are model estimates; things not modelled).
Answer only with JSON: {"faithful": n, "coverage": n, "caveats": n, "reason": "..."}"""
PAIRWISE = """You are comparing two summaries of the same football match against a fact sheet \
computed from data. Which summary is more faithful to the fact sheet, covers the detected \
moments better, and states limitations better? Answer only with JSON: \
{"winner": "A" or "B", "reason": "..."}"""
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        **{c: {"type": "integer", "minimum": 1, "maximum": 5} for c in CRITERIA},
        "reason": {"type": "string"},
    },
    "required": [*CRITERIA, "reason"],
}
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"],
}


def fact_sheet(toolbox: Toolbox, match: str) -> dict:
    """Deterministic facts the judge grades against (tool outputs, trimmed)."""
    ov = toolbox.get_match_overview(match).model_dump()
    facts = {
        "overview": {
            k: ov[k]
            for k in ("periods", "passes_by_team", "moments_by_type", "most_common_formation")
        }
    }
    facts["moments"] = [
        {k: m[k] for k in ("type", "team", "emit_clock", "before", "after")}
        for m in toolbox.find_moments(match).model_dump()["moments"]
    ]
    for team in ("home", "away"):
        for phase in ("in", "out"):
            f = toolbox.get_formation(match, team, phase)
            labels = [w.label for w in f.windows if w.label]
            facts[f"formation_{team}_{phase}"] = {
                "most_common_label": f.most_common_label,
                "share": f.most_common_share,
                "label_counts": {lab: labels.count(lab) for lab in sorted(set(labels))},
                "close_call_windows": sum(bool(w.close_call) for w in f.windows),
                "windows": len(f.windows),
            }
        facts[f"press_{team}"] = [
            t.model_dump() for t in toolbox.get_press_stats(match, team).by_third
        ]
        facts[f"line_height_out_{team}"] = toolbox.get_team_dimensions(
            match, team, "out"
        ).line_height_m
    return facts


def _path(kind: str, split: str, model: str) -> Path:
    return RESULTS / f"{kind}_{split}_{model.replace(':', '_').replace('/', '_')}.jsonl"


def generate(split: str, model: str, think) -> Path:
    toolbox = Toolbox(io.data_dir() / "store")
    provider = OllamaProvider(model, think=think)
    agent = Agent(toolbox, provider)
    out = _path("summaries", split, model)
    try:
        with out.open("w") as fh:
            for match in SPLITS[split]:
                a = agent.answer(SUMMARY_QUESTION, match).model_dump()
                fh.write(json.dumps({"match": match, "model": model, "summary": a}) + "\n")
                print(f"{match}: {a['status']} {a['latency_s']}s")
    finally:
        provider.unload()
    return out


def _judge_call(provider: OllamaProvider, system: str, user: str, schema: dict) -> dict:
    r = provider.client.chat(
        model=provider.name,
        format=schema,
        options=provider.options,
        think=provider.think,
        keep_alive=provider.keep_alive,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(r.message.content)


def judge(split: str, judge_model: str, think=False) -> Path:
    toolbox = Toolbox(io.data_dir() / "store")
    summaries: dict[str, dict[str, str]] = {}
    for f in sorted(RESULTS.glob(f"summaries_{split}_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            summaries.setdefault(r["match"], {})[r["model"]] = r["summary"]["answer_text"]
    provider = OllamaProvider(judge_model, think=think)
    out = _path("judge", split, judge_model)
    try:
        with out.open("w") as fh:
            for match, by_model in summaries.items():
                facts = json.dumps(fact_sheet(toolbox, match), default=str)
                for model, text in by_model.items():
                    s = _judge_call(
                        provider, RUBRIC, f"FACT SHEET:\n{facts}\n\nSUMMARY:\n{text}", SCORE_SCHEMA
                    )
                    fh.write(
                        json.dumps(
                            {
                                "kind": "pointwise",
                                "match": match,
                                "model": model,
                                "judge": judge_model,
                                **s,
                            }
                        )
                        + "\n"
                    )
                for a, b in itertools.combinations(sorted(by_model), 2):
                    verdicts = {}
                    for first, second in ((a, b), (b, a)):
                        v = _judge_call(
                            provider,
                            PAIRWISE,
                            f"FACT SHEET:\n{facts}\n\nSUMMARY A:\n{by_model[first]}"
                            f"\n\nSUMMARY B:\n{by_model[second]}",
                            PAIR_SCHEMA,
                        )
                        verdicts[f"{first}|{second}"] = first if v["winner"] == "A" else second
                    fh.write(
                        json.dumps(
                            {
                                "kind": "pairwise",
                                "match": match,
                                "judge": judge_model,
                                "pair": [a, b],
                                "verdicts": verdicts,
                                "order_agree": len(set(verdicts.values())) == 1,
                            }
                        )
                        + "\n"
                    )
                print(f"judged {match}")
    finally:
        provider.unload()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["generate", "judge"])
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--model")
    ap.add_argument("--judge")
    ap.add_argument("--think", default=None)
    args = ap.parse_args()
    think = {"true": True, "false": False}.get(str(args.think).lower(), args.think)
    if args.action == "generate":
        print(f"wrote {generate(args.split, args.model, think)}")
    else:
        print(f"wrote {judge(args.split, args.judge, think if think is not None else False)}")


if __name__ == "__main__":
    main()
