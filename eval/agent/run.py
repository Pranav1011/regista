"""Agent evaluation: template questions with store-computed answers, scored automatically.

Dev split: Metrica games 1-2 (prompts, templates, and model choice are developed
here). Test split: Metrica game 3 + every SkillCorner match, run ONCE with the
prompt and model frozen in eval/agent/frozen.json; a second test run is refused.

  uv run python eval/agent/run.py --split dev --model llama3.1:8b
  uv run python eval/agent/run.py --split test          # uses the frozen model
  uv run python eval/agent/run.py --report              # regenerate the report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from _common import SKILLCORNER_MATCHES, TEST_GAME, TRAIN_GAMES  # noqa: E402
from questions import generate, score  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.loop import SYSTEM_PROMPT, Agent, OllamaProvider  # noqa: E402
from regista.agent.tools import Toolbox  # noqa: E402

RESULTS = HERE / "results"
FROZEN = HERE / "frozen.json"
REPORT = HERE.parent / "reports" / "phase2_agent.md"
SPLITS = {
    "dev": [f"metrica/{g}" for g in TRAIN_GAMES],
    "test": [f"metrica/{TEST_GAME}"] + [f"skillcorner/{m}" for m in SKILLCORNER_MATCHES],
}
THINK_DEFAULTS = {
    "qwen3.5:9b": False,
    "gemma4:12b": False,
    "gpt-oss:20b": "low",
}  # gpt-oss always reasons


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def results_path(split: str, model: str) -> Path:
    return RESULTS / f"{split}_{model.replace(':', '_').replace('/', '_')}.jsonl"


def run(split: str, model: str, think, limit: int | None = None) -> Path:
    toolbox = Toolbox(io.data_dir() / "store")
    provider = OllamaProvider(model, think=think)
    agent = Agent(toolbox, provider)
    out = results_path(split, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    questions = [q.to_dict() for m in SPLITS[split] for q in generate(toolbox, m)]
    if limit:
        questions = questions[:limit]
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with out.open("w") as fh:
            for i, q in enumerate(questions):
                a = agent.answer(q["question"], q["match"]).model_dump()
                s = score(q, a, toolbox)
                fh.write(
                    json.dumps(
                        {
                            "question": q,
                            "answer": a,
                            "score": s,
                            "meta": {
                                "split": split,
                                "model": model,
                                "think": think,
                                "prompt_hash": prompt_hash(),
                                "started": started,
                            },
                        },
                        default=str,
                    )
                    + "\n"
                )
                fh.flush()
                mark = "OK " if s["correct"] else "BAD"
                print(
                    f"[{i + 1}/{len(questions)}] {q['category']:<12} {mark} "
                    f"{a['latency_s']:>6.1f}s  {q['question'][:70]}"
                )
    finally:
        provider.unload()
    return out


def rescore(path: Path) -> None:
    """Recompute scores for stored answers (scorer changes only; no model is run)."""
    toolbox = Toolbox(io.data_dir() / "store")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fresh = {
        q.qid: q.to_dict()
        for m in {r["question"]["match"] for r in rows}
        for q in generate(toolbox, m)
    }
    for r in rows:
        new = fresh.get(r["question"]["qid"])
        if new is None or new["question"] != r["question"]["question"]:
            raise RuntimeError(f"question {r['question']['qid']} changed wording; re-run it")
        r["question"] = new  # updated gold definitions, same question text
        r["score"] = score(r["question"], r["answer"], toolbox)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.append(
            {
                **{f"q_{k}": v for k, v in r["question"].items()},
                **r["score"],
                "latency_s": r["answer"]["latency_s"],
                "status": r["answer"]["status"],
                "retried": r["answer"]["retried"],
                "answer_text": r["answer"]["answer_text"],
                "tools": [t["tool"] for t in r["answer"]["tools_used"]],
                "model": r["meta"]["model"],
                "split": r["meta"]["split"],
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    answerable = df[df["q_category"] != "unanswerable"]
    unanswerable = df[df["q_category"] == "unanswerable"]
    per_cat = df.groupby("q_category").agg(
        questions=("correct", "size"),
        accuracy=("correct", "mean"),
        tool_selection=("tool_selection", "mean"),
    )
    return {
        "per_category": per_cat,
        "overall_accuracy": float(df["correct"].mean()),
        "tool_selection": float(answerable["tool_selection"].mean()),
        "citation_validity": float(answerable["citation_valid"].dropna().astype(bool).mean()),
        "number_grounding": float(df["grounded"].mean()),
        "abstention_accuracy": float(unanswerable["correct"].mean())
        if len(unanswerable)
        else np.nan,
        "false_abstention": float(answerable["abstained"].mean()),
        "latency_p50": float(df["latency_s"].quantile(0.5)),
        "latency_p95": float(df["latency_s"].quantile(0.95)),
        "retried": float(df["retried"].mean()),
        "n": len(df),
    }


def worst(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Failures ranked: wrong and ungrounded first, then wrong, then slowest."""
    bad = df[~df["correct"]].copy()
    bad["rank"] = (~bad["grounded"]).astype(int) * 2 + (~bad["tool_selection"]).astype(int)
    bad = bad.sort_values(["rank", "latency_s"], ascending=[False, False]).head(n)
    return bad[
        ["model", "q_match", "q_category", "q_question", "q_gold", "tools", "status", "answer_text"]
    ]


def write_report() -> Path:
    files = sorted([*RESULTS.glob("dev_*.jsonl"), *RESULTS.glob("test_*.jsonl")])
    if not files:
        raise FileNotFoundError("no results yet; run a split first")
    lines = [
        "# Regista Phase 2: agent evaluation",
        "",
        "Generated by `eval/agent/run.py --report` from `eval/agent/results/`. Questions "
        "and gold answers are generated from the match stores; nothing is hand-labelled.",
        "Dev split: Metrica games 1-2. Test split: Metrica game 3 + SkillCorner, run once "
        "with the frozen prompt and model.",
        "",
    ]
    rows = []
    for f in files:
        df = load(f)
        s = summarize(df)
        rows.append(
            {
                "split": df["split"].iat[0],
                "model": df["model"].iat[0],
                "questions": s["n"],
                "accuracy": s["overall_accuracy"],
                "tool selection": s["tool_selection"],
                "citation validity": s["citation_validity"],
                "number grounding": s["number_grounding"],
                "abstention accuracy": s["abstention_accuracy"],
                "false abstention": s["false_abstention"],
                "latency p50 (s)": s["latency_p50"],
                "latency p95 (s)": s["latency_p95"],
            }
        )
    lines += ["## Summary", "", pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
    for f in files:
        df = load(f)
        s = summarize(df)
        lines += [
            f"## {df['split'].iat[0]} / {df['model'].iat[0]}: accuracy per category",
            "",
            s["per_category"].round(3).to_markdown(),
            "",
        ]
    dev_files = [f for f in files if f.name.startswith("dev_")]
    if dev_files:
        allf = pd.concat([load(f) for f in dev_files], ignore_index=True)
        w = worst(allf)
        w = w.assign(
            q_gold=w["q_gold"].map(lambda g: json.dumps(g, default=str)[:80]),
            answer_text=w["answer_text"].str.slice(0, 160).str.replace("\n", " "),
            tools=w["tools"].map(", ".join),
        )
        lines += [
            "## Ten worst dev failures (all models)",
            "",
            "Ranked: ungrounded first, then wrong tool, then slowest.",
            "",
            w.to_markdown(index=False),
            "",
        ]
    for jf in sorted(RESULTS.glob("judge_*.jsonl")):
        rows = [json.loads(line) for line in jf.read_text().splitlines()]
        point = pd.DataFrame([r for r in rows if r["kind"] == "pointwise"])
        pair = [r for r in rows if r["kind"] == "pairwise"]
        split = jf.stem.split("_")[1]
        lines += [
            f"## Summaries judged by {rows[0]['judge']} ({split})",
            "",
            "Rubric scores 1-5 (faithful to the fact sheet, covers flagged moments, "
            "states caveats); the judge is from a different model family than the agent.",
            "",
            point.groupby("model")[["faithful", "coverage", "caveats"]]
            .mean()
            .round(2)
            .to_markdown(),
            "",
        ]
        if pair:
            agree = sum(r["order_agree"] for r in pair) / len(pair)
            wins: dict[str, int] = {}
            for r in pair:
                if r["order_agree"]:
                    winner = next(iter(set(r["verdicts"].values())))
                    wins[winner] = wins.get(winner, 0) + 1
            lines += [
                f"Pairwise, both orders: {len(pair)} pairs; the verdict was the same in "
                f"both orders for {agree:.2f} of them (1 - position bias). Consistent wins: "
                + (", ".join(f"{k} {v}" for k, v in sorted(wins.items())) or "none")
                + ".",
                "",
            ]
    labels = HERE / "human_labels.jsonl"
    if labels.exists():
        lines += [
            "## Judge-human agreement",
            "",
            "See `eval/agent/label.py`; computed once the 30 human labels exist.",
            "",
        ]
    if FROZEN.exists():
        lines += ["## Frozen configuration", "", "```json", FROZEN.read_text().strip(), "```", ""]
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=list(SPLITS))
    ap.add_argument("--model")
    ap.add_argument("--think", default=None, help="true, false, low, medium or high")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="re-score stored dev answers")
    args = ap.parse_args()
    if args.rescore:
        for f in sorted(RESULTS.glob("dev_*.jsonl")):
            rescore(f)
            print(f"rescored {f}")
    if args.split:
        model, think = args.model, args.think
        if args.split == "test":
            if not FROZEN.exists():
                sys.exit("no eval/agent/frozen.json: freeze the model and prompt on dev first")
            frozen = json.loads(FROZEN.read_text())
            if frozen["prompt_hash"] != prompt_hash():
                sys.exit("the system prompt changed since it was frozen; the test split is locked")
            model, think = frozen["model"], frozen.get("think")
            if results_path("test", model).exists():
                sys.exit("the test split has already been run once; refusing to run it again")
        if model is None:
            sys.exit("--model is required for the dev split")
        if isinstance(think, str) and think.lower() in ("true", "false"):
            think = think.lower() == "true"
        think = think if think is not None else THINK_DEFAULTS.get(model)
        print(f"wrote {run(args.split, model, think, args.limit)}")
    print(f"wrote {write_report()}")


if __name__ == "__main__":
    main()
