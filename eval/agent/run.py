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
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from _common import SKILLCORNER_MATCHES, TEST_GAME, TRAIN_GAMES  # noqa: E402
from premise_judge import load as load_premise_judgments  # noqa: E402
from premise_judge import verdict as premise_verdict  # noqa: E402
from questions import PARAPHRASES, generate  # noqa: E402
from scoring import score  # noqa: E402

from regista import io  # noqa: E402
from regista.agent.grounding import _CLOCK, _FORMATION, _ID_LIKE, _NUMBER, normalise  # noqa: E402
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


AGENT_CODE = [
    HERE.parent.parent / "src" / "regista" / "agent" / f
    for f in ("loop.py", "tools.py", "grounding.py")
]


QUESTION_BANK = [HERE / "questions.py"]
SCORER_CODE = [HERE / "scoring.py", HERE / "premise_judge.py"]


def _files_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def code_hash() -> str:
    """Hash of the agent loop, tools, and grounding check the frozen results depend on."""
    return _files_hash(AGENT_CODE)


def bank_hash() -> str:
    """Hash of the question templates, gold definitions, paraphrases and false premises."""
    return _files_hash(QUESTION_BANK)


def scorer_hash() -> str:
    """Hash of the rule scorer and the false-premise judge (prompt, rubric, model)."""
    return _files_hash(SCORER_CODE)


def current_hashes() -> dict[str, str]:
    return {
        "prompt_hash": prompt_hash(),
        "code_hash": code_hash(),
        "question_bank_hash": bank_hash(),
        "scorer_hash": scorer_hash(),
    }


def check_frozen() -> dict:
    """The frozen configuration, or exit if anything it pins has changed since."""
    if not FROZEN.exists():
        sys.exit("no eval/agent/frozen.json: freeze the model and prompt on dev first")
    frozen = json.loads(FROZEN.read_text())
    changed = [k for k, v in current_hashes().items() if frozen.get(k) != v]
    if changed:
        sys.exit(f"changed since the freeze: {', '.join(changed)}; the test split is locked")
    return frozen


MIN_FREE_GB = 5.0


def preflight() -> None:
    """Check everything the test run needs, without reading or running any test item.

    Verifies the frozen hashes, that the agent and judge models are installed, free
    disk, that the output paths are writable, that every test match has a store
    (manifest only), and answers one dev question end to end. Exits non-zero on
    any failure.
    """
    import shutil
    import tempfile

    import ollama
    from premise_judge import JUDGE_MODEL

    checks: list[tuple[str, bool, str]] = []
    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    checks.append(("frozen.json exists", bool(frozen), str(FROZEN)))
    for k, v in current_hashes().items():
        checks.append((f"{k} matches frozen", frozen.get(k) == v, f"{frozen.get(k)} vs {v}"))
    model = frozen.get("model", "")
    installed = {m.model for m in ollama.Client().list().models}
    for name in (model, JUDGE_MODEL):
        checks.append((f"model {name} installed", name in installed, ""))
    free = shutil.disk_usage(RESULTS).free / 1e9
    checks.append(("free disk", free >= MIN_FREE_GB, f"{free:.1f} GB (need {MIN_FREE_GB})"))
    for d in (RESULTS, REPORT.parent):
        try:
            with tempfile.NamedTemporaryFile(dir=d):
                ok = True
        except OSError:
            ok = False
        checks.append((f"writable {d.relative_to(HERE.parent.parent)}", ok, ""))
    out = results_path("test", model) if model else None
    if out and out.exists():
        n = sum(1 for _ in out.open())
        checks.append(("test results file", True, f"exists with {n} lines; the run resumes"))
    toolbox = Toolbox(io.data_dir() / "store")
    missing = sorted(set(SPLITS["test"]) - set(toolbox.matches()))  # manifests only
    checks.append(("test match stores present", not missing, ", ".join(missing)))
    if model and all(ok for _, ok, _ in checks):
        q = generate(toolbox, SPLITS["dev"][0])[0].to_dict()
        provider = OllamaProvider(model, think=frozen.get("think"))
        try:
            a = Agent(toolbox, provider).answer(q["question"], q["match"]).model_dump()
            sc = score(q, a, toolbox)
            checks.append(
                (
                    "one dev question end to end",
                    True,
                    f"{q['qid']}: {a['status']}, correct={sc['correct']}, {a['latency_s']:.1f}s",
                )
            )
        except Exception as e:  # report it as a failed check, then exit non-zero
            checks.append(("one dev question end to end", False, repr(e)))
        finally:
            provider.unload()
    else:
        checks.append(("one dev question end to end", False, "skipped: earlier checks failed"))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not all(ok for _, ok, _ in checks):
        sys.exit("preflight failed")
    print("preflight passed")


def freeze(model: str, think) -> None:
    """Pin the model and every hash, with the dev summary of that model's results."""
    path = results_path("dev", model)
    df = load(path)
    missing = df[(df["q_category"] == "false_premise") & (df["premise_scorer"] != "judge")]
    if len(missing):
        sys.exit(f"{len(missing)} false-premise dev items lack judge verdicts; run premise_judge")
    stale = {r["meta"]["code_hash"] for r in map(json.loads, path.read_text().splitlines())}
    if stale != {code_hash()}:
        sys.exit(f"{path.name} was produced by agent code {stale}, not {code_hash()}; re-run dev")
    s = summarize(df)
    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {"dev": {}}
    key = model.replace(":", "_").replace("/", "_")
    frozen.update(
        {
            "model": model,
            "think": think,
            **current_hashes(),
            "frozen_at": time.strftime("%Y-%m-%d"),
            "dev_question_count": s["n"],
        }
    )
    frozen["dev"][key] = {
        "overall_accuracy": round(s["overall_accuracy"], 3),
        "per_category": s["per_category"]["accuracy"].round(3).to_dict(),
        "latency_p50": round(s["latency_p50"], 2),
        "latency_p95": round(s["latency_p95"], 2),
        "number_grounding": round(s["number_grounding"], 3),
        "citation_validity": round(s["citation_validity"], 3),
        "false_premise_full_correction": round(s["false_premise_full_correction"], 3),
    }
    FROZEN.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"froze {model} in {FROZEN}")


def results_path(split: str, model: str) -> Path:
    return RESULTS / f"{split}_{model.replace(':', '_').replace('/', '_')}.jsonl"


def run(split: str, model: str, think, limit: int | None = None) -> Path:
    """Answer every question of a split, appending results as they complete.

    If the results file already has some items (e.g. after a crash), those are
    kept and never regenerated; only the remaining questions are asked.
    """
    toolbox = Toolbox(io.data_dir() / "store")
    out = results_path(split, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    questions = [q.to_dict() for m in SPLITS[split] for q in generate(toolbox, m)]
    if limit:
        questions = questions[:limit]
    done = set()
    if out.exists():
        done = {json.loads(line)["question"]["qid"] for line in out.read_text().splitlines()}
    todo = [q for q in questions if q["qid"] not in done]
    if not todo:
        sys.exit(f"{out} already holds every question of this split; refusing to regenerate")
    print(f"{len(done)} done, {len(todo)} to go")
    provider = OllamaProvider(model, think=think)
    agent = Agent(toolbox, provider)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with out.open("a") as fh:
            for i, q in enumerate(todo):
                a = agent.answer(q["question"], q["match"]).model_dump()
                s = score(q, a, toolbox)
                meta = {
                    "split": split,
                    "model": model,
                    "think": think,
                    **current_hashes(),
                    "started": started,
                }
                fh.write(
                    json.dumps({"question": q, "answer": a, "score": s, "meta": meta}, default=str)
                    + "\n"
                )
                fh.flush()
                mark = "OK " if s["correct"] else "BAD"
                print(
                    f"[{len(done) + i + 1}/{len(questions)}] {q['category']:<13} {mark} "
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


_VALUE_TOKENS = [_CLOCK, _FORMATION, _ID_LIKE, _NUMBER]


# a reported absence ("the tools show no detected press change moment") is a finding
_ABSENCE = re.compile(
    r"\bno (?:detected |such )?[\w-]+(?: [\w-]+){0,2} moments?\b|did not detect", re.I
)


def _states_a_value(answer: str, question: str) -> bool:
    """True if the answer gives a finding: a number, clock, formation label, or id not in the
    question, or a reported absence of detected moments. (The absence rule was added after
    reading test answers; the report says so.)"""
    answer, question = normalise(answer), normalise(question)
    if _ABSENCE.search(answer):
        return True
    for pattern in _VALUE_TOKENS:
        for m in pattern.finditer(answer):
            if m.group(0).strip() not in question:
                return True
    return False


def load(path: Path) -> pd.DataFrame:
    """One row per answer. False-premise items are scored by the LLM premise judge
    (primary) when its verdicts exist; ``premise_scorer`` records which scorer set
    ``correct``, and ``correct_pattern`` keeps the pattern rule's verdict."""
    judged = load_premise_judgments(path)
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        s = dict(r["score"])
        s["correct_pattern"] = s["correct"]
        s["premise_scorer"] = None
        if r["question"]["category"] == "false_premise":
            v = judged.get(r["question"]["qid"])
            s["premise_scorer"] = "judge" if v else "pattern"
            if v:
                s["premise_rejected_judge"] = premise_verdict(v)
                s["premise_quote_valid"] = v["quote_valid"] if v["rejects_premise"] else None
                s["correct"] = s["premise_rejected_judge"] and r["answer"]["status"] == "verified"
        rows.append(
            {
                **{f"q_{k}": v for k, v in r["question"].items()},
                **s,
                "base_category": r["question"]
                .get("extra", {})
                .get("base_category", r["question"]["category"]),
                # strict decline: the decline pattern matches and no substantive answer is
                # given (no number, clock, formation, or id beyond the question's own)
                "declined_strict": s["abstained"]
                and not _states_a_value(r["answer"]["answer_text"], r["question"]["question"]),
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
    # split on the base category, so reworded unanswerable questions (xg paraphrases)
    # count as unanswerable
    answerable = df[df["base_category"] != "unanswerable"]
    unanswerable = df[df["base_category"] == "unanswerable"]
    per_cat = df.groupby(report_category(df)).agg(
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
        "false_abstention_strict": float(answerable["declined_strict"].mean()),
        "latency_p50": float(df["latency_s"].quantile(0.5)),
        "latency_p95": float(df["latency_s"].quantile(0.95)),
        "retried": float(df["retried"].mean()),
        "false_premise_full_correction": (
            float(
                df.loc[df["q_category"] == "false_premise", "full_correction"].astype(bool).mean()
            )
            if (df["q_category"] == "false_premise").any()
            else np.nan
        ),
        "n": len(df),
    }


# templates whose fact to name is a team: an answer that names either team, or a
# generic denial, looks alike to the pattern rule, so its agreement says little here
TEAM_NAME_TEMPLATES = ("fp_wrong_team", "fp_press_harder", "fp_higher_line")


def premise_agreement(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> dict | None:
    """Judge (primary) vs pattern (secondary) on false-premise items: raw agreement and
    Cohen's kappa, on premise rejection and on the final correct verdict."""
    fp = df[
        (df["q_category"] == "false_premise")
        & (df["premise_scorer"] == "judge")
        & ~df["q_template"].isin(exclude)
    ]
    if fp.empty:
        return None

    def kappa(a: pd.Series, b: pd.Series) -> float:
        a, b = a.astype(bool), b.astype(bool)
        po = float((a == b).mean())
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return float("nan") if pe == 1 else (po - pe) / (1 - pe)

    claimed = fp["premise_quote_valid"].dropna()
    out = {
        "items": len(fp),
        "quotes_claimed": len(claimed),
        "quotes_invalid": int((~claimed.astype(bool)).sum()),
    }
    for name, j, p in (
        ("rejection", fp["premise_rejected_judge"], fp["premise_rejected_pattern"]),
        ("correct", fp["correct"], fp["correct_pattern"]),
    ):
        out[f"{name}_agreement"] = float((j.astype(bool) == p.astype(bool)).mean())
        out[f"{name}_kappa"] = kappa(j, p)
        out[f"{name}_judge_rate"] = float(j.astype(bool).mean())
        out[f"{name}_pattern_rate"] = float(p.astype(bool).mean())
    return out


def worst(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Failures ranked: wrong and ungrounded first, then wrong, then slowest."""
    bad = df[~df["correct"]].copy()
    bad["rank"] = (~bad["grounded"]).astype(int) * 2 + (~bad["tool_selection"]).astype(int)
    bad = bad.sort_values(["rank", "latency_s"], ascending=[False, False]).head(n)
    return bad[
        ["model", "q_match", "q_category", "q_question", "q_gold", "tools", "status", "answer_text"]
    ]


BOOTSTRAP = 2000


def report_category(df: pd.DataFrame) -> pd.Series:
    """Category for tables: rewordings are split into answerable and unanswerable (xg), so
    correct refusals do not inflate paraphrase accuracy."""
    para = df["q_category"] == "paraphrase"
    kind = np.where(df["base_category"] == "unanswerable", "unanswerable", "answerable")
    return df["q_category"].where(~para, "paraphrase (" + pd.Series(kind, index=df.index) + ")")


def category_ci(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Accuracy per category with a 95% CI from resampling matches (not questions)."""
    rng = np.random.default_rng(seed)
    matches = sorted(df["q_match"].unique())
    rows = []
    for cat, g in df.groupby(report_category(df)):
        per = g.groupby("q_match")["correct"].agg(["sum", "size"]).reindex(matches, fill_value=0)
        a, n = per["sum"].to_numpy(float), per["size"].to_numpy(float)
        idx = rng.integers(0, len(matches), size=(BOOTSTRAP, len(matches)))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = a[idx].sum(axis=1) / n[idx].sum(axis=1)
        boot = boot[~np.isnan(boot)]
        rows.append(
            {
                "category": cat,
                "questions": int(n.sum()),
                "matches": int((n > 0).sum()),
                "accuracy": a.sum() / n.sum(),
                "ci_low": float(np.percentile(boot, 2.5)) if len(boot) else np.nan,
                "ci_high": float(np.percentile(boot, 97.5)) if len(boot) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("category")


def paraphrase_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Per template: accuracy of the original wording and of each rewording."""
    base = df[df["q_category"] != "paraphrase"].groupby("q_template")["correct"].mean()
    para = df[df["q_category"] == "paraphrase"]
    if para.empty:
        return pd.DataFrame()
    variant = para["q_extra"].map(lambda e: f"rewording {e['variant'] + 1}")
    table = para.assign(variant=variant).pivot_table(
        index="q_template", columns="variant", values="correct", aggfunc="mean"
    )
    table.insert(0, "original", base.reindex(table.index))
    table["spread"] = table.max(axis=1) - table.min(axis=1)
    return table


# where a paraphrase came from, for the weak-wording notes (variant index -> note)
PARAPHRASE_NOTES = {
    ("top_pass_pair", 0): "the original ambiguous wording, kept on purpose in question review",
    ("top_pass_pair", 1): "added in question review (undirected wording)",
    ("top_pass_pair", 2): "added in question review (undirected wording)",
}
WEAK_WORDING = 0.7


# low judge scores whose rationale was checked by hand against the fact sheet
REVIEWED_JUDGE_NOTES = {
    ("test", "skillcorner/2017461"): (
        "the stated reason is false: the judge says the summary claims 10 moments where "
        "the fact sheet lists 9, but the fact sheet lists 10. The low score is still "
        'deserved: the summary says the away team "sat deeper at 35.13 m" than the home '
        "team at 34.07 m (35.13 m is the higher line) and calls the 72:00 press change a "
        '"shift"'
    ),
}


def test_headline(df: pd.DataFrame) -> list[str]:
    """SkillCorner with match-level CIs as the headline; Metrica game 3 as one match."""
    lines = ["## Test results", ""]
    src = df["q_match"].str.split("/").str[0]
    sc = df[src == "skillcorner"]
    if len(sc):
        s = summarize(sc)
        lines += [
            f"### Headline: SkillCorner ({sc['q_match'].nunique()} matches, {len(sc)} questions)",
            "",
            "Accuracy per category with a 95% CI from resampling matches (not questions).",
            "",
            category_ci(sc).round(3).to_markdown(),
            "",
            f"Overall {s['overall_accuracy']:.3f}; matched expected tools "
            f"{s['tool_selection']:.3f}; citation validity {s['citation_validity']:.3f}; "
            f"number grounding {s['number_grounding']:.3f}; abstention accuracy "
            f"{s['abstention_accuracy']:.3f}; declined answerable questions: "
            f"{s['false_abstention']:.3f} by the decline pattern, "
            f"{s['false_abstention_strict']:.3f} with no substantive answer; latency p50 "
            f"{s['latency_p50']:.1f} s, p95 {s['latency_p95']:.1f} s.",
            "",
        ]
    m3 = df[src == "metrica"]
    if len(m3):
        per = m3.groupby(report_category(m3))["correct"].agg(questions="size", accuracy="mean")
        lines += [
            f"### Metrica game 3: a single-match observation ({len(m3)} questions)",
            "",
            "One match, so no confidence interval is given; read it as one observation.",
            "",
            per.round(3).to_markdown(),
            "",
        ]
        a = premise_agreement(m3)
        if a:
            lines += [
                f"On its {a['items']} false-premise items the judge scored "
                f"{a['correct_judge_rate']:.2f} and the pattern rule "
                f"{a['correct_pattern_rate']:.2f}; these items are hand-labelled as an extra "
                "stratum in `eval/agent/label.py`.",
                "",
            ]
    spread = paraphrase_spread(df)
    if not spread.empty:
        weak = [
            (t, int(c.split()[-1]) - 1, v)
            for t, row in spread.iterrows()
            for c, v in row.items()
            if c.startswith("rewording") and v < WEAK_WORDING
        ]
        if weak:
            lines += [f"### Weakest test paraphrases (accuracy below {WEAK_WORDING})", ""]
            for t, k, v in sorted(weak, key=lambda x: x[2]):
                note = PARAPHRASE_NOTES.get((t, k), "written before question review")
                lines.append(f'- `{t}` rewording {k + 1} ({v:.2f}): "{PARAPHRASES[t][k]}"; {note}.')
            lines.append("")
    return lines


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
        '"Matched expected tools" is the share of answers whose tools include one of the '
        "listed tool sets for the question; the lists are not every valid path, so it is "
        "a lower bound on sensible tool use, not a tool-selection accuracy.",
        "",
    ]
    for tf in sorted(RESULTS.glob("test_*.jsonl")):
        lines += test_headline(load(tf))
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
                "matched expected tools": s["tool_selection"],
                "citation validity": s["citation_validity"],
                "number grounding": s["number_grounding"],
                "abstention accuracy": s["abstention_accuracy"],
                "declined (pattern)": s["false_abstention"],
                "declined (no substantive answer)": s["false_abstention_strict"],
                "latency p50 (s)": s["latency_p50"],
                "latency p95 (s)": s["latency_p95"],
                "false premise full correction": s["false_premise_full_correction"],
                "false premise scorer": ", ".join(sorted(df["premise_scorer"].dropna().unique())),
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
        if df["q_match"].nunique() > 2:
            for source, g in df.groupby(df["q_match"].str.split("/").str[0]):
                lines += [
                    f"### {source} ({g['q_match'].nunique()} matches), match-level "
                    f"bootstrap 95% CI",
                    "",
                    category_ci(g).round(3).to_markdown(),
                    "",
                ]
        agree = premise_agreement(df)
        if agree:
            lines += [
                "### False premise: LLM judge (primary) vs pattern rule (secondary)",
                "",
                f"{agree['items']} items. The judge claimed a rejection in "
                f"{agree['quotes_claimed']}; {agree['quotes_invalid']} of its quotes were not "
                "found in the answer and count as not rejected. Premise rejected: judge "
                f"{agree['rejection_judge_rate']:.2f}, "
                f"pattern {agree['rejection_pattern_rate']:.2f}; "
                f"agreement {agree['rejection_agreement']:.2f}, Cohen's kappa "
                f"{agree['rejection_kappa']:.2f}. Correct (rejection and grounded): judge "
                f"{agree['correct_judge_rate']:.2f}, pattern {agree['correct_pattern_rate']:.2f}; "
                f"agreement {agree['correct_agreement']:.2f}, kappa {agree['correct_kappa']:.2f}.",
                "",
                "The pattern rule's agreement is uninformative for templates whose fact to "
                f"name is a team ({', '.join(TEAM_NAME_TEMPLATES)}): a team name appears in "
                "almost every answer.",
            ]
            rest = premise_agreement(df, TEAM_NAME_TEMPLATES)
            if rest:
                lines += [
                    f"Without them ({rest['items']} items): correct agreement "
                    f"{rest['correct_agreement']:.2f}, kappa {rest['correct_kappa']:.2f}.",
                ]
            lines += [""]
        spread = paraphrase_spread(df)
        if not spread.empty:
            lines += [
                "### Paraphrase robustness (accuracy per wording)",
                "",
                spread.round(2).to_markdown(),
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
        low = point[point[["faithful", "coverage", "caveats"]].min(axis=1) < 4]
        if len(low):
            lines += ["Scores below 4:", ""]
            for r in low.itertuples():
                note = REVIEWED_JUDGE_NOTES.get((split, r.match), "rationale not reviewed")
                lines.append(
                    f"- {r.match}, {r.model} (faithful {r.faithful}, coverage {r.coverage}, "
                    f"caveats {r.caveats}): {note}."
                )
            lines += [
                "",
                "Judge rationales are not reliable evidence on their own; a score is "
                "checked against the fact sheet before it is cited.",
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
    lines += [
        "## Limitations",
        "",
        "- Gold answers come from the same tools the agent calls, so this evaluation "
        "measures faithfulness to the tools (right tool, right reading, grounded numbers "
        "and times), not whether the tools are right; tool correctness is covered by "
        "the Phase 1 evaluations (`eval/reports/phase1.md`).",
        "- Grounding verifies values, not their meaning: a number is grounded if a tool "
        "returned it, even when the answer describes it wrongly (e.g. back-line counts, "
        "5 -> 3 defenders, described as line heights in metres).",
        "- Declines are reported two ways: the decline pattern (which also matches premise "
        'corrections and caveats such as "Regista does not model the reasons"), and '
        '"no substantive answer" (the pattern matches and the answer gives no number, '
        "clock, formation, or id beyond the question's, and reports no absence of detected "
        "moments). The absence rule was added after reading test answers.",
        "- The false-premise judge is an LLM (gemma4:12b); its quote is verified to be "
        "in the answer, but whether that sentence rejects the premise is its judgment. "
        "Judge-human agreement comes from the 30 hand labels (`eval/agent/label.py`).",
        "",
    ]
    lines += [
        "## Future work",
        "",
        "- A deterministic check of directional claims in answers and summaries (higher / "
        "deeper, more / less) against the tools' comparison fields. Grounding confirms a "
        "value came from a tool, not what the answer says about it; the summary for "
        'skillcorner/2017461 ("sat deeper at 35.13 m" for the higher line) and back-line '
        'counts described as "metres" are both examples. Not part of agent v1.1.',
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
    ap.add_argument("--freeze", action="store_true", help="pin --model and all hashes")
    ap.add_argument("--preflight", action="store_true", help="check readiness for the test run")
    args = ap.parse_args()
    if args.preflight:
        preflight()
        return
    if args.freeze:
        if not args.model:
            sys.exit("--freeze needs --model")
        freeze(args.model, THINK_DEFAULTS.get(args.model))
        return
    if args.rescore:
        for f in sorted(RESULTS.glob("dev_*.jsonl")):
            rescore(f)
            print(f"rescored {f}")
    if args.split:
        model, think = args.model, args.think
        if args.split == "test":
            frozen = check_frozen()
            model, think = frozen["model"], frozen.get("think")
            # an incomplete test file resumes; a complete one is refused inside run()
        if model is None:
            sys.exit("--model is required for the dev split")
        if isinstance(think, str) and think.lower() in ("true", "false"):
            think = think.lower() == "true"
        think = think if think is not None else THINK_DEFAULTS.get(model)
        print(f"wrote {run(args.split, model, think, args.limit)}")
    print(f"wrote {write_report()}")


if __name__ == "__main__":
    main()
