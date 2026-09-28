"""LLM judge for false-premise answers (primary scorer; the pattern rule is secondary).

For every false-premise item in a results file, gemma4:12b (a different family
from the agent) reads the question, what the data shows, the answer, and the
evidence ranges the answer cites, and decides whether the answer rejects the
premise, explicitly or implicitly, with evidence. The judge must quote the answer
sentence that rejects the premise; the quote is checked deterministically to be a
substring of the answer (after normalisation), and without a valid quote the
premise counts as not rejected. Run after the agent has
finished and been unloaded; the judge is never loaded alongside it.

  uv run python eval/agent/premise_judge.py --results eval/agent/results/dev_qwen3.5_9b.jsonl
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

from regista.agent.grounding import normalise  # noqa: E402
from regista.agent.loop import OllamaProvider  # noqa: E402

JUDGE_MODEL = "gemma4:12b"
RUBRIC = """You grade answers to football questions that contain a FALSE PREMISE.
You are told what the data actually shows. Decide:
- rejects_premise: does the answer reject the false premise, either explicitly (says it
  is wrong or not supported) or implicitly (states facts that contradict it)? An answer
  that accepts the premise and explains it, or only says it cannot explain "why", does
  NOT reject it.
- with_evidence: does the answer support its rejection with evidence from the data
  (values, match times, the detected moments, or a statement that the tools searched
  the match or a stated time range and found no such moment), rather than a bare
  denial? Judge only the answer; do not reward facts that appear only in WHAT THE DATA
  SHOWS.
- quote: if rejects_premise is true, copy the answer sentence that rejects the premise,
  word for word; otherwise an empty string.
Answer only with JSON: {"rejects_premise": true|false, "with_evidence": true|false,
"quote": "...", "reason": "..."}"""
SCHEMA = {
    "type": "object",
    "properties": {
        "rejects_premise": {"type": "boolean"},
        "with_evidence": {"type": "boolean"},
        "quote": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["rejects_premise", "with_evidence", "quote", "reason"],
}


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", normalise(text)).strip().casefold()
    return text.strip(" \"'.,;:")


def quote_valid(quote: str, answer: str) -> bool:
    """The quote is a non-trivial substring of the answer, after normalisation."""
    q = _norm(quote)
    return len(q) >= 10 and q in _norm(answer)


def judgments_path(results: Path) -> Path:
    return results.with_name(f"premise_{results.stem}.jsonl")


def judge(results: Path, judge_model: str = JUDGE_MODEL) -> Path:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    items = [r for r in rows if r["question"]["category"] == "false_premise"]
    out = judgments_path(results)
    done = set()
    if out.exists():
        done = {json.loads(line)["qid"] for line in out.read_text().splitlines()}
    provider = OllamaProvider(judge_model, think=False)
    try:
        with out.open("a") as fh:
            for r in items:
                q, a = r["question"], r["answer"]
                if q["qid"] in done:
                    continue
                cites = [
                    f"period {c['period']} {c['clock_start']}-{c['clock_end']}"
                    for c in a["citations"]
                ]
                user = (
                    f"QUESTION: {q['question']}\nWHAT THE DATA SHOWS: {q['gold']['truth']}\n"
                    f"ANSWER: {a['answer_text']}\nEVIDENCE CITED: {cites or 'none'}"
                )
                reply = provider.client.chat(
                    model=provider.name,
                    format=SCHEMA,
                    options=provider.options,
                    think=False,
                    keep_alive=provider.keep_alive,
                    messages=[
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": user},
                    ],
                )
                v = json.loads(reply.message.content)
                v["quote_valid"] = quote_valid(v["quote"], a["answer_text"])
                fh.write(json.dumps({"qid": q["qid"], "judge": judge_model, **v}) + "\n")
                fh.flush()
                print(
                    f"{q['qid']}: rejects={v['rejects_premise']} evidence={v['with_evidence']} "
                    f"quote_valid={v['quote_valid']}"
                )
    finally:
        provider.unload()
    return out


def verdict(v: dict) -> bool:
    """Primary false-premise verdict: rejected, with evidence, and a verified quote."""
    return bool(v["rejects_premise"] and v["quote_valid"] and v["with_evidence"])


def load(results: Path) -> dict[str, dict]:
    path = judgments_path(results)
    if not path.exists():
        return {}
    return {r["qid"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--judge", default=JUDGE_MODEL)
    args = ap.parse_args()
    print(f"wrote {judge(args.results, args.judge)}")


if __name__ == "__main__":
    main()
