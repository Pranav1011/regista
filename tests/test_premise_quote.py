"""The premise judge's quote must be verbatim in the answer (synthetic strings)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agent"))

from premise_judge import quote_valid, verdict  # noqa: E402

ANSWER = "The home team did not drop deep during extra time: no line‑height shift was detected."


def test_quote_must_be_a_substring_after_normalisation():
    assert quote_valid("the home team did not drop deep during extra time", ANSWER)
    assert quote_valid("  No line-height shift was detected. ", ANSWER)  # dash, case, space
    assert not quote_valid("There was no extra time.", ANSWER)  # paraphrase, not a quote
    assert not quote_valid("no", ANSWER)  # too short to be a sentence


def test_no_valid_quote_means_not_rejected():
    base = {"rejects_premise": True, "with_evidence": True}
    assert verdict({**base, "quote_valid": True})
    assert not verdict({**base, "quote_valid": False})
