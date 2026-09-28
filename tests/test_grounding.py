"""Tests for the number-grounding check."""

from regista.agent.grounding import check_numbers

TOOL = {
    "press_intensity": 0.4088,
    "line_height_m": 34.2515,
    "label": "4-2-3-1",
    "windows": [{"clock_start": "60:00", "margin": 0.0694}],
    "emit_clock": "45+2:00",
    "passes": 423,
}


def test_rounded_values_percentages_clocks_and_labels_are_grounded():
    answer = (
        "From 60:00 home pressed in 41% of frames (0.41), line height about 34.3 m, "
        "playing 4-2-3-1 with margin 0.069; 423 passes; an alert at 45+2:00."
    )
    r = check_numbers(answer, [TOOL])
    assert r.grounded, r.ungrounded


def test_invented_numbers_are_caught():
    r = check_numbers("Home pressed 55% of the time, in a 4-4-2, at 71:30.", [TOOL])
    assert set(r.ungrounded) == {"55%", "4-4-2", "71:30"}


def test_numbers_from_the_question_count_and_ids_are_ignored():
    r = check_numbers(
        "After 60 minutes home_11 made 423 passes.", [TOOL, "what happened after 60'"]
    )
    assert r.grounded, r.ungrounded


def test_rounding_must_respect_written_precision():
    assert not check_numbers("line height 34.3 m", [{"x": 34.4}]).grounded
    assert check_numbers("line height 34 m", [{"x": 34.4}]).grounded
