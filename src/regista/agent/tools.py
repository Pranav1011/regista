"""Read-only, typed tools over match stores. The same functions back the MCP server,
the local agent loop, and the agent evals.

Every result carries ``evidence`` (match, period, frame range, match clock) and,
where relevant, reliability signals. Times are match clocks ("62:00",
"45+2:00"); ranges crossing half-time are split per period. Times outside the
recorded match raise ``ToolError``. "45:00" means the start of the second
half; first-half stoppage time is written "45+m:ss". Tools never estimate anything the store does
not contain; unsupported topics return an explicit "not available" result.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from regista.clock import PERIOD_LENGTH_S, match_clock, parse_clock
from regista.store import Store

Team = Literal["home", "away"]
Phase = Literal["in", "out"]
Third = Literal["all", "defensive", "middle", "attacking"]
MomentType = Literal["back_line_change", "press_change", "line_height_shift"]

FORMATION_NOTE = (
    "Exact template labels are noisy at 5-minute windows; position groups and the "
    "back-line count are more reliable. margin and relative_margin measure how much "
    "better the winning template fits than the runner-up; they are not probabilities."
)
PROBABILITY_NOTE = (
    "display_probability is a model estimate: raw pitch control calibrated on Metrica "
    "games 1-2. Options for teammates who were not passed to cannot be validated."
)
ANONYMISED_NOTE = "Players are anonymised ids; names are not available."
UNSUPPORTED = {
    "xg": "Expected goals (xG) is not modelled by Regista.",
    "expected goals": "Expected goals (xG) is not modelled by Regista.",
    "shot quality": "Shot quality is not modelled by Regista.",
    "player name": "Player names are not available: the tracking data is anonymised.",
    "names": "Player names are not available: the tracking data is anonymised.",
    "injur": "Injuries are not in the data.",
    "referee": "Referee decisions are not in the data.",
    "weather": "Weather is not in the data.",
    "odds": "Betting odds are not in the data.",
    "score": "The score is not modelled by Regista.",
    "goal": "Goals and the score are not modelled by Regista.",
}
SUPPORTED = (
    "formations and roles (in and out of possession), team shape (line height, "
    "length, width, compactness), pressing intensity, pass counts and pass networks, "
    "passing options at pass moments (model estimates), and detected tactical moments "
    "(back-line changes, press changes, line-height shifts)"
)


CLOCK_TOLERANCE_S = 1.0  # a clock may sit up to 1 s outside the first/last recorded frame


def moment_type(text: str) -> str:
    """Map a moment type or a plain description of one onto a detector name."""
    t = text.lower().replace("-", " ").replace("_", " ")
    if "press" in t:
        return "press_change"
    if "height" in t or ("line" in t and ("high" in t or "deep" in t or "shift" in t)):
        return "line_height_shift"
    if "back" in t or "defender" in t or "line" in t or "formation" in t:
        return "back_line_change"
    raise ToolError(
        f"unknown moment type {text!r}; use back_line_change, press_change, or line_height_shift"
    )


class ToolError(ValueError):
    """Invalid tool input (e.g. a time outside the match); the message is user-facing."""


class Evidence(BaseModel):
    match: str
    period: int
    frame_start: int
    frame_end: int
    clock_start: str
    clock_end: str


class NotAvailable(BaseModel):
    available: Literal[False] = False
    topic: str
    reason: str
    supported: str = SUPPORTED


class Capability(BaseModel):
    available: bool
    topic: str
    reason: str
    supported: str = SUPPORTED


class FormationWindow(BaseModel):
    period: int
    clock_start: str
    clock_end: str
    label: str | None
    runner_up: str | None
    margin: float | None
    relative_margin: float | None
    back_line: int | None
    close_call: bool | None = Field(description="margin below this match's median margin")


class FormationResult(BaseModel):
    match: str
    team: Team
    phase: Phase
    most_common_label: str | None
    most_common_share: float | None
    back_line_counts: dict[str, int]
    windows: list[FormationWindow]
    match_median_margin: float
    evidence: list[Evidence]
    notes: list[str]


class ShapeResult(BaseModel):
    match: str
    team: Team
    phase: Phase
    line_height_m: float | None
    length_m: float | None
    width_m: float | None
    hull_area_m2: float | None
    windows_used: int
    evidence: list[Evidence]
    notes: list[str]


class PressThird(BaseModel):
    third: Third
    carrier_frames: int
    press_intensity: float | None
    tight_intensity: float | None
    mean_defenders_within: float | None


class PressResult(BaseModel):
    match: str
    pressing_team: Team
    by_third: list[PressThird]
    evidence: list[Evidence]
    notes: list[str]


class Edge(BaseModel):
    from_player: str
    to_player: str
    passes: int


class Node(BaseModel):
    player_id: str
    x: float
    y: float
    passes_made: int


class NetworkResult(BaseModel):
    match: str
    team: Team
    total_passes: int
    top_edges: list[Edge]
    nodes: list[Node]
    evidence: list[Evidence]
    notes: list[str]


class PassSummary(BaseModel):
    pass_id: int
    period: int
    clock: str
    to_player: str
    completed: bool


class PlayerPassing(BaseModel):
    player_id: str
    attempted: int
    completed: int
    completion_share: float
    received: int


class TeamPassingResult(BaseModel):
    match: str
    team: Team
    players: list[PlayerPassing]
    evidence: list[Evidence]
    notes: list[str]


class PlayerPassesResult(BaseModel):
    match: str
    player_id: str
    team: Team
    attempted: int
    completed: int
    received: int
    top_receivers: list[Edge]
    passes: list[PassSummary]
    evidence: list[Evidence]
    notes: list[str]


class Option(BaseModel):
    player_id: str
    x: float
    y: float
    pitch_control: float
    display_probability: float
    is_target: bool


class PassingOptionsResult(BaseModel):
    match: str
    pass_id: int
    period: int
    clock: str
    team: Team
    from_player: str
    to_player: str
    completed: bool
    target_display_probability: float
    options: list[Option]
    evidence: list[Evidence]
    notes: list[str]


class MomentItem(BaseModel):
    type: MomentType
    team: Team
    period: int
    emit_clock: str
    start_clock_estimate: str
    severity: float
    before: float
    after: float
    evidence: list[Evidence]


class MomentsResult(BaseModel):
    match: str
    moments: list[MomentItem]
    evidence: list[Evidence]  # the time range searched, so "none found" is citable
    notes: list[str]


class Overview(BaseModel):
    match: str
    evidence: list[Evidence]
    source: str
    periods: list[dict]
    frame_rate: float
    passes_by_team: dict[str, int]
    moments_by_type: dict[str, int]
    most_common_formation: dict[str, dict[str, str | None]]
    credits: str
    notes: list[str]


CREDITS = {
    "metrica": "Tracking and event data: Metrica Sports sample data.",
    "skillcorner": "Broadcast tracking data: SkillCorner open data (MIT).",
}


class MatchData:
    """A loaded match store plus the time helpers tools need."""

    def __init__(self, store: Store):
        self.store = store
        self.key = f"{store.source}/{store.match_id}"

    @cached_property
    def frame_times(self) -> pd.DataFrame:
        return self.store.sql("SELECT DISTINCT period, frame, t FROM frames ORDER BY period, frame")

    def period_range(self, period: int) -> tuple[float, float]:
        p = self.store.manifest["periods"].get(str(period))
        if p is None:
            raise ToolError(f"{self.key} has no period {period}")
        ft = self.frame_times[self.frame_times["period"] == period]
        return float(ft["t"].min()), float(ft["t"].max())

    def segments(
        self, from_clock: str | None, to_clock: str | None
    ) -> list[tuple[int, float, float]]:
        """(period, t_from, t_to) segments for a clock range; the whole match if both are None."""
        periods = sorted(int(p) for p in self.store.manifest["periods"])
        start = self._resolve(from_clock, periods, default="start")
        end = self._resolve(to_clock, periods, default="end")
        if end <= start:
            raise ToolError(f"empty time range {from_clock!r} to {to_clock!r}")
        out = []
        for p in periods:
            lo, hi = self.period_range(p)
            a = max(lo, start[1]) if start[0] == p else (lo if start[0] < p else None)
            b = min(hi, end[1]) if end[0] == p else (hi if end[0] > p else None)
            if a is not None and b is not None and b > a and start[0] <= p <= end[0]:
                out.append((p, a, b))
        if not out:
            raise ToolError(f"no data between {from_clock!r} and {to_clock!r}")
        return out

    def _resolve(self, clock: str | None, periods: list[int], default: str) -> tuple[int, float]:
        if clock is None:
            p = periods[0] if default == "start" else periods[-1]
            lo, hi = self.period_range(p)
            return (p, lo) if default == "start" else (p, hi)
        try:
            period, seconds = parse_clock(clock)
        except ValueError as e:
            raise ToolError(str(e)) from None
        if period is None:
            period = 1 if seconds < PERIOD_LENGTH_S else 2
            t = seconds if period == 1 else seconds - PERIOD_LENGTH_S
        else:
            t = seconds
        if period not in periods:
            raise ToolError(f"{clock!r} is outside the match (periods {periods})")
        lo, hi = self.period_range(period)
        if not (lo - CLOCK_TOLERANCE_S <= t <= hi + CLOCK_TOLERANCE_S):
            raise ToolError(
                f"{clock!r} is outside the recorded match: period {period} runs "
                f"{match_clock(period, lo)} to {match_clock(period, hi)}"
            )
        return period, t

    def evidence(self, period: int, t0: float, t1: float) -> Evidence:
        ft = self.frame_times[
            (self.frame_times["period"] == period) & self.frame_times["t"].between(t0, t1)
        ]
        if ft.empty:
            raise ToolError(f"no frames in period {period} between {t0:.0f} s and {t1:.0f} s")
        return Evidence(
            match=self.key,
            period=period,
            frame_start=int(ft["frame"].min()),
            frame_end=int(ft["frame"].max()),
            clock_start=match_clock(period, float(ft["t"].min())),
            clock_end=match_clock(period, float(ft["t"].max())),
        )


def _in_segments(df: pd.DataFrame, segs, t_col: str = "t", end_col: str | None = None) -> pd.Series:
    """Rows whose time (or [t_col, end_col) interval) falls in any segment."""
    mask = pd.Series(False, index=df.index)
    for p, a, b in segs:
        if end_col is None:
            mask |= (df["period"] == p) & (df[t_col] >= a) & (df[t_col] <= b)
        else:
            mask |= (df["period"] == p) & (df[end_col] > a) & (df[t_col] < b)
    return mask


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


class Toolbox:
    """All tools, over the stores under ``store_root`` (``<root>/<source>/<match_id>``)."""

    def __init__(self, store_root: Path):
        self.root = Path(store_root)
        self._cache: dict[str, MatchData] = {}

    def matches(self) -> list[str]:
        return sorted(
            f"{p.parent.name}/{p.name}"
            for p in self.root.glob("*/*")
            if (p / "manifest.json").exists()
        )

    def _match(self, match: str) -> MatchData:
        if match not in self._cache:
            path = self.root / match
            if not (path / "manifest.json").exists():
                raise ToolError(f"unknown match {match!r}; available: {self.matches()}")
            self._cache[match] = MatchData(Store(path))
        return self._cache[match]

    # ------------------------------------------------------------------ tools

    def check_capability(self, topic: str) -> Capability:
        """Whether Regista can answer questions about a topic (e.g. 'xG', 'pressing')."""
        t = topic.lower()
        for key, reason in UNSUPPORTED.items():
            if key in t:
                return Capability(available=False, topic=topic, reason=reason)
        return Capability(
            available=True,
            topic=topic,
            reason="Use the analytics tools; if none covers it, it is not available.",
        )

    def get_match_overview(self, match: str) -> Overview:
        """Match periods and clocks, pass counts, detected moments, and typical formations."""
        m = self._match(match)
        periods = []
        for p in sorted(int(k) for k in m.store.manifest["periods"]):
            lo, hi = m.period_range(p)
            periods.append(
                {"period": p, "clock_start": match_clock(p, lo), "clock_end": match_clock(p, hi)}
            )
        passes = m.store.sql(
            "SELECT team, count(*) AS n FROM passes WHERE kind = 'pass' GROUP BY team"
        )
        moments = m.store.sql("SELECT type, count(*) AS n FROM moments GROUP BY type")
        forms = m.store.table("formations").dropna(subset=["label"])
        common = {}
        for team in ("home", "away"):
            common[team] = {}
            for phase in ("in", "out"):
                s = forms[(forms["team"] == team) & (forms["phase"] == phase)]["label"]
                common[team][phase] = s.value_counts().index[0] if len(s) else None
        return Overview(
            match=m.key,
            evidence=[
                m.evidence(p, *m.period_range(p))
                for p in sorted(int(k) for k in m.store.manifest["periods"])
            ],
            source=m.store.source,
            periods=periods,
            frame_rate=float(m.store.manifest["frame_rate"]),
            passes_by_team=dict(zip(passes["team"], passes["n"].astype(int), strict=True)),
            moments_by_type={
                **{k: 0 for k in MomentType.__args__},
                **dict(zip(moments["type"], moments["n"].astype(int), strict=True)),
            },
            most_common_formation=common,
            credits=CREDITS.get(m.store.source, ""),
            notes=[FORMATION_NOTE, ANONYMISED_NOTE, "The score is not modelled."],
        )

    def get_formation(
        self,
        match: str,
        team: Team,
        phase: Phase,
        from_clock: str | None = None,
        to_clock: str | None = None,
    ) -> FormationResult:
        """Formation (the team's shape, system, or setup, e.g. 4-4-2) per 5-minute window.

        For a team and phase in a time range, with runner-up and margins. For line
        height, length, or width use get_team_dimensions.
        """
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        f = m.store.table("formations")
        f = f.assign(t_end=f["t_end"])
        median_margin = float(f["margin"].median())
        sel = f[
            (f["team"] == team) & (f["phase"] == phase) & _in_segments(f, segs, "t_start", "t_end")
        ]
        sel = sel.sort_values(["period", "t_start"])
        windows = []
        for r in sel.itertuples():
            has = isinstance(r.label, str)
            windows.append(
                FormationWindow(
                    period=int(r.period),
                    clock_start=match_clock(r.period, r.t_start),
                    clock_end=match_clock(r.period, r.t_end),
                    label=r.label if has else None,
                    runner_up=r.runner_up if has else None,
                    margin=_f(r.margin) if has else None,
                    relative_margin=_f(r.relative_margin) if has else None,
                    back_line=int(r.label.split("-")[0]) if has else None,
                    close_call=bool(r.margin < median_margin) if has else None,
                )
            )
        labels = pd.Series([w.label for w in windows if w.label])
        return FormationResult(
            match=m.key,
            team=team,
            phase=phase,
            most_common_label=labels.value_counts().index[0] if len(labels) else None,
            most_common_share=_f(labels.value_counts(normalize=True).iloc[0])
            if len(labels)
            else None,
            back_line_counts={
                str(k): int(v)
                for k, v in pd.Series([w.back_line for w in windows if w.back_line])
                .value_counts()
                .items()
            },
            windows=windows,
            match_median_margin=round(median_margin, 4),
            evidence=[
                m.evidence(w_p, a, b)
                for w_p, a, b in [
                    (int(r.period), float(r.t_start), float(r.t_end)) for r in sel.itertuples()
                ]
            ],
            notes=[FORMATION_NOTE],
        )

    def get_team_dimensions(
        self,
        match: str,
        team: Team,
        phase: Phase,
        from_clock: str | None = None,
        to_clock: str | None = None,
    ) -> ShapeResult:
        """Team dimensions in metres (median of 5-minute medians): line height (distance of
        the deepest outfield player from the team's own goal line), length, width.

        Not the formation; for the shape or system (e.g. 4-4-2) use get_formation.
        """
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        s = m.store.table("shape_windows")
        s = s.assign(t_end=s["t_start"] + m.store.manifest["window_s"])
        sel = s[
            (s["team"] == team) & (s["phase"] == phase) & _in_segments(s, segs, "t_start", "t_end")
        ]
        med = sel[["line_height", "length", "width", "hull_area"]].median()
        return ShapeResult(
            match=m.key,
            team=team,
            phase=phase,
            line_height_m=_f(med["line_height"]),
            length_m=_f(med["length"]),
            width_m=_f(med["width"]),
            hull_area_m2=_f(med["hull_area"]),
            windows_used=len(sel),
            evidence=[m.evidence(p, a, b) for p, a, b in segs],
            notes=[
                "Line height is the deepest outfield player's distance from their own goal line."
            ],
        )

    def get_press_stats(
        self,
        match: str,
        pressing_team: Team,
        from_clock: str | None = None,
        to_clock: str | None = None,
    ) -> PressResult:
        """Press intensity: share of opponent-carrier frames with a defender within 5 yd."""
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        p = m.store.table("pressure")
        p = p[(p["pressing_team"] == pressing_team) & _in_segments(p, segs)]
        rows = []
        for third in ("all", "defensive", "middle", "attacking"):
            s = p if third == "all" else p[p["third"] == third]
            n = len(s)
            rows.append(
                PressThird(
                    third=third,
                    carrier_frames=n,
                    press_intensity=_f((s["nearest_defender_m"] <= 4.572).mean()) if n else None,
                    tight_intensity=_f((s["nearest_defender_m"] <= 2.0).mean()) if n else None,
                    mean_defenders_within=_f(s["defenders_within"].mean()) if n else None,
                )
            )
        return PressResult(
            match=m.key,
            pressing_team=pressing_team,
            by_third=rows,
            evidence=[m.evidence(pp, a, b) for pp, a, b in segs],
            notes=[
                "Pressure = defender within 5 yd (4.572 m) of the carrier, fixed radius; "
                "tight = within 2 m. Thirds are from the pressing team's view "
                "(attacking = high press)."
            ],
        )

    def get_pass_network(
        self,
        match: str,
        team: Team,
        from_clock: str | None = None,
        to_clock: str | None = None,
        top_k: int = 10,
    ) -> NetworkResult:
        """Pass counts between teammates (passes inferred from tracking) and mean positions."""
        if not 1 <= top_k <= 50:
            raise ToolError("top_k must be between 1 and 50")
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        passes = m.store.table("passes")
        passes = passes[
            (passes["kind"] == "pass")
            & (passes["team"] == team)
            & _in_segments(passes, segs, "t_start")
        ]
        edges = passes.groupby(["from_player", "to_player"]).size().rename("passes").reset_index()
        edges = edges.sort_values(
            ["passes", "from_player", "to_player"], ascending=[False, True, True]
        )
        nodes = m.store.table("network_nodes")
        nodes = nodes[nodes["team"] == team]
        made = passes.groupby("from_player").size()
        return NetworkResult(
            match=m.key,
            team=team,
            total_passes=len(passes),
            top_edges=[
                Edge(from_player=r.from_player, to_player=r.to_player, passes=int(r.passes))
                for r in edges.head(top_k).itertuples()
            ],
            nodes=[
                Node(
                    player_id=r.player_id,
                    x=round(r.x, 2),
                    y=round(r.y, 2),
                    passes_made=int(made.get(r.player_id, 0)),
                )
                for r in nodes.itertuples()
            ],
            evidence=[m.evidence(p, a, b) for p, a, b in segs],
            notes=[
                "Passes are inferred from tracking (detector v2), not labelled events. "
                "Node positions are full-match means in possession, attacking left to right.",
                ANONYMISED_NOTE,
            ],
        )

    def get_player_passes(
        self, match: str, player_id: str, from_clock: str | None = None, to_clock: str | None = None
    ) -> PlayerPassesResult:
        """A player's pass attempts (completed and failed), receptions, and top receivers."""
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        pm = m.store.table("pass_moments")
        pm = pm[_in_segments(pm, segs)]
        made = pm[pm["from_player"] == player_id]
        if made.empty and not (pm["to_player"] == player_id).any():
            known = sorted(set(pm["from_player"]))
            raise ToolError(f"no passes by or to {player_id!r} in that range; players: {known}")
        team = (
            made["team"].iat[0]
            if len(made)
            else pm.loc[pm["to_player"] == player_id, "team"].iat[0]
        )
        received = pm[(pm["to_player"] == player_id) & (pm["completed"] == 1)]
        comp = made[made["completed"] == 1]
        top = comp.groupby("to_player").size().sort_values(ascending=False).head(5)
        return PlayerPassesResult(
            match=m.key,
            player_id=player_id,
            team=team,
            attempted=len(made),
            completed=len(comp),
            received=len(received),
            top_receivers=[
                Edge(from_player=player_id, to_player=k, passes=int(v)) for k, v in top.items()
            ],
            passes=[
                PassSummary(
                    pass_id=int(r.moment_id),
                    period=int(r.period),
                    clock=match_clock(r.period, r.t),
                    to_player=r.to_player,
                    completed=bool(r.completed),
                )
                for r in made.head(25).itertuples()
            ],
            evidence=[m.evidence(p, a, b) for p, a, b in segs],
            notes=[
                "A failed attempt's receiver is the opponent who won the ball.",
                ANONYMISED_NOTE,
            ],
        )

    def get_team_passing(
        self, match: str, team: Team, from_clock: str | None = None, to_clock: str | None = None
    ) -> TeamPassingResult:
        """A team's players ranked by pass attempts, with completions and receptions."""
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        pm = m.store.table("pass_moments")
        pm = pm[_in_segments(pm, segs) & (pm["team"] == team)]
        made = pm.groupby("from_player").agg(
            attempted=("completed", "size"), completed=("completed", "sum")
        )
        received = pm[pm["completed"] == 1].groupby("to_player").size()
        made = made.sort_values(["attempted", "completed"], ascending=False)
        return TeamPassingResult(
            match=m.key,
            team=team,
            players=[
                PlayerPassing(
                    player_id=pid,
                    attempted=int(r.attempted),
                    completed=int(r.completed),
                    completion_share=round(float(r.completed / r.attempted), 4),
                    received=int(received.get(pid, 0)),
                )
                for pid, r in made.iterrows()
            ],
            evidence=[m.evidence(p, a, b) for p, a, b in segs],
            notes=["Passes are inferred from tracking (detector v2).", ANONYMISED_NOTE],
        )

    def get_passing_options(self, match: str, pass_id: int) -> PassingOptionsResult:
        """At one pass moment: model success estimate for the pass played and every option."""
        m = self._match(match)
        pm = m.store.table("pass_moments")
        row = pm[pm["moment_id"] == pass_id]
        if row.empty:
            raise ToolError(f"unknown pass_id {pass_id}; ids run 0 to {int(pm['moment_id'].max())}")
        r = row.iloc[0]
        opts = m.store.table("pass_options")
        opts = opts[opts["moment_id"] == pass_id].sort_values(
            "display_probability", ascending=False
        )
        return PassingOptionsResult(
            match=m.key,
            pass_id=pass_id,
            period=int(r["period"]),
            clock=match_clock(r["period"], r["t"]),
            team=r["team"],
            from_player=r["from_player"],
            to_player=r["to_player"],
            completed=bool(r["completed"]),
            target_display_probability=_f(r["display_probability"]),
            options=[
                Option(
                    player_id=o.player_id,
                    x=round(o.x, 2),
                    y=round(o.y, 2),
                    pitch_control=_f(o.pitch_control),
                    display_probability=_f(o.display_probability),
                    is_target=bool(o.is_target),
                )
                for o in opts.itertuples()
            ],
            evidence=[m.evidence(int(r["period"]), float(r["t"]), float(r["t"]))],
            notes=[PROBABILITY_NOTE],
        )

    def find_moments(
        self,
        match: str,
        type: str | None = None,
        team: Team | None = None,
        from_clock: str | None = None,
        to_clock: str | None = None,
    ) -> MomentsResult:
        """Detected tactical moments (causal detectors), optionally filtered by type, team, time.

        type: back_line_change, press_change, or line_height_shift (omit for all).
        """
        m = self._match(match)
        segs = m.segments(from_clock, to_clock)
        mo = m.store.table("moments")
        if type:
            mo = mo[mo["type"] == moment_type(type)]
        if team:
            mo = mo[mo["team"] == team]
        mo = mo[_in_segments(mo, segs, "emit_t")].sort_values(["period", "emit_t"])
        items = []
        for r in mo.itertuples():
            ev = json.loads(r.evidence)
            first = ev["run_windows"][0]
            items.append(
                MomentItem(
                    type=r.type,
                    team=r.team,
                    period=int(r.period),
                    emit_clock=match_clock(r.period, r.emit_t),
                    start_clock_estimate=match_clock(r.period, r.start_t),
                    severity=float(r.severity),
                    before=round(float(ev["before"]), 4),
                    after=round(float(ev["after"]), 4),
                    evidence=[m.evidence(int(first[0]), float(first[1]), float(r.emit_t))],
                )
            )
        return MomentsResult(
            match=m.key,
            moments=items,
            evidence=[m.evidence(p, a, b) for p, a, b in segs],
            notes=[
                "Moments come from causal detectors: each used only data up to its emit "
                "time. The start time is an estimate."
            ],
        )


TOOL_NAMES = (
    "check_capability",
    "get_match_overview",
    "get_formation",
    "get_team_dimensions",
    "get_press_stats",
    "get_pass_network",
    "get_team_passing",
    "get_player_passes",
    "get_passing_options",
    "find_moments",
)
