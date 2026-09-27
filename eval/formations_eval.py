"""Formation diagnostics: label-free stability on Metrica, role checks on SkillCorner.

Metrica is anonymised and has no formation or position labels, so formations
cannot be validated there. For Metrica we report label-free diagnostics only:
the runner-up template and cost margin per window, the share of consecutive
windows with an unchanged label, per-player role consistency, and how close
the away team's 3-5-2 windows are to 5-3-2. SkillCorner lists one position per player (mostly
starters). We compare, per player and window, the detected role's position
group (DEF / MID / FWD) and side (left / centre / right) with the listed one.

Wide players are genuinely ambiguous: a winger is a forward in a 4-3-3 but a
midfielder in a 4-4-2, and a wing-back a defender in a 5-3-2 but a midfielder
in a 3-5-2. "Strict" uses one group per listed position; "lenient" accepts
either group for those ambiguous positions.

Run: uv run python eval/formations_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import (
    SKILLCORNER_MATCHES,
    TEST_GAME,
    TRAIN_GAMES,
    load_params,
    metrica_game,
    possession_phase,
    skillcorner_match,
)

from regista.analytics.formations import (
    back_line,
    detect_formations,
    label_stability,
    role_consistency,
    window_shapes,
)

METRICA_GAMES = (*TRAIN_GAMES, TEST_GAME)

WINDOW_S = 300.0

# kloppy PositionType name -> (strict group, lenient groups)
_GROUPS: dict[str, tuple[str, frozenset[str]]] = {}
for _names, _group, _lenient in (
    (["Defender", "FullBack", "LeftBack", "RightBack", "CenterBack", "LeftCenterBack",
      "RightCenterBack"], "DEF", {"DEF"}),
    (["LeftWingBack", "RightWingBack"], "DEF", {"DEF", "MID"}),
    (["Midfielder", "DefensiveMidfield", "LeftDefensiveMidfield", "CenterDefensiveMidfield",
      "RightDefensiveMidfield", "CentralMidfield", "LeftCentralMidfield", "CenterMidfield",
      "RightCentralMidfield", "AttackingMidfield", "LeftAttackingMidfield",
      "CenterAttackingMidfield", "RightAttackingMidfield"], "MID", {"MID"}),
    (["WideMidfield", "LeftMidfield", "RightMidfield"], "MID", {"MID", "FWD"}),
    (["LeftWing", "RightWing"], "FWD", {"MID", "FWD"}),
    (["Attacker", "LeftForward", "RightForward", "Striker"], "FWD", {"FWD"}),
):  # fmt: skip
    for _name in _names:
        _GROUPS[_name] = (_group, frozenset(_lenient))

_NO_SIDE = {"Defender", "FullBack", "Midfielder", "WideMidfield", "Attacker"}


def listed_side(position: str) -> str | None:
    if position in _NO_SIDE:
        return None
    if position.startswith("Left"):
        return "L"
    if position.startswith("Right"):
        return "R"
    return "C"


def role_side(role: str) -> str:
    return role[0] if role[0] in "LR" else "C"


WINDOW_KEYS = ["match_id", "period", "team", "phase", "window"]
METHODS = ("template", "depth")  # group methods; side: template vs thirds
SIDE_OUTLIER = 0.5  # a flipped half would score near 0; chance level is about 1/3


def depth_baseline(shapes: pd.DataFrame) -> pd.Series:
    """Group by depth rank within each window: 4 deepest DEF, next 4 MID, 2 highest FWD."""
    rank = shapes.groupby(WINDOW_KEYS)["x"].rank(method="first").astype(int)
    return pd.Series(np.select([rank <= 4, rank <= 8], ["DEF", "MID"], "FWD"), index=shapes.index)


def thirds_baseline(shapes: pd.DataFrame) -> pd.Series:
    """Side by splitting each window's width (y range) into equal thirds; +y is left."""
    g = shapes.groupby(WINDOW_KEYS)["y"]
    lo, hi = g.transform("min"), g.transform("max")
    third = (hi - lo) / 3
    side = np.select([shapes["y"] >= hi - third, shapes["y"] <= lo + third], ["L", "R"], "C")
    return pd.Series(side, index=shapes.index)


def match_roles(match_id: str, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(formations, per player-window predictions joined with listed positions).

    Template roles and both baselines are computed on the same player-windows,
    so every method is scored on identical rows.
    """
    frames, players = skillcorner_match(match_id)
    shapes = window_shapes(frames, possession_phase(frames, params), WINDOW_S, WINDOW_S)
    formations, roles = detect_formations(shapes)
    shapes = shapes.merge(roles[[*WINDOW_KEYS, "player_id", "role", "group"]],
                          on=[*WINDOW_KEYS, "player_id"])  # fmt: skip
    shapes["group_template"] = shapes.pop("group")
    shapes["group_depth"] = depth_baseline(shapes)
    shapes["side_template"] = shapes["role"].map(role_side)
    shapes["side_thirds"] = thirds_baseline(shapes)

    listed = players[players["position"].isin(list(_GROUPS))][["player_id", "position"]]
    joined = shapes.merge(listed, on="player_id", how="inner")
    joined["listed_group"] = joined["position"].map(lambda p: _GROUPS[p][0])
    lenient = joined["position"].map(lambda p: _GROUPS[p][1])
    for method in METHODS:
        pred = joined[f"group_{method}"]
        joined[f"{method}_strict"] = pred == joined["listed_group"]
        joined[f"{method}_lenient"] = [g in ok for g, ok in zip(pred, lenient, strict=True)]
    joined["listed_side"] = joined["position"].map(listed_side)
    for method in ("template", "thirds"):
        ok = joined[f"side_{method}"] == joined["listed_side"]
        joined[f"side_{method}_ok"] = ok.where(joined["listed_side"].notna())
    return formations, joined


def metrica_diagnostics(params: dict) -> dict:
    """Label-free formation diagnostics on the three Metrica games."""
    forms, roles = [], []
    for game in METRICA_GAMES:
        frames, _ = metrica_game(game)
        f, r = detect_formations(window_shapes(frames, possession_phase(frames, params),
                                               WINDOW_S, WINDOW_S))  # fmt: skip
        forms.append(f)
        roles.append(r)
    formations = pd.concat(forms, ignore_index=True)
    roles = pd.concat(roles, ignore_index=True)
    labelled = formations.dropna(subset=["label"])

    # How close is each 3-5-2 window to 5-3-2, whether or not 5-3-2 was the runner-up?
    three_five_two = labelled[labelled["label"] == "3-5-2"].copy()
    three_five_two["gap_to_5-3-2"] = three_five_two["cost_5-3-2"] - three_five_two["cost"]
    all_margin_median = float(labelled["margin"].median())
    away_352 = three_five_two[three_five_two["team"] == "away"]
    flips = changes = 0
    for _, g in labelled[labelled["team"] == "away"].groupby(["match_id", "phase"]):
        seq = g.sort_values(["period", "window"])["label"].tolist()
        for a, b in zip(seq, seq[1:], strict=False):
            changes += a != b
            flips += {a, b} == {"3-5-2", "5-3-2"}
    return {
        "away_352_windows": len(away_352),
        "away_352_narrow": int((away_352["gap_to_5-3-2"] < all_margin_median).sum()),
        "away_label_changes": changes,
        "away_352_532_flips": flips,
        "formations": formations,
        "roles": roles,
        "margin_quantiles": labelled["margin"].quantile([0.1, 0.25, 0.5, 0.75]),
        "relative_margin_median": float(labelled["relative_margin"].median()),
        "label_counts": labelled.groupby(["match_id", "team", "phase"])["label"]
        .value_counts()
        .rename("windows"),
        "runner_up": pd.crosstab(
            [labelled["match_id"], labelled["team"], labelled["label"]], labelled["runner_up"]
        ),
        "stability": label_stability(formations),
        "back_line_stability": label_stability(formations, key=back_line),
        "role_consistency": role_consistency(roles)
        .groupby(["match_id", "team", "phase"])[["role_consistency", "group_consistency"]]
        .median(),
        "three_five_two": three_five_two[
            [
                "match_id",
                "team",
                "phase",
                "period",
                "t_start",
                "runner_up",
                "margin",
                "gap_to_5-3-2",
            ]
        ],  # fmt: skip
        "all_margin_median": all_margin_median,
    }


def _agreement(df: pd.DataFrame) -> dict:
    def mean(col: str) -> float:
        return float(df[col].dropna().astype(bool).mean())

    return {
        "player_windows": len(df),
        **{f"group_{m}_{k}": mean(f"{m}_{k}") for m in METHODS for k in ("strict", "lenient")},
        "side_template": mean("side_template_ok"),
        "side_thirds": mean("side_thirds_ok"),
    }


def skillcorner_roles(params: dict) -> dict:
    """Detected roles vs listed positions across all SkillCorner open matches."""
    all_forms, all_roles = [], []
    for m in SKILLCORNER_MATCHES:
        forms, roles = match_roles(m, params)
        all_forms.append(forms)
        all_roles.append(roles)
    formations = pd.concat(all_forms, ignore_index=True)
    roles = pd.concat(all_roles, ignore_index=True)

    side_by_half = (
        roles.dropna(subset=["side_template_ok"])
        .groupby(["match_id", "period"])[["side_template_ok", "side_thirds_ok"]]
        .agg(lambda s: s.astype(bool).mean())
    )
    side_by_half["n"] = (
        roles.dropna(subset=["side_template_ok"]).groupby(["match_id", "period"]).size()
    )
    table = pd.DataFrame(
        {"all": _agreement(roles), **{p: _agreement(g) for p, g in roles.groupby("phase")}}
    ).T
    return {
        "matches": len(SKILLCORNER_MATCHES),
        "agreement": table,
        "by_listed_group": roles.groupby("listed_group")[
            [f"{m}_{k}" for m in METHODS for k in ("strict", "lenient")]
        ].mean(),
        "confusion": pd.crosstab(roles["listed_group"], roles["group_template"]),
        "side_by_half": side_by_half,
        "side_outliers": side_by_half[side_by_half["side_template_ok"] < SIDE_OUTLIER],
        "stability": label_stability(formations),
        "back_line_stability": label_stability(formations, key=back_line),
        "formations": formations,
        "roles": roles,
    }


def main() -> None:
    params = load_params()
    m = metrica_diagnostics(params)
    print("## Metrica (label-free)")
    print("margin quantiles:", m["margin_quantiles"].round(3).to_dict())
    print("relative margin median (not a probability):", round(m["relative_margin_median"], 3))
    stab = m["stability"].merge(
        m["back_line_stability"], on=["match_id", "team", "phase"], suffixes=("", "_back_line")
    )
    print(stab.round(2).to_markdown(index=False))
    print(m["role_consistency"].round(2).to_markdown())
    print(m["runner_up"].to_markdown())
    t = m["three_five_two"]
    print(f"3-5-2 windows: {len(t)}, gap to 5-3-2 quantiles:",
          t["gap_to_5-3-2"].quantile([0.25, 0.5, 0.75]).round(3).to_dict(),
          f"(all-window margin median {m['all_margin_median']:.3f})")  # fmt: skip
    print(t.groupby(["match_id", "team"])["gap_to_5-3-2"].describe().round(3).to_markdown())
    print(
        f"away 3-5-2 windows closer to 5-3-2 than the all-window median margin: "
        f"{m['away_352_narrow']} of {m['away_352_windows']}; "
        f"away label changes that are 3-5-2<->5-3-2: "
        f"{m['away_352_532_flips']} of {m['away_label_changes']}"
    )

    r = skillcorner_roles(params)
    print(f"\n## SkillCorner roles vs listed positions ({r['matches']} matches)")
    print(r["agreement"].round(3).to_markdown())
    print(r["by_listed_group"].round(3).to_markdown())
    print(r["confusion"].to_markdown())
    sh = r["side_by_half"]
    side = sh["side_template_ok"]
    print(f"side agreement per half: min {side.min():.3f}, median {side.median():.3f}")
    by_period = sh.groupby(level="period")[["side_template_ok", "side_thirds_ok"]].median()
    print("median side agreement by period:", by_period.round(3).to_dict("index"))
    print(f"halves below {SIDE_OUTLIER}:", "none" if r["side_outliers"].empty else "")
    if not r["side_outliers"].empty:
        print(r["side_outliers"].round(3).to_markdown())
    print(sh.round(3).to_markdown())
    for name in ("stability", "back_line_stability"):
        st = r[name].groupby("phase")["stability"].median()
        print(f"{name} median by phase:", st.round(3).to_dict())


if __name__ == "__main__":
    main()
