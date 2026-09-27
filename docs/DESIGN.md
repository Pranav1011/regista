# Design decisions

Short architecture decision records (ADRs): context, decision, consequence.
Every number here comes from a script in `eval/`; the script is named next to it.

## ADR-001: Drop out-of-bounds rows at ingest; keep the 5 m schema margin

**Context.** `schema.validate_frames` rejects positions more than 5 m outside
the pitch lines. Provider data occasionally records real positions beyond that
(the ball behind the goal after a shot, players past the line). In Metrica games
1-3 this is 9 / 75 / 3 rows out of about 3.2 M per game; in SkillCorner it is
tens to a few hundred rows per match.

**Decision.** Keep the margin at 5 m and drop the offending rows at ingest.
Counts are recorded per match, split into ball and player rows, in
`data/processed/<source>/<match>_ingest.json`, and printed by `regista ingest`.

**Consequence.** The margin stays a meaningful filter for Phase 3, where
off-pitch detections (spectators, staff, reflections) must be rejected. The
cost is a handful of lost rows per match, which is disclosed rather than hidden.

## ADR-002: Attacking direction from goalkeepers, not kick-off positions

**Context.** Canonical frames require the home team to attack +x in every
period. The first implementation read each team's median x at kick-off. On
SkillCorner broadcast tracking this failed for match 1925299: at kick-off some
off-camera players are extrapolated across the halfway line, and the home
median was 0.96 m, too close to call.

**Decision.** Per period, take each team's deepest regular player (appearing in
at least half the frames) as its goalkeeper, and use the sign of their mean x.
Ingest raises if the two goalkeepers are not on opposite sides at least 20 m
from halfway.

**Consequence.** All 20 SkillCorner open matches and all 3 Metrica games ingest,
and Metrica's flips are unchanged from the kick-off method. Side agreement with
SkillCorner's listed positions is at least 0.670 in every half (median 0.820),
with no half near zero, so no half is flipped (`eval/formations_eval.py`).

## ADR-003: Pass detector v2 (per-match kick-moment radius), frozen before evaluation

**Context.** v1 owns the ball to the nearest player within a fixed radius tuned
on Metrica games 1-2 (0.5 m). On held-out game 3 it scored F1 0.801 against
0.914 on training. The failure analysis showed game 3 records the ball further
from the passer at release: more than 0.5 m in 10.0% of labelled releases,
against 2.5-3.3% in games 1-2.

**Decision.** v2 derives the ownership radius per match from unlabelled
tracking: the 0.95 quantile of ball-to-nearest-player distance just before each
kick (ball speed crossing 3 m/s upwards and reaching 7 m/s within 0.4 s). A
release gate requires the ball to accelerate away from the passer, and the pass
starts at that frame. A first idea (a quantile of distances in slow-ball frames)
was rejected on training data alone: it gave 0.91 m for game 1 and 0.068 m for
game 2. v2 was tuned on games 1-2, committed with its parameters frozen
(commit `8ce2eb6`), and only then evaluated on game 3, once.

**Consequence.** v2 scores F1 0.809 on game 3 against v1's 0.801; the gain comes
mostly from release timing, and may be within noise. v2 was designed after
seeing v1's game-3 failures, which the report states. Both versions stay
reproducible (`eval/pass_detection.py --version v1|v2`, parameters in
`eval/params_phase1.json`).

## ADR-004: Template labels are noisy at 5-minute windows; role groups and sides are the output

**Context.** Formations are detected per team, phase, and 5-minute window by
matching the 10 outfield players' mean shape to seven templates. There is no
formation ground truth, so stability is measured without labels, and roles are
checked against SkillCorner's listed positions on all 20 open matches
(16,580 player-windows; `eval/formations_eval.py`). Every method below is
scored on the same player-windows. Confidence intervals come from resampling
matches (2,000 resamples), since windows within a match are not independent.

- **Groups (DEF / MID / FWD): templates about tie a depth-rank baseline** (4
  deepest DEF, next 4 MID, 2 highest FWD). Strict agreement is 0.731 vs 0.724,
  a difference of +0.007 (95% CI -0.006 to +0.020); with wingers allowed as MID
  or FWD it is 0.868 vs 0.855, +0.013 (CI +0.001 to +0.026). Out of possession
  the baseline is slightly ahead.
- **Templates add value on midfield roles and sides.** For players listed as
  midfielders, templates agree 0.889 vs 0.764 for depth rank, +0.125 (CI +0.104
  to +0.148); depth rank is better for listed forwards (0.537 vs 0.450 strict).
  Template sides (left / centre / right) agree 0.819 vs 0.685 for splitting the
  width into thirds, +0.133 (CI +0.106 to +0.160).
- **Exact labels are unstable.** The template label repeats in consecutive
  windows only 26-63% of the time per Metrica team and phase (median 0.50 on
  SkillCorner). The back-line count holds better: median 0.684 in possession
  and 0.800 out of possession on SkillCorner. Players keep their position group
  in 86-100% of windows (Metrica medians) but their exact role in only 40-76%.

**Decision.** Treat role groups and sides as the trusted output. Show template
labels only with their margin to the runner-up, the primary ambiguity signal.
Keep `relative_margin` (margin divided by the runner-up cost) as a secondary
number; it is not a probability and must never be presented as one.

**Consequence.** For DEF / MID / FWD alone, depth ranking is as good and far
simpler; the templates earn their place through sides and midfield structure.
Anything downstream that needs a formation works from persistent group-level
structure, not a single window's label.

### Phase 2 note: formation-change alerts

Alerts must be driven by persistent group-level changes, such as the back line
going from four to five, held over consecutive windows. They must not be driven
by raw template flips: the exact label changes in about half of consecutive
windows (label stability median 0.50 on SkillCorner). Back-line alerts use
out-of-possession windows only, where the back-line count is stable in 0.800 of
consecutive windows against 0.684 in possession. An alert on every template flip
would be mostly noise.
