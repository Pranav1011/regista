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
matching the 10 outfield players' mean shape to seven templates. With no
formation ground truth, stability is measured without labels, and roles are
checked against SkillCorner's listed positions (`eval/formations_eval.py`).

- The exact template label repeats in consecutive windows only 26-63% of the
  time per Metrica team and phase (median 0.50 on SkillCorner). The back-line
  count is steadier: median 0.684 in possession and 0.800 out of possession on
  SkillCorner, and 0.37-1.00 per Metrica team and phase.
- Players keep their position group (DEF / MID / FWD) in 86-100% of windows
  (Metrica medians) but their exact role in only 40-76%.
- Against listed positions, template groups agree 0.731 (strict) and 0.868
  (wingers count as MID or FWD), barely above a depth-rank baseline (4 deepest
  DEF, next 4 MID, 2 highest FWD) at 0.724 and 0.855; out of possession the
  baseline is slightly better. Template sides agree 0.819, well above a
  width-thirds baseline at 0.685.

**Decision.** Treat role groups and sides as the trusted output. Show template
labels only with their margin to the runner-up (the primary ambiguity signal)
and never as a certainty. `relative_margin` (margin / runner-up cost) is not a
probability and must not be presented as one.

**Consequence.** Group assignment by templates adds little over depth ranking;
their value is in side (left / centre / right) and midfield structure. Anything
downstream that needs a formation should work from persistent group-level
structure, not a single window's label.

### Phase 2 note: formation-change alerts

Alerts must be driven by persistent group-level changes, such as the back line
going from four to five, held over consecutive windows. They must not be driven
by raw template flips: the exact label changes in about half of consecutive
windows (label stability median 0.50 on SkillCorner), while the
back line holds far more often (0.684 in possession, 0.800 out of possession, on
SkillCorner). An alert on every template flip would be mostly noise.
