# Claude Code kickoff: Phase 1

Paste everything below the line into Claude Code from the repo root.
Start in plan mode.

---

Read `CLAUDE.md`, `docs/ROADMAP.md`, `THIRD_PARTY_NOTICES.md`, and
`src/regista/schema.py` first. They are binding. We are doing **Phase 1 only**:
tactical analytics on open tracking data. No computer vision, no LLM, no web UI.

Propose a plan before writing code, and wait for my approval. Then implement
step by step, committing after each step. Stop and report to me at the two
checkpoints marked below.

## Step 0: Project hygiene
- `uv sync --group dev`; confirm `uv run pytest` passes on the existing schema tests.
- Add a `regista` CLI with `typer` (`src/regista/cli.py`) and register it in
  `pyproject.toml` under `[project.scripts]`.
- Before using `kloppy`, `mplsoccer`, and `supervision`, read their LICENSE
  files, then update the "status" column in `THIRD_PARTY_NOTICES.md`.

## Step 1: Ingest (`src/regista/ingest/`)
- `metrica.py` and `skillcorner.py`: use kloppy's open-data loaders, convert to
  the canonical table with a `to_canonical()` function, and normalise direction
  so home attacks +x in both periods.
- Metres on 105 x 68, pitch-centre origin, `confidence=1.0`, correct `source`.
- Call `validate_frames()` and write to `data/processed/<source>/<match_id>.parquet`.
- CLI: `regista ingest metrica --game 1`, `regista ingest skillcorner --match <id>`.
- Tests: the direction-normalisation and coordinate-conversion functions get
  known-answer tests on tiny synthetic inputs.

## Step 2: Kinematics (`analytics/kinematics.py`)
- Velocities via Savitzky–Golay smoothing per player; mask speeds above
  12 m/s as NaN (not clipped).
- If you adapt anything from LaurieOnTracking (MIT), follow the license policy.

## Step 3: Possession and passes (`analytics/possession.py`)
- Frame-level ball owner: nearest player within radius r, gated by ball speed.
- Pass = owner change between teammates. Owner change to an opponent = turnover.
- Validate against Metrica PASS events, allowing a +/- k frame tolerance.
  Tune r and k on games 1–2 only. Report precision, recall, and F1 on game 3.

**Checkpoint A:** stop here and show me the pass-detection numbers plus the
three most common failure cases, with frame ranges.

## Step 4: Formations and roles (`analytics/formations.py`)
- For each team and rolling window (default 5 min), handle in and out of
  possession separately.
- Use mean positions relative to the team centroid, scaled to a unit shape.
- Template library: 4-4-2, 4-3-3, 4-2-3-1, 4-1-4-1, 3-5-2, 3-4-3, 5-3-2.
- Assign players to template slots with `scipy.optimize.linear_sum_assignment`.
  Output the formation label, each player's role, and a cost-based confidence.
- Change point = label differs for 2 or more consecutive windows.
- Validation: compare role assignment with SkillCorner's listed positions at
  the position-group level. Metrica is anonymised and has no formation labels,
  so state that plainly in the report. Do not invent ground truth.
- Tests: synthetic noisy formations must be recovered correctly.

## Step 5: Shape and pass networks
- `analytics/shape.py`: defensive line height, team length, width, convex hull
  area, and centroid per frame, with windowed summaries.
- `analytics/pass_network.py`: node = mean position while in possession;
  edge = pass count. Build one from inferred passes and one from Metrica
  events, and report how similar they are.

## Step 6: Passing options (`analytics/passing_options.py`)
- Pitch-control model (Spearman), adapting LaurieOnTracking (MIT) per the
  license policy. For each pass moment, compute the success probability of a
  pass to every teammate.
- Calibration: on game 3 only, bin predicted probabilities for the passes that
  were actually played and compare with real completed vs failed outcomes.
  Report a reliability diagram and Brier score.
- Tests: synthetic scenes with an obviously open and an obviously marked
  teammate must rank correctly.

## Step 7: Evaluation report and plots
- `eval/phase1.py` regenerates `eval/reports/phase1.md` end to end, covering
  datasets used, method summary, metrics tables, known limitations, and credits.
- Plots with `mplsoccer` (formation snapshot, pass network, passing options, reliability diagram) go to `outputs/`
  (gitignored). Small, non-NDA plots may be committed under `docs/img/`.

**Checkpoint B:** show me the report before updating the README.

## Acceptance criteria
- CI is green (ruff check, ruff format --check, pytest).
- Every analytics function has a known-answer test.
- One command reproduces every reported number.
- No data, video, or weights are committed.
- `THIRD_PARTY_NOTICES.md` is up to date.
